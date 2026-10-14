"""Stage 3 tree builder decomposing biconnected blocks via SPQR triconnectivity."""

from __future__ import annotations

import collections
from typing import Any, Sequence, cast

import spqrtree

from romutil.layout.hierarchy.block_cut import BlockCutTreeBuilder
from romutil.layout.ir import (
    CutType,
    LayoutEdge,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.layout.protocols import TreeBuilder
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Direction
from romutil.types import LayoutOptions


def _collect_all_spqr_nodes(root: spqrtree.SPQRNode) -> list[spqrtree.SPQRNode]:
    """Collect all nodes in an SPQR tree via depth-first traversal."""
    nodes: list[spqrtree.SPQRNode] = []

    def _visit(node: spqrtree.SPQRNode) -> None:
        nodes.append(node)
        for child in node.children:
            _visit(child)

    _visit(root)
    return nodes


def _reroot_spqr_tree_to_primary_cluster(root: spqrtree.SPQRNode) -> spqrtree.SPQRNode:
    """Reroot SPQR tree so that the primary rigid cluster (R-node) forms the tree root."""
    all_nodes = _collect_all_spqr_nodes(root)
    r_nodes = [n for n in all_nodes if n.type == spqrtree.NodeType.R]
    if not r_nodes:
        return root

    # If the root is a P-node, only consider R-nodes that are external parent clusters
    # (i.e. whose boundary poles differ from the root P-node poles).
    if root.type == spqrtree.NodeType.P:
        root_poles = set(root.poles)
        external_r_nodes = [n for n in r_nodes if set(n.poles) != root_poles]
        if not external_r_nodes:
            return root
        candidates = external_r_nodes
    else:
        candidates = r_nodes

    def _node_rank(n: spqrtree.SPQRNode) -> tuple[int, tuple[int, ...]]:
        verts = tuple(sorted(cast(int, v) for v in (n.skeleton.vertices if n.skeleton else n.poles)))
        return (len(verts), verts)

    target_root = max(candidates, key=_node_rank)
    if target_root is root:
        return root

    node_by_id = {id(n): n for n in all_nodes}
    adj: dict[int, list[int]] = {id(n): [] for n in all_nodes}
    for n in all_nodes:
        if n.parent is not None:
            adj[id(n)].append(id(n.parent))
        for child in n.children:
            adj[id(n)].append(id(child))

    visited: set[int] = {id(target_root)}
    target_root.parent = None
    target_root.children = []
    queue: list[int] = [id(target_root)]

    while queue:
        curr_id = queue.pop(0)
        curr = node_by_id[curr_id]
        for neighbor_id in adj[curr_id]:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                neighbor = node_by_id[neighbor_id]
                neighbor.parent = curr
                neighbor.children = []
                curr.children.append(neighbor)
                queue.append(neighbor_id)

    return target_root


def _get_all_spqr_vertices(node: spqrtree.SPQRNode) -> set[int]:
    """Collect all vertices across an SPQR component and its subtree."""
    source = node.skeleton.vertices if node.skeleton else node.poles
    vertices = set(cast(int, v) for v in source)
    for child in node.children:
        vertices.update(_get_all_spqr_vertices(child))
    return vertices


def _is_safe_2cut_allowlist(
    spqr_p_node: spqrtree.SPQRNode,
    poles: tuple[int, int],
    edges: Sequence[LayoutEdge],
) -> bool:
    """Evaluate whether candidate branches of a P-node satisfy the Safe 2-Cut Allowlist Contract.

    Criteria:
      1. Elasticity: All candidate branches must be elastic series loops without internal
         biconnected rigid blocks or cross-cycles.
      2. Single Port per Pole: Each candidate branch connects to pole u through exactly
         one exit direction and pole v through exactly one exit direction.
      3. Disjoint Half-Spaces: Candidate branches expand into disjoint half-spaces relative
         to the inter-pole axis without transverse bracketing.
    """
    if len(poles) != 2:
        return False
    u, v = poles

    candidate_branches = spqr_p_node.children
    if not candidate_branches:
        return False

    # Check 1 (Elasticity): All candidate branches must be elastic series loops
    # without internal biconnected rigid blocks or cross-cycles.
    for branch in candidate_branches:
        if branch.type != spqrtree.NodeType.S:
            return False
        if branch.children:
            return False

    branch_paths: list[dict[int, tuple[int, int, int]]] = []
    branch_exit_dirs_u: list[Direction] = []
    branch_exit_dirs_v: list[Direction] = []

    for branch in candidate_branches:
        branch_verts = set(cast(int, x) for x in (branch.skeleton.vertices if branch.skeleton else branch.poles))
        internal_rooms = branch_verts - {u, v}
        if not internal_rooms:
            return False

        # Check 2 (Single Port per Pole): Each candidate branch connects to pole u
        # through exactly one exit direction and pole v through exactly one exit direction.
        u_dirs: set[Direction] = set()
        v_dirs: set[Direction] = set()
        for e in edges:
            if e.cut:
                continue
            if e.src == u and e.dst in internal_rooms:
                u_dirs.add(Direction(e.direction))
            elif e.dst == u and e.src in internal_rooms:
                u_dirs.add(Direction(e.direction).invert())

            if e.src == v and e.dst in internal_rooms:
                v_dirs.add(Direction(e.direction))
            elif e.dst == v and e.src in internal_rooms:
                v_dirs.add(Direction(e.direction).invert())

        if len(u_dirs) != 1 or len(v_dirs) != 1:
            return False

        dir_u = next(iter(u_dirs))
        dir_v = next(iter(v_dirs))
        branch_exit_dirs_u.append(dir_u)
        branch_exit_dirs_v.append(dir_v)

        # Check elasticity / simple path structure by tracing from u to v
        local_edges: dict[int, set[tuple[int, Direction]]] = {r: set() for r in branch_verts}
        for e in edges:
            if e.cut:
                continue
            if e.src in branch_verts and e.dst in branch_verts and {e.src, e.dst} != {u, v}:
                local_edges[e.src].add((e.dst, Direction(e.direction)))
                local_edges[e.dst].add((e.src, Direction(e.direction).invert()))

        visited: set[int] = {u}
        coords: dict[int, tuple[int, int, int]] = {u: (0, 0, 0)}

        first_step = {nxt for nxt, d in local_edges[u] if nxt in internal_rooms and d == dir_u}
        if len(first_step) != 1:
            return False
        curr = next(iter(first_step))
        disp = direction_to_displacement(dir_u)
        coords[curr] = disp
        visited.add(curr)

        while curr != v:
            next_steps = {nxt for nxt, _ in local_edges[curr] if nxt not in visited and nxt in branch_verts}
            if len(next_steps) != 1:
                return False
            nxt = next(iter(next_steps))
            step_dirs = {d for n, d in local_edges[curr] if n == nxt}
            if len(step_dirs) != 1:
                return False
            step_dir = next(iter(step_dirs))
            s_disp = direction_to_displacement(step_dir)
            c_curr = coords[curr]
            coords[nxt] = (c_curr[0] + s_disp[0], c_curr[1] + s_disp[1], c_curr[2] + s_disp[2])
            visited.add(nxt)
            curr = nxt

        if visited != branch_verts:
            return False

        branch_paths.append(coords)

    # Check 3 (Disjoint Half-Spaces): Candidate branches expand into disjoint half-spaces
    # relative to the inter-pole axis without transverse bracketing.
    direct_edges = [e for e in edges if not e.cut and {e.src, e.dst} == {u, v}]
    inter_axis: int | None = None
    if direct_edges:
        de = direct_edges[0]
        d_dir = Direction(de.direction) if de.src == u else Direction(de.direction).invert()
        disp_de = direction_to_displacement(d_dir)
        for ax in (0, 1, 2):
            if disp_de[ax] != 0:
                inter_axis = ax
                break

    if inter_axis is None:
        candidate_axes: set[int] = set()
        for bp in branch_paths:
            v_coord = bp[v]
            max_d = max(abs(v_coord[0]), abs(v_coord[1]), abs(v_coord[2]))
            if max_d > 0:
                ax = 0 if abs(v_coord[0]) == max_d else (1 if abs(v_coord[1]) == max_d else 2)
                candidate_axes.add(ax)
        if len(candidate_axes) == 1:
            inter_axis = next(iter(candidate_axes))
        else:
            return False

    transverse_axes = [ax for ax in (0, 1, 2) if ax != inter_axis]

    branch_signatures: list[tuple[int, ...]] = []
    for i, bp in enumerate(branch_paths):
        b_rooms = set(bp.keys())
        sig_list: list[int] = []
        for tax in transverse_axes:
            vals = [bp[r][tax] for r in b_rooms]
            min_val = min(vals)
            max_val = max(vals)
            # Transverse bracketing: expands on both negative and positive sides of the axis
            if min_val < 0 and max_val > 0:
                return False

            u_step = direction_to_displacement(branch_exit_dirs_u[i])[tax]
            v_step = direction_to_displacement(branch_exit_dirs_v[i])[tax]
            if (u_step < 0 and v_step > 0) or (u_step > 0 and v_step < 0):
                return False

            if max_val > 0 or u_step > 0 or v_step > 0:
                sig_list.append(1)
            elif min_val < 0 or u_step < 0 or v_step < 0:
                sig_list.append(-1)
            else:
                sig_list.append(0)

        branch_signatures.append(tuple(sig_list))

    # At most one branch can have signature (0, 0)
    zero_count = sum(1 for sig in branch_signatures if all(s == 0 for s in sig))
    if zero_count > 1:
        return False

    # Disjoint non-zero half-spaces
    non_zero_sigs = [sig for sig in branch_signatures if any(s != 0 for s in sig)]
    if len(non_zero_sigs) != len(set(non_zero_sigs)):
        return False

    return True


def build_block_spqr_tree(
    block_nodes: frozenset[int],
    edges: Sequence[LayoutEdge],
    node_id_prefix: str = "spqr",
) -> SolveTreeNode:
    """Decompose a single biconnected block into its canonical SPQR tree."""
    if not block_nodes:
        return SolveTreeNode(
            node_id=f"{node_id_prefix}_0",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset(),
        )

    multigraph = spqrtree.MultiGraph()
    for v in block_nodes:
        multigraph.add_vertex(v)

    seen_edges: set[tuple[int, int]] = set()
    for edge in edges:
        if not edge.cut and edge.src in block_nodes and edge.dst in block_nodes:
            u, v = edge.src, edge.dst
            if u == v:
                continue
            edge_key = (min(u, v), max(u, v))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                multigraph.add_edge(u, v)

    counter = 0

    def _convert(spqr_node: spqrtree.SPQRNode, parent: spqrtree.SPQRNode | None = None) -> SolveTreeNode:
        nonlocal counter
        node_id = f"{node_id_prefix}_{counter}"
        counter += 1

        def _get_vertices(node: spqrtree.SPQRNode) -> tuple[int, ...]:
            source = node.skeleton.vertices if node.skeleton else node.poles
            return tuple(sorted(cast(int, v) for v in source))

        sorted_children = sorted(
            spqr_node.children,
            key=lambda c: (
                c.type.value,
                _get_vertices(c),
            ),
        )

        # Determine attachment poles to parent component
        if parent is not None:
            parent_v = set(parent.skeleton.vertices if parent.skeleton else parent.poles)
            child_v = set(spqr_node.skeleton.vertices if spqr_node.skeleton else spqr_node.poles)
            shared = parent_v & child_v
            if len(shared) == 2:
                attachment_poles = tuple(sorted(cast(int, v) for v in shared))
            elif spqr_node.poles:
                attachment_poles = tuple(cast(int, v) for v in spqr_node.poles)
            else:
                attachment_poles = ()
        elif spqr_node.poles:
            attachment_poles = tuple(cast(int, v) for v in spqr_node.poles)
        else:
            attachment_poles = ()

        rooms: frozenset[int]
        metadata: dict[str, Any] = {}
        if spqr_node.type == spqrtree.NodeType.P:
            raw_poles = (
                attachment_poles
                if len(attachment_poles) == 2
                else tuple(cast(int, v) for v in spqr_node.poles)
            )
            poles_to_use = tuple(sorted(raw_poles)) if len(raw_poles) == 2 else raw_poles
            is_safe = len(poles_to_use) == 2 and _is_safe_2cut_allowlist(
                spqr_node,
                (poles_to_use[0], poles_to_use[1]),
                edges,
            )

            if not is_safe:
                all_branch_rooms: set[int] = set(poles_to_use)
                for c in sorted_children:
                    all_branch_rooms.update(_get_all_spqr_vertices(c))

                if len(poles_to_use) == 2:
                    metadata["cut_type"] = CutType.SPLIT_PAIR
                    metadata["poles"] = poles_to_use
                    metadata["cut_contract"] = SplitPairCut(poles=poles_to_use)

                return SolveTreeNode(
                    node_id=node_id,
                    node_type=TreeNodeType.RIGID_CLUSTER,
                    rooms=frozenset(all_branch_rooms),
                    children=(),
                    metadata=metadata,
                )

            converted_children = tuple(_convert(c, parent=spqr_node) for c in sorted_children)
            rooms = frozenset(cast(int, v) for v in poles_to_use)
            if len(poles_to_use) == 2:
                metadata["cut_type"] = CutType.SPLIT_PAIR
                metadata["poles"] = poles_to_use
                metadata["cut_contract"] = SplitPairCut(poles=poles_to_use)
            return SolveTreeNode(
                node_id=node_id,
                node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
                rooms=rooms,
                children=converted_children,
                metadata=metadata,
            )

        converted_children = tuple(_convert(c, parent=spqr_node) for c in sorted_children)
        if spqr_node.type == spqrtree.NodeType.R:
            node_type = TreeNodeType.RIGID_CLUSTER
            rooms = frozenset(cast(int, v) for v in spqr_node.skeleton.vertices)
        elif spqr_node.type == spqrtree.NodeType.S:
            node_type = TreeNodeType.SERIES_LOOP
            rooms = frozenset(cast(int, v) for v in spqr_node.skeleton.vertices)
        else:
            node_type = TreeNodeType.RIGID_CLUSTER
            source = spqr_node.skeleton.vertices if (spqr_node.skeleton and spqr_node.skeleton.vertices) else spqr_node.poles
            rooms = frozenset(cast(int, v) for v in source)

        if parent is not None and len(attachment_poles) == 2:
            metadata["cut_type"] = CutType.SPLIT_PAIR
            metadata["poles"] = attachment_poles
            metadata["cut_contract"] = SplitPairCut(poles=attachment_poles)

        return SolveTreeNode(
            node_id=node_id,
            node_type=node_type,
            rooms=rooms,
            children=converted_children,
            metadata=metadata,
        )

    try:
        spqr_tree_root = spqrtree.build_spqr_tree(multigraph)
        rerooted_root = _reroot_spqr_tree_to_primary_cluster(spqr_tree_root)
        return _convert(rerooted_root, parent=None)
    except (ValueError, IndexError, RuntimeError):
        return SolveTreeNode(
            node_id=f"{node_id_prefix}_0",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=block_nodes,
        )


def _attach_articulation_children(
    spqr_node: SolveTreeNode,
    articulation_children: Sequence[SolveTreeNode],
) -> SolveTreeNode:
    """Attach block-cut child blocks to the SPQR node containing their articulation anchor."""
    if not articulation_children:
        return spqr_node

    if not spqr_node.children:
        return SolveTreeNode(
            node_id=spqr_node.node_id,
            node_type=spqr_node.node_type,
            rooms=spqr_node.rooms,
            children=tuple(list(spqr_node.children) + list(articulation_children)),
            metadata=spqr_node.metadata,
        )

    direct_children: list[SolveTreeNode] = []
    child_to_art_children: dict[str, list[SolveTreeNode]] = collections.defaultdict(list)

    for art_child in articulation_children:
        anchor = art_child.metadata.get("anchor_vnum")
        target_child_id: str | None = None
        if anchor is not None and anchor not in spqr_node.rooms:
            for ch in spqr_node.children:
                if anchor in ch.rooms:
                    target_child_id = ch.node_id
                    break
        if target_child_id is not None:
            child_to_art_children[target_child_id].append(art_child)
        else:
            direct_children.append(art_child)

    updated_spqr_children: list[SolveTreeNode] = []
    for ch in spqr_node.children:
        if ch.node_id in child_to_art_children:
            updated_ch = _attach_articulation_children(ch, child_to_art_children[ch.node_id])
            updated_spqr_children.append(updated_ch)
        else:
            updated_spqr_children.append(ch)

    return SolveTreeNode(
        node_id=spqr_node.node_id,
        node_type=spqr_node.node_type,
        rooms=spqr_node.rooms,
        children=tuple(updated_spqr_children + direct_children),
        metadata=spqr_node.metadata,
    )


class SpqrTreeBuilder(TreeBuilder):
    """Stage 3 tree builder decomposing 0-cuts, 1-cuts (block-cut), and 2-cuts (SPQR)."""

    def __init__(self, block_cut_builder: BlockCutTreeBuilder | None = None) -> None:
        self._block_cut_builder = block_cut_builder or BlockCutTreeBuilder()

    def build_tree(
        self,
        simplified: SimplifiedGraph,
        options: LayoutOptions | None = None,
    ) -> SolveTree:
        """Decompose a simplified graph into a hierarchical SolveTree with SPQR decomposition."""
        block_cut_tree = self._block_cut_builder.build_tree(simplified, options)

        if block_cut_tree.root.node_type == TreeNodeType.EMPTY:
            return block_cut_tree

        edges = simplified.graph.edges

        def _transform_node(node: SolveTreeNode) -> SolveTreeNode:
            if node.node_type == TreeNodeType.WORLD_ROOT:
                transformed_children = tuple(_transform_node(child) for child in node.children)
                return SolveTreeNode(
                    node_id=node_id if (node_id := node.node_id) else "",
                    node_type=node.node_type,
                    rooms=node.rooms,
                    children=transformed_children,
                    metadata=node.metadata,
                )

            # First, recursively process articulation children
            child_nodes: list[SolveTreeNode] = [_transform_node(child) for child in node.children]

            if len(node.rooms) >= 4 and node.node_type in (TreeNodeType.BLOCK, TreeNodeType.BICONNECTED_BLOCK):
                spqr_root = build_block_spqr_tree(
                    block_nodes=frozenset(node.rooms),
                    edges=edges,
                    node_id_prefix=f"{node.node_id}_spqr",
                )
                merged_meta = dict(spqr_root.metadata)
                merged_meta.update(node.metadata)

                spqr_root = SolveTreeNode(
                    node_id=spqr_root.node_id,
                    node_type=spqr_root.node_type,
                    rooms=spqr_root.rooms,
                    children=spqr_root.children,
                    metadata=merged_meta,
                )
                return _attach_articulation_children(spqr_root, child_nodes)

            return SolveTreeNode(
                node_id=node.node_id,
                node_type=node.node_type,
                rooms=node.rooms,
                children=tuple(child_nodes),
                metadata=node.metadata,
            )

        new_root = _transform_node(block_cut_tree.root)
        if new_root.node_type == TreeNodeType.WORLD_ROOT:
            components = new_root.children
        else:
            components = (new_root,)

        return SolveTree(root=new_root, components=components, graph=simplified.graph)
