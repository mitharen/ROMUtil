"""Stage 3 tree builder decomposing biconnected blocks via SPQR triconnectivity."""

from __future__ import annotations

from typing import Any, Sequence, cast

import spqrtree

from romutil.layout.hierarchy.transforms import (
    TreeTransformationPass,
    _get_anchor_vnum,
    _is_one_cut_child,
    attach_articulation_children,
    strip_anchor,
)
from romutil.layout.ir import (
    ArticulationPointCut,
    CutType,
    LayoutEdge,
    LayoutGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)


def reroot_spqr_tree(
    root: spqrtree.SPQRNode,
    target: spqrtree.SPQRNode,
) -> spqrtree.SPQRNode:
    """Reroot an SPQR tree such that target becomes the root node.

    Inverts child/parent relationships along the path from root to target in
    the undirected SPQR tree and updates attachment poles accordingly.
    """
    if target is root:
        return root

    path: list[spqrtree.SPQRNode] = []
    curr: spqrtree.SPQRNode | None = target
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    path.reverse()
    if not path or path[0] is not root:
        raise ValueError("Target node is not reachable from the provided root")

    for i in range(len(path) - 1, 0, -1):
        child = path[i]
        parent = path[i - 1]
        if child in parent.children:
            parent.children.remove(child)
        child.children.append(parent)
        parent.parent = child

        shared = set(parent.skeleton.vertices if parent.skeleton else parent.poles) & set(
            child.skeleton.vertices if child.skeleton else child.poles
        )
        if len(shared) == 2:
            parent.poles = tuple(sorted(shared))
        elif child.poles:
            parent.poles = child.poles

    target.parent = None
    return target


def _find_anchor_spqr_node(
    root: spqrtree.SPQRNode,
    anchor: int,
) -> spqrtree.SPQRNode | None:
    """Find the optimal SPQR component containing the articulation anchor vertex."""
    nodes: list[spqrtree.SPQRNode] = []

    def _collect(node: spqrtree.SPQRNode) -> None:
        nodes.append(node)
        for c in node.children:
            _collect(c)

    _collect(root)

    candidates: list[tuple[int, int, int, spqrtree.SPQRNode]] = []
    for idx, node in enumerate(nodes):
        verts = set(node.skeleton.vertices if node.skeleton else node.poles)
        if anchor in verts:
            real_incident = sum(
                1
                for e in (node.skeleton.edges if node.skeleton else ())
                if not getattr(e, "virtual", False) and (e.u == anchor or e.v == anchor)
            )
            is_not_p = 1 if node.type != spqrtree.NodeType.P else 0
            candidates.append((real_incident, is_not_p, -idx, node))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def build_block_spqr_tree(
    block_nodes: frozenset[int],
    edges: Sequence[LayoutEdge],
    node_id_prefix: str = "spqr",
    anchor: int | None = None,
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
            converted_children = tuple(_convert(c, parent=spqr_node) for c in sorted_children)
            if parent is not None and len(attachment_poles) == 2:
                rooms = frozenset(cast(int, v) for v in poles_to_use) - set(attachment_poles)
                metadata["cut_type"] = CutType.SPLIT_PAIR
                metadata["poles"] = attachment_poles
                metadata["cut_contract"] = SplitPairCut(poles=attachment_poles)
            else:
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
            source = spqr_node.skeleton.vertices
        elif spqr_node.type == spqrtree.NodeType.S:
            node_type = TreeNodeType.SERIES_LOOP
            source = spqr_node.skeleton.vertices
        else:
            node_type = TreeNodeType.RIGID_CLUSTER
            source = (
                spqr_node.skeleton.vertices
                if (spqr_node.skeleton and spqr_node.skeleton.vertices)
                else spqr_node.poles
            )

        if parent is not None and len(attachment_poles) == 2:
            rooms = frozenset(cast(int, v) for v in source) - set(attachment_poles)
            metadata["cut_type"] = CutType.SPLIT_PAIR
            metadata["poles"] = attachment_poles
            metadata["cut_contract"] = SplitPairCut(poles=attachment_poles)
        else:
            rooms = frozenset(cast(int, v) for v in source)

        return SolveTreeNode(
            node_id=node_id,
            node_type=node_type,
            rooms=rooms,
            children=converted_children,
            metadata=metadata,
        )

    try:
        spqr_tree_root = spqrtree.build_spqr_tree(multigraph)
        if anchor is not None:
            anchor_target = _find_anchor_spqr_node(spqr_tree_root, anchor)
            if anchor_target is not None:
                spqr_tree_root = reroot_spqr_tree(spqr_tree_root, anchor_target)
        return _convert(spqr_tree_root, parent=None)
    except (ValueError, IndexError, RuntimeError):
        return SolveTreeNode(
            node_id=f"{node_id_prefix}_0",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=block_nodes,
        )


class SpqrDecompositionPass(TreeTransformationPass):
    """Decompose biconnected blocks (>= 4 rooms) into SPQR trees (2-cuts)."""

    def __init__(self, graph: LayoutGraph | None = None) -> None:
        self._graph = graph

    @property
    def graph(self) -> LayoutGraph | None:
        """Optional LayoutGraph override for decomposition."""
        return self._graph

    def apply(self, tree: SolveTree) -> SolveTree:
        """Apply SPQR triconnected decomposition to biconnected blocks in a SolveTree."""
        if tree.root.node_type == TreeNodeType.EMPTY:
            return tree

        graph = self._graph or tree.graph
        edges = graph.edges if graph is not None else ()

        def _decompose_blocks(node: SolveTreeNode) -> SolveTreeNode:
            if node.node_type == TreeNodeType.WORLD_ROOT:
                transformed_children = tuple(_decompose_blocks(child) for child in node.children)
                return SolveTreeNode(
                    node_id=node.node_id if node.node_id else "",
                    node_type=node.node_type,
                    rooms=node.rooms,
                    children=transformed_children,
                    metadata=node.metadata,
                )

            child_nodes = [_decompose_blocks(child) for child in node.children]

            anchor = _get_anchor_vnum(node)

            block_verts = set(node.rooms)
            if anchor is not None:
                block_verts.add(anchor)

            if len(block_verts) >= 4 and node.node_type in (
                TreeNodeType.BLOCK,
                TreeNodeType.BICONNECTED_BLOCK,
            ):
                spqr_root = build_block_spqr_tree(
                    block_nodes=frozenset(block_verts),
                    edges=edges,
                    node_id_prefix=f"{node.node_id}_spqr",
                    anchor=anchor,
                )
                if anchor is not None:
                    spqr_root = strip_anchor(spqr_root, anchor)

                merged_meta = dict(spqr_root.metadata)
                merged_meta.update(node.metadata)

                spqr_root = SolveTreeNode(
                    node_id=spqr_root.node_id,
                    node_type=spqr_root.node_type,
                    rooms=spqr_root.rooms,
                    children=spqr_root.children,
                    metadata=merged_meta,
                )

                art_children = [c for c in child_nodes if _is_one_cut_child(c)]
                other_children = [c for c in child_nodes if not _is_one_cut_child(c)]

                spqr_root = attach_articulation_children(spqr_root, art_children)
                if other_children:
                    spqr_root = SolveTreeNode(
                        node_id=spqr_root.node_id,
                        node_type=spqr_root.node_type,
                        rooms=spqr_root.rooms,
                        children=tuple(spqr_root.children) + tuple(other_children),
                        metadata=spqr_root.metadata,
                    )

                return spqr_root

            current_rooms = node.rooms
            if anchor is not None and anchor in current_rooms:
                current_rooms = frozenset(current_rooms) - {anchor}

            return SolveTreeNode(
                node_id=node.node_id,
                node_type=node.node_type,
                rooms=current_rooms,
                children=tuple(child_nodes),
                metadata=node.metadata,
            )

        raw_root = _decompose_blocks(tree.root)
        if raw_root.node_type == TreeNodeType.WORLD_ROOT:
            components = raw_root.children
        else:
            components = (raw_root,)

        return SolveTree(root=raw_root, components=components, graph=graph)


def __getattr__(name: str) -> Any:
    if name == "SpqrTreeBuilder":
        from romutil.layout.hierarchy.pipeline import PipelineTreeBuilder

        return PipelineTreeBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
