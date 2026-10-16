"""Stage 3 tree builder decomposing biconnected blocks via SPQR triconnectivity."""

from __future__ import annotations

from typing import Any, Sequence, cast

import spqrtree

from romutil.layout.hierarchy.transforms import TreeTransformationPass
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
        rerooted_root = _reroot_spqr_tree_to_primary_cluster(spqr_tree_root)
        return _convert(rerooted_root, parent=None)
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

            anchor = node.metadata.get("anchor_vnum")
            if anchor is None and isinstance(node.cut_contract, ArticulationPointCut):
                anchor = node.cut_contract.anchor_vnum

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
                )
                merged_meta = dict(spqr_root.metadata)
                merged_meta.update(node.metadata)

                return SolveTreeNode(
                    node_id=spqr_root.node_id,
                    node_type=spqr_root.node_type,
                    rooms=spqr_root.rooms,
                    children=tuple(spqr_root.children) + tuple(child_nodes),
                    metadata=merged_meta,
                )

            return SolveTreeNode(
                node_id=node.node_id,
                node_type=node.node_type,
                rooms=node.rooms,
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
