"""Stage 3 tree builder decomposing biconnected blocks via SPQR triconnectivity."""

from __future__ import annotations

import collections
from typing import Any, Sequence, cast

import spqrtree

from romutil.layout.hierarchy.block_cut import BlockCutTreeBuilder
from romutil.layout.ir import LayoutEdge, SimplifiedGraph, SolveTree, SolveTreeNode
from romutil.layout.protocols import TreeBuilder
from romutil.types import LayoutOptions


def build_block_spqr_tree(
    block_nodes: frozenset[int],
    edges: Sequence[LayoutEdge],
    node_id_prefix: str = "spqr",
) -> SolveTreeNode:
    """Decompose a single biconnected block into its canonical SPQR tree."""
    if not block_nodes:
        return SolveTreeNode(
            node_id=f"{node_id_prefix}_0",
            node_type="rigid_cluster",
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

    def _convert(spqr_node: spqrtree.SPQRNode) -> SolveTreeNode:
        nonlocal counter
        node_id = f"{node_id_prefix}_{counter}"
        counter += 1

        def _get_vertices(node: spqrtree.SPQRNode) -> tuple[int, ...]:
            source = node.skeleton.vertices if node.skeleton.vertices else node.poles
            return tuple(sorted(cast(int, v) for v in source))

        sorted_children = sorted(
            spqr_node.children,
            key=lambda c: (
                c.type.value,
                _get_vertices(c),
            ),
        )
        converted_children = tuple(_convert(c) for c in sorted_children)

        rooms: frozenset[int]
        metadata: dict[str, Any]
        if spqr_node.type == spqrtree.NodeType.R:
            node_type = "rigid_cluster"
            rooms = frozenset(cast(int, v) for v in spqr_node.skeleton.vertices)
            metadata = {}
        elif spqr_node.type == spqrtree.NodeType.S:
            node_type = "series_loop"
            rooms = frozenset(cast(int, v) for v in spqr_node.skeleton.vertices)
            metadata = {}
        elif spqr_node.type == spqrtree.NodeType.P:
            node_type = "split_pair_container"
            rooms = frozenset(cast(int, v) for v in spqr_node.poles)
            metadata = {
                "cut_type": "split_pair",
                "poles": tuple(cast(int, v) for v in spqr_node.poles),
            }
        else:
            node_type = "rigid_cluster"
            source = spqr_node.skeleton.vertices if spqr_node.skeleton.vertices else spqr_node.poles
            rooms = frozenset(cast(int, v) for v in source)
            metadata = {}

        return SolveTreeNode(
            node_id=node_id,
            node_type=node_type,
            rooms=rooms,
            children=converted_children,
            metadata=metadata,
        )

    try:
        spqr_tree_root = spqrtree.build_spqr_tree(multigraph)
        return _convert(spqr_tree_root)
    except (ValueError, IndexError, RuntimeError):
        return SolveTreeNode(
            node_id=f"{node_id_prefix}_0",
            node_type="rigid_cluster",
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

        if block_cut_tree.root.node_type == "empty":
            return block_cut_tree

        edges = simplified.graph.edges

        def _transform_node(node: SolveTreeNode) -> SolveTreeNode:
            if node.node_type == "world_root":
                transformed_children = tuple(_transform_node(child) for child in node.children)
                return SolveTreeNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    rooms=node.rooms,
                    children=transformed_children,
                    metadata=node.metadata,
                )

            # First, recursively process articulation children
            child_nodes: list[SolveTreeNode] = [_transform_node(child) for child in node.children]

            if len(node.rooms) >= 4 and node.node_type in ("block", "biconnected_block"):
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
        if new_root.node_type == "world_root":
            components = new_root.children
        else:
            components = (new_root,)

        return SolveTree(root=new_root, components=components)
