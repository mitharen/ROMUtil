"""Stage 3 tree builder decomposing graphs into connected components and block-cut trees."""

from __future__ import annotations

import collections
from typing import Any

import networkx as nx

from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
)
from romutil.layout.protocols import TreeBuilder
from romutil.types import LayoutOptions


class BlockCutTreeBuilder(TreeBuilder):
    """Stage 3 tree builder decomposing layout graphs via 0-cut components and 1-cut block-cut trees."""

    def build_tree(
        self,
        simplified: SimplifiedGraph,
        options: LayoutOptions | None = None,
    ) -> SolveTree:
        """Decompose a simplified graph into a hierarchical SolveTree."""
        if not simplified.graph.rooms:
            return SolveTree(
                root=SolveTreeNode(
                    node_id="empty",
                    node_type=TreeNodeType.EMPTY,
                    rooms=frozenset(),
                ),
                graph=simplified.graph,
            )

        graph = nx.Graph()
        for vnum in simplified.graph.rooms:
            graph.add_node(vnum)

        for edge in simplified.graph.edges:
            if not edge.cut and edge.src in simplified.graph.rooms and edge.dst in simplified.graph.rooms:
                graph.add_edge(edge.src, edge.dst)

        raw_components = list(nx.connected_components(graph))
        sorted_components = sorted(raw_components, key=lambda c: (-len(c), min(c)))

        comp_nodes: list[SolveTreeNode] = []
        for idx, comp_nodes_set in enumerate(sorted_components):
            comp_subgraph = graph.subgraph(comp_nodes_set).copy()
            comp_node = self._build_component_tree(comp_subgraph, idx)
            comp_nodes.append(comp_node)

        if len(comp_nodes) == 1:
            return SolveTree(root=comp_nodes[0], components=tuple(comp_nodes), graph=simplified.graph)

        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=tuple(comp_nodes),
        )
        return SolveTree(root=world_root, components=tuple(comp_nodes), graph=simplified.graph)

    def _build_component_tree(
        self,
        comp_graph: nx.Graph,
        comp_idx: int,
    ) -> SolveTreeNode:
        """Build a solve tree node for a single connected component."""
        raw_blocks = list(nx.biconnected_components(comp_graph))
        if len(raw_blocks) <= 1:
            return SolveTreeNode(
                node_id=f"comp_{comp_idx}_block_0",
                node_type=TreeNodeType.BICONNECTED_BLOCK,
                rooms=frozenset(comp_graph.nodes),
            )

        blocks = [frozenset(b) for b in raw_blocks]
        cut_vertices = set(nx.articulation_points(comp_graph))

        def block_key(b: frozenset[int]) -> tuple[int, int, tuple[int, ...]]:
            sorted_rooms = tuple(sorted(b))
            return (-len(b), sorted_rooms[0] if sorted_rooms else 0, sorted_rooms)

        root_block_idx = min(range(len(blocks)), key=lambda i: block_key(blocks[i]))

        ap_to_blocks: dict[int, list[int]] = collections.defaultdict(list)
        block_to_aps: dict[int, list[int]] = collections.defaultdict(list)
        for i, block in enumerate(blocks):
            for a in cut_vertices:
                if a in block:
                    ap_to_blocks[a].append(i)
                    block_to_aps[i].append(a)

        visited_blocks = {root_block_idx}
        visited_aps: set[int] = set()
        block_children: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)

        queue = [root_block_idx]
        while queue:
            u = queue.pop(0)
            for a in sorted(block_to_aps[u]):
                if a in visited_aps:
                    continue
                visited_aps.add(a)
                for v in ap_to_blocks[a]:
                    if v not in visited_blocks:
                        visited_blocks.add(v)
                        block_children[u].append((v, a))
                        queue.append(v)

        for u in block_children:
            block_children[u].sort(key=lambda item: (item[1], block_key(blocks[item[0]])))

        counter = 0

        def build_node(b_idx: int, anchor: int | None) -> SolveTreeNode:
            nonlocal counter
            curr_id = f"comp_{comp_idx}_block_{counter}"
            counter += 1

            child_nodes = [
                build_node(child_idx, child_anchor)
                for child_idx, child_anchor in block_children[b_idx]
            ]
            meta: dict[str, Any] = {}
            if anchor is not None:
                meta = {
                    "cut_type": CutType.ARTICULATION_POINT,
                    "anchor_vnum": anchor,
                    "cut_contract": ArticulationPointCut(anchor_vnum=anchor),
                }

            return SolveTreeNode(
                node_id=curr_id,
                node_type=TreeNodeType.BLOCK,
                rooms=blocks[b_idx],
                children=tuple(child_nodes),
                metadata=meta,
            )

        return build_node(root_block_idx, None)
