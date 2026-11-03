"""Stage 3 tree builder decomposing graphs into connected components and block-cut trees."""

from __future__ import annotations

import collections
from typing import Any

import networkx as nx

from romutil.layout.hierarchy.transforms import TreeTransformationPass
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutGraph,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
)
from romutil.layout.protocols import TreeBuilder
from romutil.models import Direction
from romutil.types import LayoutOptions


class BlockCutDecompositionPass(TreeTransformationPass):
    """Decompose disconnected components (0-cuts) and biconnected blocks with 1-cut articulation trees (1-cuts)."""

    def __init__(self, graph: LayoutGraph | None = None) -> None:
        self._graph = graph

    @property
    def graph(self) -> LayoutGraph | None:
        """Optional LayoutGraph override for decomposition."""
        return self._graph

    def apply(self, tree: SolveTree) -> SolveTree:
        """Decompose a SolveTree or seed into 0-cut components and 1-cut block-cut trees."""
        target_graph = self._graph or tree.graph
        if target_graph is None or not target_graph.rooms or tree.root.node_type == TreeNodeType.EMPTY:
            return SolveTree(
                root=SolveTreeNode(
                    node_id="empty",
                    node_type=TreeNodeType.EMPTY,
                    rooms=frozenset(),
                ),
                components=(),
                graph=target_graph,
            )

        graph = nx.Graph()
        for vnum in target_graph.rooms:
            graph.add_node(vnum)

        for edge in target_graph.edges:
            if not edge.cut and edge.src in target_graph.rooms and edge.dst in target_graph.rooms:
                graph.add_edge(edge.src, edge.dst)

        raw_components = list(nx.connected_components(graph))
        sorted_components = sorted(raw_components, key=lambda c: (-len(c), min(c)))

        comp_nodes: list[SolveTreeNode] = []
        for idx, comp_nodes_set in enumerate(sorted_components):
            comp_subgraph = graph.subgraph(comp_nodes_set).copy()
            comp_node = self._build_component_tree(comp_subgraph, idx, target_graph)
            comp_nodes.append(comp_node)

        if len(comp_nodes) == 1:
            return SolveTree(root=comp_nodes[0], components=tuple(comp_nodes), graph=target_graph)

        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=tuple(comp_nodes),
        )
        return SolveTree(root=world_root, components=tuple(comp_nodes), graph=target_graph)

    def _build_component_tree(
        self,
        comp_graph: nx.Graph,
        comp_idx: int,
        target_graph: LayoutGraph | None = None,
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

        effective_graph = target_graph or self._graph
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
                block_child_rooms = frozenset(blocks[b_idx]) - {anchor}
                candidate_edges = []
                if effective_graph is not None:
                    for e in effective_graph.edges:
                        if e.cut:
                            continue
                        if (e.src == anchor and e.dst in block_child_rooms) or (
                            e.dst == anchor and e.src in block_child_rooms
                        ):
                            candidate_edges.append(e)

                connecting_rooms = {
                    e.dst if e.src == anchor else e.src for e in candidate_edges
                }

                contract: ArticulationPointCut | BridgeCut
                cut_type: CutType
                bridge_edge: tuple[int, int] | None = None
                edge_dir: int | None = None
                edge_weight: int | None = None

                if len(connecting_rooms) == 1:
                    v = next(iter(connecting_rooms))
                    fwd_edges = [e for e in candidate_edges if e.src == anchor and e.dst == v]
                    rev_edges = [e for e in candidate_edges if e.src == v and e.dst == anchor]

                    effective_dirs: set[int] = set()
                    weights: list[int] = []
                    for e in fwd_edges:
                        effective_dirs.add(int(e.direction))
                        weights.append(e.weight)
                    for e in rev_edges:
                        try:
                            effective_dirs.add(int(Direction(e.direction).invert()))
                            weights.append(e.weight)
                        except (ValueError, KeyError):
                            pass

                    if len(effective_dirs) == 1:
                        edge_dir = next(iter(effective_dirs))
                        edge_weight = max(weights) if weights else 1
                        bridge_edge = (anchor, v)
                        contract = BridgeCut(
                            bridge_edge=bridge_edge,
                            direction=edge_dir,
                            weight=edge_weight,
                            anchor_vnum=anchor,
                        )
                        cut_type = CutType.BRIDGE
                    else:
                        contract = ArticulationPointCut(anchor_vnum=anchor)
                        cut_type = CutType.ARTICULATION_POINT
                else:
                    contract = ArticulationPointCut(anchor_vnum=anchor)
                    cut_type = CutType.ARTICULATION_POINT

                meta = {
                    "cut_type": cut_type,
                    "anchor_vnum": anchor,
                    "cut_contract": contract,
                }
                if bridge_edge is not None and edge_dir is not None and edge_weight is not None:
                    meta["bridge_edge"] = bridge_edge
                    meta["direction"] = edge_dir
                    meta["weight"] = edge_weight

            node_rooms = frozenset(blocks[b_idx]) - {anchor} if anchor is not None else blocks[b_idx]

            return SolveTreeNode(
                node_id=curr_id,
                node_type=TreeNodeType.BLOCK,
                rooms=node_rooms,
                children=tuple(child_nodes),
                metadata=meta,
            )

        return build_node(root_block_idx, None)


class BlockCutTreeBuilder(TreeBuilder):
    """Stage 3 tree builder decomposing layout graphs via 0-cut components and 1-cut block-cut trees."""

    def __init__(self, pass_: BlockCutDecompositionPass | None = None) -> None:
        self._pass = pass_ or BlockCutDecompositionPass()

    def build_tree(
        self,
        simplified: SimplifiedGraph,
        options: LayoutOptions | None = None,
    ) -> SolveTree:
        """Decompose a simplified graph into a hierarchical SolveTree."""
        seed = SolveTree.seed(simplified.graph)
        return self._pass.apply(seed)
