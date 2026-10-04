"""Comprehensive unit and integration tests for SPQR tree triconnectivity decomposition."""

from __future__ import annotations

import networkx as nx
import pytest

from romutil.layout import (
    BlockCutTreeBuilder,
    HierarchicalLayoutSolver,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SpqrTreeBuilder,
    TreeBuilder,
    build_block_spqr_tree,
)
from romutil.models import Direction, ExitDef, RoomDef
from romutil.parser import parse_file
from romutil.types import LayoutOptions, SolvedLayout, SolverStatus, WorldGraph


def _make_room(vnum: int, exits: list[ExitDef] | None = None) -> RoomDef:
    """Helper to construct a RoomDef with given exits."""
    return RoomDef(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(exits or []),
    )


def _make_simplified_graph(
    rooms: list[int],
    edges: list[tuple[int, int]] | None = None,
    cut_edges: list[tuple[int, int]] | None = None,
) -> SimplifiedGraph:
    """Helper to construct a SimplifiedGraph from node lists and edge pairs."""
    room_defs = {v: _make_room(v) for v in rooms}
    layout_edges: list[LayoutEdge] = []
    if edges:
        for u, v in edges:
            layout_edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north, cut=False))
            layout_edges.append(LayoutEdge(src=v, dst=u, direction=Direction.south, cut=False))
    if cut_edges:
        for u, v in cut_edges:
            layout_edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north, cut=True))
    layout_graph = LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))
    return SimplifiedGraph(graph=layout_graph)


class TestBuildBlockSpqrTree:
    """Unit tests for build_block_spqr_tree primitive."""

    def test_empty_nodes(self) -> None:
        """Empty block nodes produce empty rigid_cluster."""
        node = build_block_spqr_tree(frozenset(), [])
        assert node.node_type == "rigid_cluster"
        assert node.rooms == frozenset()
        assert node.children == ()

    def test_single_node(self) -> None:
        """Single room block produces rigid_cluster with 1 room."""
        node = build_block_spqr_tree(frozenset({100}), [])
        assert node.node_type == "rigid_cluster"
        assert node.rooms == frozenset({100})
        assert node.children == ()

    def test_two_nodes(self) -> None:
        """Two room block produces rigid_cluster with 2 rooms."""
        edges = [LayoutEdge(src=1, dst=2, direction=Direction.north)]
        node = build_block_spqr_tree(frozenset({1, 2}), edges)
        assert node.node_type == "rigid_cluster"
        assert node.rooms == frozenset({1, 2})
        assert node.children == ()

    def test_non_biconnected_fallback(self) -> None:
        """3-room path graph (not biconnected) triggers graceful fallback to rigid_cluster."""
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=2, dst=3, direction=Direction.north),
        ]
        node = build_block_spqr_tree(frozenset({1, 2, 3}), edges)
        assert node.node_type == "rigid_cluster"
        assert node.rooms == frozenset({1, 2, 3})
        assert node.children == ()

    def test_self_loops_and_cuts_ignored(self) -> None:
        """Self-loops and cut edges are ignored when constructing MultiGraph."""
        edges = [
            LayoutEdge(src=1, dst=1, direction=Direction.north),  # self loop
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=True),  # cut
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.north, cut=False),
            LayoutEdge(src=3, dst=1, direction=Direction.north, cut=False),
        ]
        node = build_block_spqr_tree(frozenset({1, 2, 3}), edges)
        assert node.node_type == "series_loop"
        assert node.rooms == frozenset({1, 2, 3})

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_series_polygon_cycles(self, n: int) -> None:
        """Cycle graph C_n decomposes into a single S-node (series_loop)."""
        nodes = list(range(1, n + 1))
        edges: list[LayoutEdge] = []
        for i in range(n):
            u = nodes[i]
            v = nodes[(i + 1) % n]
            edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north))
            edges.append(LayoutEdge(src=v, dst=u, direction=Direction.south))

        tree_node = build_block_spqr_tree(frozenset(nodes), edges)
        assert tree_node.node_type == "series_loop"
        assert tree_node.rooms == frozenset(nodes)
        assert tree_node.children == ()

    def test_k4_rigid_cluster(self) -> None:
        """Complete graph K_4 decomposes into a single R-node (rigid_cluster)."""
        nodes = [1, 2, 3, 4]
        edges: list[LayoutEdge] = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north))

        tree_node = build_block_spqr_tree(frozenset(nodes), edges)
        assert tree_node.node_type == "rigid_cluster"
        assert tree_node.rooms == frozenset(nodes)
        assert tree_node.children == ()

    def test_3d_cube_rigid_cluster(self) -> None:
        """3D cube graph (8 vertices, 12 edges) decomposes into a single R-node (rigid_cluster)."""
        cube = nx.cubical_graph()
        # Relabel nodes to 1..8
        mapping = {node: i + 1 for i, node in enumerate(cube.nodes())}
        cube = nx.relabel_nodes(cube, mapping)
        edges = [
            LayoutEdge(src=u, dst=v, direction=Direction.north)
            for u, v in cube.edges()
        ]

        tree_node = build_block_spqr_tree(frozenset(cube.nodes()), edges)
        assert tree_node.node_type == "rigid_cluster"
        assert tree_node.rooms == frozenset(range(1, 9))
        assert tree_node.children == ()

    def test_parallel_split_pair_dipole(self) -> None:
        """Graph with 2 poles and 3 independent paths decomposes into a P-node (split_pair_container)."""
        # Poles 1, 2 with paths: (1, 2), (1, 3, 2), (1, 4, 2)
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.north),
            LayoutEdge(src=3, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=4, direction=Direction.north),
            LayoutEdge(src=4, dst=2, direction=Direction.north),
        ]
        tree_node = build_block_spqr_tree(frozenset({1, 2, 3, 4}), edges)
        assert tree_node.node_type == "split_pair_container"
        assert tree_node.rooms == frozenset({1, 2})
        assert tree_node.metadata.get("cut_type") == "split_pair"
        assert set(tree_node.metadata.get("poles", ())) == {1, 2}
        assert len(tree_node.children) == 2
        for child in tree_node.children:
            assert child.node_type == "series_loop"

    def test_two_k4s_split_pair(self) -> None:
        """Two K4s sharing an edge (split pair) decompose into P-node container with R-node children."""
        edges: list[LayoutEdge] = []
        # K4 on 1, 2, 3, 4
        for u in [1, 2, 3, 4]:
            for v in [1, 2, 3, 4]:
                if u < v:
                    edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north))
        # K4 on 1, 2, 5, 6 (shares edge 1, 2)
        for u in [1, 2, 5, 6]:
            for v in [1, 2, 5, 6]:
                if u < v and not (u == 1 and v == 2):
                    edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north))

        tree_node = build_block_spqr_tree(frozenset({1, 2, 3, 4, 5, 6}), edges)
        assert tree_node.node_type == "split_pair_container"
        assert tree_node.rooms == frozenset({1, 2})
        assert set(tree_node.metadata.get("poles", ())) == {1, 2}
        assert len(tree_node.children) == 2
        for child in tree_node.children:
            assert child.node_type == "rigid_cluster"


class TestSpqrTreeBuilder:
    """Unit tests for SpqrTreeBuilder class."""

    def test_implements_tree_builder_protocol(self) -> None:
        """Verify SpqrTreeBuilder implements runtime TreeBuilder protocol."""
        builder = SpqrTreeBuilder()
        assert isinstance(builder, TreeBuilder)

    def test_empty_graph(self) -> None:
        """Empty graph returns SolveTree with empty root."""
        builder = SpqrTreeBuilder()
        simplified = _make_simplified_graph([])
        tree = builder.build_tree(simplified)
        assert tree.root.node_type == "empty"
        assert tree.components == ()

    def test_small_block_preserved_without_spqr(self) -> None:
        """Blocks with < 4 nodes are preserved as flat blocks."""
        builder = SpqrTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3],
            edges=[(1, 2), (2, 3), (3, 1)],
        )
        tree = builder.build_tree(simplified)
        assert tree.root.node_type == "biconnected_block"
        assert tree.root.rooms == frozenset({1, 2, 3})
        assert len(tree.components) == 1

    def test_c4_cycle_decomposed_to_s_node(self) -> None:
        """C_4 biconnected block decomposes into series_loop."""
        builder = SpqrTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4],
            edges=[(1, 2), (2, 3), (3, 4), (4, 1)],
        )
        tree = builder.build_tree(simplified)
        assert len(tree.components) == 1
        assert tree.root.node_type == "series_loop"
        assert tree.root.rooms == frozenset({1, 2, 3, 4})

    def test_k4_complete_decomposed_to_r_node(self) -> None:
        """K_4 biconnected block decomposes into rigid_cluster."""
        builder = SpqrTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4],
            edges=[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        )
        tree = builder.build_tree(simplified)
        assert len(tree.components) == 1
        assert tree.root.node_type == "rigid_cluster"
        assert tree.root.rooms == frozenset({1, 2, 3, 4})

    def test_disconnected_components_0_cuts(self) -> None:
        """Multiple disconnected components produce world_root with SPQR decomposed components."""
        builder = SpqrTreeBuilder()
        # Comp 0: K4 on 1..4; Comp 1: C4 on 10..13
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 10, 11, 12, 13],
            edges=[
                (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                (10, 11), (11, 12), (12, 13), (13, 10),
            ],
        )
        tree = builder.build_tree(simplified)
        assert tree.root.node_type == "world_root"
        assert len(tree.components) == 2

        comp_types = {comp.node_type for comp in tree.components}
        assert comp_types == {"rigid_cluster", "series_loop"}

    def test_mixed_barbell_articulation_r_and_s(self) -> None:
        """Barbell graph connecting K_4 and C_4 via articulation point room 4."""
        builder = SpqrTreeBuilder()
        # K4 on 1, 2, 3, 4 (6 edges)
        # C4 on 4, 5, 6, 7 (4 edges)
        # Articulation point is 4
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6, 7],
            edges=[
                (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                (4, 5), (5, 6), (6, 7), (7, 4),
            ],
        )
        tree = builder.build_tree(simplified)
        assert len(tree.components) == 1
        root = tree.root
        assert root.node_type == "rigid_cluster"
        assert root.rooms == frozenset({1, 2, 3, 4})
        assert len(root.children) == 1

        child = root.children[0]
        assert child.node_type == "series_loop"
        assert child.rooms == frozenset({4, 5, 6, 7})
        assert child.metadata.get("cut_type") == "articulation_point"
        assert child.metadata.get("anchor_vnum") == 4

    def test_articulation_child_attached_to_spqr_container_internal_branch(self) -> None:
        """Articulation child attaching to an internal room of a split pair branch is routed to that branch."""
        builder = SpqrTreeBuilder()
        # Block 1 (Dipole between 1 and 2):
        # Path 1: 1 - 2
        # Path 2: 1 - 3 - 2
        # Path 3: 1 - 4 - 2
        # Block 2 (C4 attached at room 3):
        # 3 - 5 - 6 - 7 - 3
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6, 7],
            edges=[
                (1, 2), (1, 3), (3, 2), (1, 4), (4, 2),
                (3, 5), (5, 6), (6, 7), (7, 3),
            ],
        )
        tree = builder.build_tree(simplified)
        assert len(tree.components) == 1
        root = tree.root
        assert root.node_type == "split_pair_container"
        assert root.rooms == frozenset({1, 2})

        # Find child S-node that contains room 3
        s_node_3 = next(c for c in root.children if 3 in c.rooms)
        assert s_node_3.node_type == "series_loop"
        # The C4 child block on {3, 5, 6, 7} should be attached under s_node_3!
        assert len(s_node_3.children) == 1
        art_child = s_node_3.children[0]
        assert art_child.node_type == "series_loop"
        assert art_child.rooms == frozenset({3, 5, 6, 7})
        assert art_child.metadata.get("anchor_vnum") == 3

    def test_articulation_child_attached_to_pole(self) -> None:
        """Articulation child attaching to a pole attaches directly to the split pair container."""
        builder = SpqrTreeBuilder()
        # Block 1 (Dipole between 1 and 2):
        # Path 1: 1 - 2, Path 2: 1 - 3 - 2, Path 3: 1 - 4 - 2
        # Block 2 (C4 attached at room 1, a pole):
        # 1 - 5 - 6 - 7 - 1
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6, 7],
            edges=[
                (1, 2), (1, 3), (3, 2), (1, 4), (4, 2),
                (1, 5), (5, 6), (6, 7), (7, 1),
            ],
        )
        tree = builder.build_tree(simplified)
        root = tree.root
        assert root.node_type == "split_pair_container"
        assert root.rooms == frozenset({1, 2})

        # The C4 child should be attached directly to the root container because anchor 1 is a pole
        c4_children = [c for c in root.children if c.rooms == frozenset({1, 5, 6, 7})]
        assert len(c4_children) == 1
        assert c4_children[0].metadata.get("anchor_vnum") == 1

    def test_custom_block_cut_builder_injection(self) -> None:
        """Verify constructor accepts custom block_cut_builder."""
        custom_bct = BlockCutTreeBuilder()
        builder = SpqrTreeBuilder(block_cut_builder=custom_bct)
        simplified = _make_simplified_graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (4, 1)])
        tree = builder.build_tree(simplified)
        assert tree.root.node_type == "series_loop"


class TestSpqrEndToEndIntegration:
    """End-to-end integration tests using HierarchicalLayoutSolver."""

    def test_solver_with_spqr_synthetic_world_graph(self) -> None:
        """HierarchicalLayoutSolver solves synthetic topology with SpqrTreeBuilder."""
        rooms = {
            1: RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=Direction.north, dst_vnum=2), ExitDef(direction=Direction.east, dst_vnum=3))),
            2: RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=Direction.south, dst_vnum=1), ExitDef(direction=Direction.east, dst_vnum=4))),
            3: RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=Direction.west, dst_vnum=1), ExitDef(direction=Direction.north, dst_vnum=4))),
            4: RoomDef(vnum=4, name="R4", description="", exits=(ExitDef(direction=Direction.west, dst_vnum=2), ExitDef(direction=Direction.south, dst_vnum=3))),
        }
        all_exits = tuple(e for r in rooms.values() for e in r.exits)
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)
        solver = HierarchicalLayoutSolver(tree_builder=SpqrTreeBuilder())
        solved = solver.solve(world_graph, LayoutOptions())

        assert isinstance(solved, SolvedLayout)
        assert solved.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert set(solved.coordinates.keys()) == {1, 2, 3, 4}

    def test_solver_with_spqr_school_fixture(self) -> None:
        """HierarchicalLayoutSolver solves real area school.are with SpqrTreeBuilder."""
        area = parse_file("tests/fixtures/areas/school.are")
        world_graph = WorldGraph.from_area_data(area)

        solver = HierarchicalLayoutSolver(tree_builder=SpqrTreeBuilder())
        solved = solver.solve(world_graph, LayoutOptions())

        assert isinstance(solved, SolvedLayout)
        assert solved.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(solved.coordinates) == len(world_graph.rooms)
