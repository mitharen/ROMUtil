"""Comprehensive unit and integration tests for SPQR tree triconnectivity decomposition."""

from __future__ import annotations

import networkx as nx
import pytest

from romutil.layout import (
    ArticulationPointCut,
    BlockCutTreeBuilder,
    BridgeCut,
    CutType,
    HierarchicalLayoutSolver,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    SpqrTreeBuilder,
    TreeBuilder,
    TreeNodeType,
    build_block_spqr_tree,
    reroot_spqr_tree,
)
from romutil.layout.hierarchy.transforms import (
    TransverseSNodeSplitPass,
    _get_anchor_vnum,
)
from romutil.models import Direction, Exit, Room
from romutil.parser import parse_file
from romutil.types import LayoutOptions, SolvedLayout, SolverStatus, WorldGraph


def _make_room(vnum: int, exits: list[Exit] | None = None) -> Room:
    """Helper to construct a Room with given exits."""
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(exits or []),
    )


def _make_simplified_graph(
    rooms: list[int],
    edges: list[tuple[int, int]] | None = None,
    cut_edges: list[tuple[int, int]] | None = None,
    directed_edges: list[tuple[int, int, Direction]] | None = None,
) -> SimplifiedGraph:
    """Helper to construct a SimplifiedGraph from node lists and edge pairs."""
    room_defs = {v: _make_room(v) for v in rooms}
    layout_edges: list[LayoutEdge] = []
    if edges:
        for u, v in edges:
            layout_edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north, cut=False))
            layout_edges.append(LayoutEdge(src=v, dst=u, direction=Direction.south, cut=False))
    if directed_edges:
        for u, v, d in directed_edges:
            layout_edges.append(LayoutEdge(src=u, dst=v, direction=d, cut=False))
            layout_edges.append(LayoutEdge(src=v, dst=u, direction=d.invert(), cut=False))
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
        # Poles 1, 2 with paths: (1, 2) direct North, (1, 3, 2) East branch, (1, 4, 2) West branch
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=2, direction=Direction.west),
            LayoutEdge(src=1, dst=4, direction=Direction.west),
            LayoutEdge(src=4, dst=2, direction=Direction.east),
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
        """Two K4s sharing an edge produce a split-pair container with two rigid children."""
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

        # SPQR decomposition produces a split-pair container with 2 rigid children
        raw_node = build_block_spqr_tree(frozenset({1, 2, 3, 4, 5, 6}), edges)
        assert raw_node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert raw_node.rooms == frozenset({1, 2})
        assert len(raw_node.children) == 2
        assert all(child.node_type == TreeNodeType.RIGID_CLUSTER for child in raw_node.children)
        assert set(raw_node.metadata.get("poles", ())) == {1, 2}
        assert isinstance(raw_node.cut_contract, SplitPairCut)
        assert set(raw_node.cut_contract.poles) == {1, 2}

    def test_subdivided_k4_series_child_preserves_poles(self) -> None:
        """K_4 with subdivided edge produces R-node root with S-node child attaching via typed split pair."""
        # K_4 has (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4).
        # Subdivide (1, 2) through node 5: replace (1, 2) with (1, 5) and (5, 2).
        edges = [
            LayoutEdge(src=1, dst=5, direction=Direction.north),
            LayoutEdge(src=5, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.north),
            LayoutEdge(src=1, dst=4, direction=Direction.north),
            LayoutEdge(src=2, dst=3, direction=Direction.north),
            LayoutEdge(src=2, dst=4, direction=Direction.north),
            LayoutEdge(src=3, dst=4, direction=Direction.north),
        ]
        root = build_block_spqr_tree(frozenset({1, 2, 3, 4, 5}), edges)
        assert root.node_type == TreeNodeType.RIGID_CLUSTER
        assert {1, 2, 3, 4} <= set(root.rooms)
        assert len(root.children) == 1
        s_child = root.children[0]
        assert s_child.node_type == TreeNodeType.SERIES_LOOP
        assert s_child.cut_type == CutType.SPLIT_PAIR
        assert isinstance(s_child.cut_contract, SplitPairCut)
        assert set(s_child.cut_contract.poles) == {1, 2}
        assert s_child.cut_contract.poles[0] in root.rooms
        assert s_child.cut_contract.poles[1] in root.rooms


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
        assert child.rooms == frozenset({5, 6, 7})
        assert child.metadata.get("cut_type") == "articulation_point"
        assert child.metadata.get("anchor_vnum") == 4

    def test_articulation_child_attached_to_spqr_container_internal_branch(self) -> None:
        """Articulation child attaching to an internal room of a split pair branch is routed to that branch."""
        builder = SpqrTreeBuilder()
        # Block 1 (Dipole between 1 and 2):
        # Path 1: 1 - 2
        # Path 2: 1 - 3 - 2 (East)
        # Path 3: 1 - 4 - 2 (West)
        # Block 2 (C4 attached at room 3):
        # 3 - 5 - 6 - 7 - 3
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6, 7],
            directed_edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.west),
                (4, 2, Direction.east),
                (3, 5, Direction.north),
                (5, 6, Direction.east),
                (6, 7, Direction.south),
                (7, 3, Direction.west),
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
        assert art_child.rooms == frozenset({5, 6, 7})
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
            directed_edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.west),
                (4, 2, Direction.east),
                (1, 5, Direction.south),
                (5, 6, Direction.east),
                (6, 7, Direction.north),
                (7, 1, Direction.west),
            ],
        )
        tree = builder.build_tree(simplified)
        root = tree.root
        assert root.node_type == "split_pair_container"
        assert root.rooms == frozenset({1, 2})

        # The C4 child should be attached directly to the root container because anchor 1 is a pole
        c4_children = [c for c in root.children if c.rooms == frozenset({5, 6, 7})]
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
            1: Room(vnum=1, name="R1", description="", exits=(Exit(direction=Direction.north, dst_vnum=2), Exit(direction=Direction.east, dst_vnum=3))),
            2: Room(vnum=2, name="R2", description="", exits=(Exit(direction=Direction.south, dst_vnum=1), Exit(direction=Direction.east, dst_vnum=4))),
            3: Room(vnum=3, name="R3", description="", exits=(Exit(direction=Direction.west, dst_vnum=1), Exit(direction=Direction.north, dst_vnum=4))),
            4: Room(vnum=4, name="R4", description="", exits=(Exit(direction=Direction.west, dst_vnum=2), Exit(direction=Direction.south, dst_vnum=3))),
        }
        all_exits = tuple(e for r in rooms.values() for e in r.exits)
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)
        solver = HierarchicalLayoutSolver(tree_builder=SpqrTreeBuilder())
        solved = solver.solve(world_graph, LayoutOptions())

        assert isinstance(solved, SolvedLayout)
        assert solved.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert set(solved.coordinates.keys()) == {1, 2, 3, 4}

    def test_solver_with_spqr_school_fixture(self) -> None:
        """HierarchicalLayoutSolver solves triconnected graph with SpqrTreeBuilder."""
        # 6-room graph: 5-room wheel graph with attached room 6
        rooms = {
            1: _make_room(1, [
                Exit(direction=int(Direction.north), dst_vnum=2),
                Exit(direction=int(Direction.east), dst_vnum=3),
                Exit(direction=int(Direction.south), dst_vnum=4),
                Exit(direction=int(Direction.west), dst_vnum=5),
            ]),
            2: _make_room(2, [
                Exit(direction=int(Direction.south), dst_vnum=1),
                Exit(direction=int(Direction.east), dst_vnum=3),
                Exit(direction=int(Direction.west), dst_vnum=5),
            ]),
            3: _make_room(3, [
                Exit(direction=int(Direction.west), dst_vnum=1),
                Exit(direction=int(Direction.south), dst_vnum=4),
                Exit(direction=int(Direction.north), dst_vnum=2),
            ]),
            4: _make_room(4, [
                Exit(direction=int(Direction.north), dst_vnum=1),
                Exit(direction=int(Direction.west), dst_vnum=5),
                Exit(direction=int(Direction.east), dst_vnum=3),
                Exit(direction=int(Direction.south), dst_vnum=6),
            ]),
            5: _make_room(5, [
                Exit(direction=int(Direction.east), dst_vnum=1),
                Exit(direction=int(Direction.north), dst_vnum=2),
                Exit(direction=int(Direction.south), dst_vnum=4),
            ]),
            6: _make_room(6, [
                Exit(direction=int(Direction.north), dst_vnum=4),
            ]),
        }
        all_exits = [e for r in rooms.values() for e in r.exits]
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)

        solver = HierarchicalLayoutSolver(tree_builder=SpqrTreeBuilder())
        solved = solver.solve(world_graph, LayoutOptions())

        assert isinstance(solved, SolvedLayout)
        assert solved.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(solved.coordinates) == len(world_graph.rooms)

    test_solver_with_spqr_synthetic_fixture = test_solver_with_spqr_school_fixture


class TestSafe2CutAllowlistContract:
    """Unit tests for the safe 2-cut allowlist contract and combined child fusion."""

    def test_safe_series_branches_decouple_cleanly(self) -> None:
        """Candidate series branches expanding into disjoint half-spaces decouple as independent children."""
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=2, direction=Direction.west),
            LayoutEdge(src=1, dst=4, direction=Direction.west),
            LayoutEdge(src=4, dst=2, direction=Direction.east),
        ]
        raw_node = build_block_spqr_tree(frozenset({1, 2, 3, 4}), edges)
        assert raw_node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert raw_node.rooms == frozenset({1, 2})
        assert raw_node.metadata.get("cut_type") == CutType.SPLIT_PAIR
        assert set(raw_node.metadata.get("poles", ())) == {1, 2}
        assert len(raw_node.children) == 2
        for child in raw_node.children:
            assert child.node_type == TreeNodeType.SERIES_LOOP

    def test_interlocked_branches_span_conflict_decomposition(self) -> None:
        """A rigid internal block pinning pole span alongside peer series branch decomposes into a split-pair container."""
        edges: list[LayoutEdge] = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=4, direction=Direction.north),
            LayoutEdge(src=4, dst=2, direction=Direction.west),
            LayoutEdge(src=1, dst=4, direction=Direction.northeast),
            LayoutEdge(src=2, dst=3, direction=Direction.southeast),
            LayoutEdge(src=1, dst=20, direction=Direction.south),
            LayoutEdge(src=20, dst=21, direction=Direction.east),
            LayoutEdge(src=21, dst=2, direction=Direction.north),
        ]
        raw_node = build_block_spqr_tree(frozenset({1, 2, 3, 4, 20, 21}), edges)
        assert raw_node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert raw_node.rooms == frozenset({1, 2})
        assert set(raw_node.metadata.get("poles", ())) == {1, 2}
        assert len(raw_node.children) == 2
        child_types = {c.node_type for c in raw_node.children}
        assert child_types == {TreeNodeType.RIGID_CLUSTER, TreeNodeType.SERIES_LOOP}

    def test_interlocked_branches_concentric_enclosure_decomposition(self) -> None:
        """Concentric arches around the pole axis decompose into a split-pair container with series branches."""
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            # Inner branch in +X half-space
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=2, direction=Direction.west),
            # Outer arch with transverse bracketing (spans both +X and -X)
            LayoutEdge(src=1, dst=4, direction=Direction.south),
            LayoutEdge(src=4, dst=5, direction=Direction.east),
            LayoutEdge(src=5, dst=6, direction=Direction.north),
            LayoutEdge(src=6, dst=7, direction=Direction.west),
            LayoutEdge(src=7, dst=8, direction=Direction.south),
            LayoutEdge(src=8, dst=2, direction=Direction.east),
        ]
        raw_node = build_block_spqr_tree(frozenset({1, 2, 3, 4, 5, 6, 7, 8}), edges)
        assert raw_node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert raw_node.rooms == frozenset({1, 2})
        assert len(raw_node.children) == 2
        assert all(c.node_type == TreeNodeType.SERIES_LOOP for c in raw_node.children)

    def test_interlocked_branches_overlapping_halfspace_decomposition(self) -> None:
        """Multiple candidate branches expanding into the same half-space decompose into a split-pair container."""
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            # Branch 1 in +X half-space
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=30, direction=Direction.north),
            LayoutEdge(src=30, dst=2, direction=Direction.west),
            # Branch 2 also in +X half-space
            LayoutEdge(src=1, dst=4, direction=Direction.south),
            LayoutEdge(src=4, dst=5, direction=Direction.east),
            LayoutEdge(src=5, dst=6, direction=Direction.north),
            LayoutEdge(src=6, dst=7, direction=Direction.west),
            LayoutEdge(src=7, dst=2, direction=Direction.south),
        ]
        raw_node = build_block_spqr_tree(frozenset({1, 2, 3, 30, 4, 5, 6, 7}), edges)
        assert raw_node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert raw_node.rooms == frozenset({1, 2})
        assert len(raw_node.children) == 2
        assert all(c.node_type == TreeNodeType.SERIES_LOOP for c in raw_node.children)

    def test_split_pair_with_outer_parent_pipeline_decomposition(self) -> None:
        """Split pair with outer parent and conflicting branches decomposes into hierarchical solve tree."""
        builder = SpqrTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[100, 101, 102, 103, 1, 2, 3, 4, 20, 21, 22, 23],
            directed_edges=[
                # Outer parent rigid cluster (3-connected square with diagonals)
                (100, 101, Direction.north),
                (101, 102, Direction.west),
                (102, 103, Direction.south),
                (103, 100, Direction.east),
                (100, 102, Direction.northwest),
                (103, 101, Direction.northeast),
                # Attachment to poles
                (100, 1, Direction.east),
                (101, 2, Direction.east),
                # Direct edge between poles
                (1, 2, Direction.north),
                # Rigid K4 branch pinning span = 1
                (1, 3, Direction.east),
                (3, 4, Direction.north),
                (4, 2, Direction.west),
                (1, 4, Direction.northeast),
                (2, 3, Direction.southeast),
                # Longer series loop branch requiring span >= 3
                (1, 20, Direction.south),
                (20, 21, Direction.east),
                (21, 22, Direction.north),
                (22, 23, Direction.west),
                (23, 2, Direction.south),
            ],
        )
        tree = builder.build_tree(simplified)
        assert tree.root.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert tree.root.rooms == frozenset({1, 2})
        assert len(tree.root.children) == 3


class TestTransverseCorridorSNodeSplitting:
    """Unit and integration tests for SPQR tree-level transverse corridor S-node splitting."""

    def test_transverse_corridor_s_node_split_detection_and_severing(self) -> None:
        """Series loop with parallel entrance corridors severs transverse corridor into two sibling clusters."""
        # Parent cluster has rooms {1, 2}
        # S-node child across poles {1, 2} has rooms {1, 2, 10, 11, 20}
        # Entrance from 1 to 10 is West
        # Entrance from 2 to 20 is West
        # Internal spine corridor from 10 to 11 is North (transverse)
        # Internal spine corridor from 11 to 20 is North
        edges = [
            LayoutEdge(src=1, dst=10, direction=Direction.west),
            LayoutEdge(src=10, dst=11, direction=Direction.north),
            LayoutEdge(src=11, dst=20, direction=Direction.north),
            LayoutEdge(src=2, dst=20, direction=Direction.west),
        ]
        graph = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 10, 11, 20)},
            edges=tuple(edges),
        )

        child = SolveTreeNode(
            node_id="s_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )

        transformed = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(transformed.children) == 2
        c_a, c_b = transformed.children

        assert c_a.node_type == TreeNodeType.RIGID_CLUSTER
        assert c_b.node_type == TreeNodeType.RIGID_CLUSTER
        assert isinstance(c_a.cut_contract, BridgeCut)
        assert isinstance(c_b.cut_contract, BridgeCut)

        # Sibling clusters are anchored to the respective poles
        anchors = {_get_anchor_vnum(c_a), _get_anchor_vnum(c_b)}
        assert anchors == {1, 2}

        # Rooms are partitioned across the severed transverse corridor (poles owned by ancestor)
        assert c_a.rooms | c_b.rooms == frozenset({10, 11, 20})
        assert c_a.rooms & c_b.rooms == frozenset()

        if _get_anchor_vnum(c_a) == 1:
            assert c_a.rooms == frozenset({10})
            assert c_b.rooms == frozenset({11, 20})
        else:
            assert c_b.rooms == frozenset({10})
            assert c_a.rooms == frozenset({11, 20})

    def test_transverse_corridor_nested_child_partition(self) -> None:
        """Nested child clusters of an S-node partition to their respective split halves."""
        edges = [
            LayoutEdge(src=1, dst=10, direction=Direction.west),
            LayoutEdge(src=10, dst=11, direction=Direction.north),
            LayoutEdge(src=11, dst=20, direction=Direction.north),
            LayoutEdge(src=2, dst=20, direction=Direction.west),
            LayoutEdge(src=10, dst=100, direction=Direction.up),
            LayoutEdge(src=20, dst=200, direction=Direction.up),
        ]
        graph = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 10, 11, 20, 100, 200)},
            edges=tuple(edges),
        )

        nested_a = SolveTreeNode(
            node_id="nested_tower_1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 100}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 10,
                "cut_contract": ArticulationPointCut(anchor_vnum=10),
            },
        )
        nested_b = SolveTreeNode(
            node_id="nested_tower_2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({20, 200}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 20,
                "cut_contract": ArticulationPointCut(anchor_vnum=20),
            },
        )

        child = SolveTreeNode(
            node_id="s_loop_with_nested",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(nested_a, nested_b),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )

        transformed = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(transformed.children) == 2
        c_a, c_b = transformed.children

        if _get_anchor_vnum(c_a) == 1:
            assert len(c_a.children) == 1
            assert c_a.children[0].node_id == "nested_tower_1"
            assert len(c_b.children) == 1
            assert c_b.children[0].node_id == "nested_tower_2"
        else:
            assert len(c_b.children) == 1
            assert c_b.children[0].node_id == "nested_tower_1"
            assert len(c_a.children) == 1
            assert c_a.children[0].node_id == "nested_tower_2"

    def test_perpendicular_entrance_corridors_not_split(self) -> None:
        """When entrance corridors are perpendicular, S-node is not split."""
        # Pole 1 connects West, but pole 2 connects North (different cardinal axes)
        edges = [
            LayoutEdge(src=1, dst=10, direction=Direction.west),
            LayoutEdge(src=10, dst=11, direction=Direction.north),
            LayoutEdge(src=11, dst=20, direction=Direction.east),
            LayoutEdge(src=2, dst=20, direction=Direction.north),
        ]
        graph = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 10, 11, 20)},
            edges=tuple(edges),
        )

        child = SolveTreeNode(
            node_id="s_loop_perp",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )

        transformed = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(transformed.children) == 1
        assert transformed.children[0].node_id == "s_loop_perp"

    def test_no_transverse_internal_corridor_not_split(self) -> None:
        """When S-node spine contains no transverse corridors, S-node is not split."""
        # Entrances along X (West), but internal corridor is also along X (West)
        edges = [
            LayoutEdge(src=1, dst=10, direction=Direction.west),
            LayoutEdge(src=10, dst=20, direction=Direction.west),
            LayoutEdge(src=2, dst=20, direction=Direction.west),
        ]
        graph = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 10, 20)},
            edges=tuple(edges),
        )

        child = SolveTreeNode(
            node_id="s_loop_collinear",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 20}),
            children=(),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )

        transformed = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(transformed.children) == 1
        assert transformed.children[0].node_id == "s_loop_collinear"

    def test_thalos_area_decomposition_splits_west_watchtowers(self) -> None:
        """SpqrTreeBuilder automatically splits the West watchtowers S-node in Thalos into two sibling clusters."""
        thalos_area = parse_file("tests/fixtures/areas/thalos.are")
        world_graph = WorldGraph.from_area_data(thalos_area)
        solver = HierarchicalLayoutSolver.production()
        layout_graph = LayoutGraph.from_world_graph(world_graph)
        simplified = solver.simplifier.simplify(layout_graph)
        tree = solver.tree_builder.build_tree(simplified)

        # West watchtowers S-node across poles 5209 and 5214 is split into sibling clusters
        # Root cluster must have sibling children anchored at 5209 and 5214
        root = tree.root
        anchors = {_get_anchor_vnum(c) for c in root.children}
        assert 5209 in anchors
        assert 5214 in anchors

        child_5209 = [c for c in root.children if _get_anchor_vnum(c) == 5209][0]
        child_5214 = [c for c in root.children if _get_anchor_vnum(c) == 5214][0]

        # Room 5224 belongs to child_5209, room 5215 belongs to child_5214
        assert 5224 in child_5209.rooms
        assert 5215 in child_5214.rooms

    def test_canonical_spqr_root_preserves_natural_root_without_greedy_inversion(self) -> None:
        """SPQR tree root reflects the canonical triconnected decomposition without greedy size-based rerooting."""
        # Dipole graph across poles {1, 2} with three independent paths
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=10, direction=Direction.east),
            LayoutEdge(src=10, dst=2, direction=Direction.west),
            LayoutEdge(src=1, dst=20, direction=Direction.west),
            LayoutEdge(src=20, dst=2, direction=Direction.east),
        ]
        node = build_block_spqr_tree(frozenset({1, 2, 10, 20}), edges)
        assert node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert node.rooms == frozenset({1, 2})
        assert len(node.children) == 2
        for child in node.children:
            assert child.node_type == TreeNodeType.SERIES_LOOP

    def test_reroot_spqr_tree_identity(self) -> None:
        """Rerooting at the current root returns the root node unchanged."""
        import spqrtree

        mg = spqrtree.MultiGraph()
        for u, v in [(1, 2), (2, 3), (3, 1), (2, 4), (4, 3)]:
            mg.add_edge(u, v)
        root = spqrtree.build_spqr_tree(mg)
        rerooted = reroot_spqr_tree(root, root)
        assert rerooted is root
        assert rerooted.parent is None

    def test_reroot_spqr_tree_inverts_parent_child_hierarchy(self) -> None:
        """Rerooting inverts parent-child relationships and updates attachment poles along the path."""
        import spqrtree

        mg = spqrtree.MultiGraph()
        for u, v in [(1, 2), (2, 3), (3, 1), (2, 4), (4, 3)]:
            mg.add_edge(u, v)
        root = spqrtree.build_spqr_tree(mg)
        assert len(root.children) >= 1
        child = root.children[0]

        rerooted = reroot_spqr_tree(root, child)
        assert rerooted is child
        assert rerooted.parent is None
        assert root in rerooted.children
        assert root.parent is rerooted

    def test_reroot_spqr_tree_unreachable_node_raises_value_error(self) -> None:
        """Attempting to reroot to a disconnected node raises ValueError."""
        import spqrtree

        mg = spqrtree.MultiGraph()
        for u, v in [(1, 2), (2, 1)]:
            mg.add_edge(u, v)
        root = spqrtree.build_spqr_tree(mg)
        foreign_node = spqrtree.SPQRNode(
            type=spqrtree.NodeType.S,
            skeleton=spqrtree.MultiGraph(),
            poles=(99, 100),
            parent=None,
        )
        with pytest.raises(ValueError, match="not reachable"):
            reroot_spqr_tree(root, foreign_node)

    def test_anchor_centric_spqr_rooting_places_anchor_at_root(self) -> None:
        """When an anchor vertex is specified, the SPQR tree reroots at the component containing the anchor."""
        from romutil.layout import ChainedSimplifier, HallwayCompactor, WarpSeverer
        from romutil.layout.hierarchy.transforms import strip_anchor
        from romutil.parser import parse_file

        area = parse_file("tests/fixtures/areas/sewer.are")
        wg = WorldGraph.from_area_data(area)
        lg = LayoutGraph.from_world_graph(wg)
        simp = ChainedSimplifier([HallwayCompactor(), WarpSeverer()]).simplify(lg, LayoutOptions())
        anchor = 7402
        b9_rooms = {
            7402, 7403, 7404, 7406, 7407, 7409, 7411, 7412, 7414, 7417, 7418,
            7419, 7420, 7422, 7423, 7426, 7431, 7433, 7434, 7436, 7437, 7438,
            7439, 7440, 7442, 7443, 7444, 7445,
        }
        spqr_unanchored = build_block_spqr_tree(frozenset(b9_rooms), simp.graph.edges, anchor=None)
        assert anchor not in spqr_unanchored.rooms

        spqr_anchored = build_block_spqr_tree(frozenset(b9_rooms), simp.graph.edges, anchor=anchor)
        assert anchor in spqr_anchored.rooms

        spqr_stripped = strip_anchor(spqr_anchored, anchor)
        assert anchor not in spqr_stripped.rooms
        assert 7403 in spqr_stripped.rooms
        assert 7418 in spqr_stripped.rooms
