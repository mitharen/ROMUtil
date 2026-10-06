"""Comprehensive unit tests for 0-cuts and 1-cut block-cut tree decomposition."""

from __future__ import annotations

import pytest

from romutil.layout import (
    BlockCutTreeBuilder,
    HierarchicalLayoutSolver,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeBuilder,
)
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, LayoutOptions, SolvedLayout, SolverStatus, WorldGraph


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
) -> SimplifiedGraph:
    """Helper to construct a SimplifiedGraph from node lists and edge pairs."""
    room_defs = {v: _make_room(v) for v in rooms}
    layout_edges: list[LayoutEdge] = []
    if edges:
        for u, v in edges:
            layout_edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north, cut=False))
    if cut_edges:
        for u, v in cut_edges:
            layout_edges.append(LayoutEdge(src=u, dst=v, direction=Direction.north, cut=True))
    layout_graph = LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))
    return SimplifiedGraph(graph=layout_graph)


class TestBlockCutTreeBuilderProtocol:
    """Verify runtime protocol conformance."""

    def test_implements_tree_builder_protocol(self) -> None:
        builder = BlockCutTreeBuilder()
        assert isinstance(builder, TreeBuilder)


class TestBlockCutTreeEmptyAndIsolated:
    """Unit tests for empty graphs and isolated single rooms."""

    def test_empty_graph(self) -> None:
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph([])
        tree = builder.build_tree(simplified)

        assert isinstance(tree, SolveTree)
        assert tree.root.node_id == "empty"
        assert tree.root.node_type == "empty"
        assert tree.root.rooms == frozenset()
        assert tree.root.children == ()
        assert tree.components == ()

    def test_single_isolated_room(self) -> None:
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph([42])
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        assert tree.root == tree.components[0]
        assert tree.root.node_id == "comp_0_block_0"
        assert tree.root.node_type == "biconnected_block"
        assert tree.root.rooms == frozenset({42})
        assert tree.root.children == ()
        assert tree.root.metadata == {}

    def test_multiple_isolated_rooms(self) -> None:
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph([10, 20])
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 2
        assert tree.root.node_id == "world_root"
        assert tree.root.node_type == "world_root"
        assert tree.root.rooms == frozenset()
        assert len(tree.root.children) == 2
        assert tree.components == tree.root.children

        c0, c1 = tree.components
        assert c0.node_id == "comp_0_block_0"
        assert c0.node_type == "biconnected_block"
        assert c0.rooms == frozenset({10})

        assert c1.node_id == "comp_1_block_0"
        assert c1.node_type == "biconnected_block"
        assert c1.rooms == frozenset({20})


class TestBlockCutTreeBiconnected:
    """Unit tests for single biconnected blocks (cycles, complete subgraphs)."""

    def test_single_biconnected_triangle(self) -> None:
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3],
            edges=[(1, 2), (2, 3), (3, 1)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        assert tree.root == tree.components[0]
        assert tree.root.node_id == "comp_0_block_0"
        assert tree.root.node_type == "biconnected_block"
        assert tree.root.rooms == frozenset({1, 2, 3})
        assert tree.root.children == ()
        assert tree.root.metadata == {}

    def test_single_biconnected_4cycle(self) -> None:
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[10, 20, 30, 40],
            edges=[(10, 20), (20, 30), (30, 40), (40, 10)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        assert tree.root.node_id == "comp_0_block_0"
        assert tree.root.node_type == "biconnected_block"
        assert tree.root.rooms == frozenset({10, 20, 30, 40})
        assert tree.root.children == ()


class TestBlockCutTreeZeroCuts:
    """Unit tests for Stage 1: 0-cuts (connected components)."""

    def test_multiple_disconnected_components(self) -> None:
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6],
            edges=[(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 2
        assert tree.root.node_id == "world_root"
        assert tree.root.node_type == "world_root"
        assert tree.root.rooms == frozenset()
        assert len(tree.root.children) == 2

        c0, c1 = tree.components
        assert c0.node_id == "comp_0_block_0"
        assert c0.rooms == frozenset({1, 2, 3})
        assert c0.node_type == "biconnected_block"

        assert c1.node_id == "comp_1_block_0"
        assert c1.rooms == frozenset({4, 5, 6})
        assert c1.node_type == "biconnected_block"

    def test_component_size_ordering_descending(self) -> None:
        builder = BlockCutTreeBuilder()
        # Small component {1, 2}, large component {10, 20, 30, 40}
        simplified = _make_simplified_graph(
            rooms=[1, 2, 10, 20, 30, 40],
            edges=[(1, 2), (10, 20), (20, 30), (30, 40), (40, 10)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 2
        c0, c1 = tree.components
        # Larger component must be comp_0
        assert c0.node_id == "comp_0_block_0"
        assert c0.rooms == frozenset({10, 20, 30, 40})
        # Smaller component must be comp_1
        assert c1.node_id == "comp_1_block_0"
        assert c1.rooms == frozenset({1, 2})


class TestBlockCutTreeOneCuts:
    """Unit tests for Stage 3 Tier 1: 1-cuts (articulation points and bridges)."""

    def test_articulation_point_two_triangles(self) -> None:
        # Two triangles sharing vertex 3: (1-2-3-1) and (3-4-5-3)
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5],
            edges=[(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        root = tree.root
        assert root.node_id == "comp_0_block_0"
        assert root.node_type == "block"
        assert root.rooms == frozenset({1, 2, 3})
        assert root.metadata == {}
        assert len(root.children) == 1

        child = root.children[0]
        assert child.node_id == "comp_0_block_1"
        assert child.node_type == "block"
        assert child.rooms == frozenset({3, 4, 5})
        assert child.children == ()
        assert child.metadata == {
            "cut_type": "articulation_point",
            "anchor_vnum": 3,
        }

    def test_bridge_barbell_graph(self) -> None:
        # Barbell: Triangle {1, 2, 3}, Bridge {3, 4}, Triangle {4, 5, 6}
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6],
            edges=[(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        root = tree.root
        assert root.node_id == "comp_0_block_0"
        assert root.node_type == "block"
        assert root.rooms == frozenset({1, 2, 3})
        assert root.metadata == {}
        assert len(root.children) == 1

        bridge_node = root.children[0]
        assert bridge_node.node_id == "comp_0_block_1"
        assert bridge_node.node_type == "block"
        assert bridge_node.rooms == frozenset({3, 4})
        assert bridge_node.metadata == {
            "cut_type": "articulation_point",
            "anchor_vnum": 3,
        }
        assert len(bridge_node.children) == 1

        t2_node = bridge_node.children[0]
        assert t2_node.node_id == "comp_0_block_2"
        assert t2_node.node_type == "block"
        assert t2_node.rooms == frozenset({4, 5, 6})
        assert t2_node.metadata == {
            "cut_type": "articulation_point",
            "anchor_vnum": 4,
        }
        assert t2_node.children == ()

    def test_deep_articulation_tree_chain(self) -> None:
        # Chain of 4 blocks linked via articulation points 3, 5, 7
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            edges=[
                (1, 2), (2, 3), (3, 1),
                (3, 4), (4, 5), (5, 3),
                (5, 6), (6, 7), (7, 5),
                (7, 8), (8, 9), (9, 7),
            ],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        b0 = tree.root
        assert b0.node_id == "comp_0_block_0"
        assert b0.rooms == frozenset({1, 2, 3})
        assert len(b0.children) == 1

        b1 = b0.children[0]
        assert b1.node_id == "comp_0_block_1"
        assert b1.rooms == frozenset({3, 4, 5})
        assert b1.metadata["anchor_vnum"] == 3
        assert len(b1.children) == 1

        b2 = b1.children[0]
        assert b2.node_id == "comp_0_block_2"
        assert b2.rooms == frozenset({5, 6, 7})
        assert b2.metadata["anchor_vnum"] == 5
        assert len(b2.children) == 1

        b3 = b2.children[0]
        assert b3.node_id == "comp_0_block_3"
        assert b3.rooms == frozenset({7, 8, 9})
        assert b3.metadata["anchor_vnum"] == 7
        assert b3.children == ()

    def test_star_articulation_multiple_children(self) -> None:
        # Star of 3 triangles meeting at central vertex 10
        # T1: 10, 1, 2
        # T2: 10, 3, 4
        # T3: 10, 5, 6
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3, 4, 5, 6, 10],
            edges=[
                (10, 1), (1, 2), (2, 10),
                (10, 3), (3, 4), (4, 10),
                (10, 5), (5, 6), (6, 10),
            ],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        root = tree.root
        assert root.node_id == "comp_0_block_0"
        assert root.rooms == frozenset({1, 2, 10})
        assert len(root.children) == 2

        c1, c2 = root.children
        assert c1.node_id == "comp_0_block_1"
        assert c1.rooms == frozenset({3, 4, 10})
        assert c1.metadata == {"cut_type": "articulation_point", "anchor_vnum": 10}

        assert c2.node_id == "comp_0_block_2"
        assert c2.rooms == frozenset({5, 6, 10})
        assert c2.metadata == {"cut_type": "articulation_point", "anchor_vnum": 10}

    def test_primary_largest_block_root_selection(self) -> None:
        # 3-node block {10, 1, 2} sharing vertex 10 with 4-node block {10, 20, 21, 22}
        # Largest node count must be selected as root even if it has higher vnums
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 10, 20, 21, 22],
            edges=[
                (10, 1), (1, 2), (2, 10),
                (10, 20), (20, 21), (21, 22), (22, 10),
            ],
        )
        tree = builder.build_tree(simplified)

        root = tree.root
        assert root.node_id == "comp_0_block_0"
        # 4-cycle {10, 20, 21, 22} has size 4, triangle has size 3
        assert root.rooms == frozenset({10, 20, 21, 22})
        assert len(root.children) == 1

        child = root.children[0]
        assert child.node_id == "comp_0_block_1"
        assert child.rooms == frozenset({1, 2, 10})
        assert child.metadata["anchor_vnum"] == 10


class TestBlockCutTreeEdgeFiltering:
    """Unit tests verifying active edge filtering and ghost room handling."""

    def test_cut_edges_ignored_producing_articulation(self) -> None:
        # Triangle 1-2-3, but edge 2-3 is cut -> path 2-1-3, vertex 1 is articulation point
        builder = BlockCutTreeBuilder()
        simplified = _make_simplified_graph(
            rooms=[1, 2, 3],
            edges=[(1, 2), (1, 3)],
            cut_edges=[(2, 3)],
        )
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        root = tree.root
        assert root.node_type == "block"
        assert len(root.children) == 1
        child = root.children[0]
        assert child.metadata["anchor_vnum"] == 1

    def test_edges_referencing_nonexistent_rooms_ignored(self) -> None:
        # Edge pointing to room 999 which is not registered in rooms
        builder = BlockCutTreeBuilder()
        room_defs = {1: _make_room(1), 2: _make_room(2)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=0, cut=False),
            LayoutEdge(src=2, dst=999, direction=0, cut=False),
        ]
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms=room_defs, edges=tuple(edges)))
        tree = builder.build_tree(simplified)

        assert len(tree.components) == 1
        assert tree.root.rooms == frozenset({1, 2})


class TestHierarchicalLayoutSolverIntegration:
    """End-to-end integration tests using BlockCutTreeBuilder in HierarchicalLayoutSolver."""

    def test_end_to_end_solve_articulated_world_graph(self) -> None:
        # Build WorldGraph representing barbell graph
        ex1 = Exit(direction=Direction.east, dst_vnum=2)
        ex2 = Exit(direction=Direction.south, dst_vnum=3)
        ex3 = Exit(direction=Direction.northwest, dst_vnum=1)
        ex4 = Exit(direction=Direction.east, dst_vnum=4)
        ex5 = Exit(direction=Direction.east, dst_vnum=5)
        ex6 = Exit(direction=Direction.south, dst_vnum=6)
        ex7 = Exit(direction=Direction.northwest, dst_vnum=4)

        r1 = _make_room(1, [ex1])
        r2 = _make_room(2, [ex2])
        r3 = _make_room(3, [ex3, ex4])
        r4 = _make_room(4, [ex5])
        r5 = _make_room(5, [ex6])
        r6 = _make_room(6, [ex7])

        world_graph = WorldGraph(
            rooms={1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6},
            exits=(ex1, ex2, ex3, ex4, ex5, ex6, ex7),
        )

        solver = HierarchicalLayoutSolver(tree_builder=BlockCutTreeBuilder())
        solved = solver.solve(world_graph, LayoutOptions())

        assert isinstance(solved, SolvedLayout)
        assert solved.diagnostics.status == SolverStatus.FEASIBLE
        assert set(solved.coordinates.keys()) == {1, 2, 3, 4, 5, 6}
        for vnum, coord in solved.coordinates.items():
            assert isinstance(coord, Coord3D)

    def test_end_to_end_solve_multiple_components(self) -> None:
        # Two disconnected triangles in WorldGraph
        ex1 = Exit(direction=Direction.east, dst_vnum=2)
        ex2 = Exit(direction=Direction.south, dst_vnum=3)
        ex3 = Exit(direction=Direction.northwest, dst_vnum=1)
        ex4 = Exit(direction=Direction.east, dst_vnum=5)
        ex5 = Exit(direction=Direction.south, dst_vnum=6)
        ex6 = Exit(direction=Direction.northwest, dst_vnum=4)

        r1 = _make_room(1, [ex1])
        r2 = _make_room(2, [ex2])
        r3 = _make_room(3, [ex3])
        r4 = _make_room(4, [ex4])
        r5 = _make_room(5, [ex5])
        r6 = _make_room(6, [ex6])

        world_graph = WorldGraph(
            rooms={1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6},
            exits=(ex1, ex2, ex3, ex4, ex5, ex6),
        )

        solver = HierarchicalLayoutSolver(tree_builder=BlockCutTreeBuilder())
        solved = solver.solve(world_graph)

        assert isinstance(solved, SolvedLayout)
        assert solved.diagnostics.status == SolverStatus.FEASIBLE
        assert set(solved.coordinates.keys()) == {1, 2, 3, 4, 5, 6}

    def test_end_to_end_solve_empty_graph(self) -> None:
        world_graph = WorldGraph(rooms={}, exits=(), header=None)
        solver = HierarchicalLayoutSolver(tree_builder=BlockCutTreeBuilder())
        solved = solver.solve(world_graph)

        assert isinstance(solved, SolvedLayout)
        assert solved.coordinates == {}
