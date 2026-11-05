"""Comprehensive unit tests for hierarchical tree solver in romutil.layout.milp.tree."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
    TreeSolution,
)
import romutil.layout.milp.cluster as cluster_module
from romutil.layout.milp import (
    HierarchicalTreeSolver as TopLevelTreeSolver,
    pack_world_root_components,
    solve_cluster,
)
from romutil.layout.milp.tree import HierarchicalTreeSolver
from romutil.layout.protocols import TreeSolver
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus


def _make_room(vnum: int, exits: dict[Direction, int] | None = None) -> Room:
    exits_dict = exits or {}
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(
            Exit(direction=d, dst_vnum=to_v)
            for d, to_v in exits_dict.items()
        ),
    )


class TestHierarchicalTreeSolverBasics:
    """Basic initialization and protocol conformance tests."""

    def test_implements_tree_solver_protocol(self) -> None:
        """HierarchicalTreeSolver is an instance of TreeSolver protocol."""
        solver = HierarchicalTreeSolver()
        assert isinstance(solver, TreeSolver)

    def test_canonical_and_top_level_exports(self) -> None:
        """HierarchicalTreeSolver is identically exported from tree module and milp package."""
        assert HierarchicalTreeSolver is TopLevelTreeSolver
        assert "HierarchicalTreeSolver" not in cluster_module.__all__

    def test_default_and_custom_initialization(self) -> None:
        """Initializes with default or custom solver name and graph."""
        solver_default = HierarchicalTreeSolver()
        assert solver_default.graph is None
        assert solver_default.solver_name == "cbc"

        g = LayoutGraph(rooms={1: _make_room(1)}, edges=())
        solver_custom = HierarchicalTreeSolver(graph=g, solver_name="glpk")
        assert solver_custom.graph is g
        assert solver_custom.solver_name == "glpk"


class TestHierarchicalTreeSolverExecution:
    """Bottom-up traversal and cluster solving execution tests."""

    def test_solve_single_leaf_root(self) -> None:
        """Solves a tree containing a single leaf cluster."""
        r1 = _make_room(1, {Direction.east: 2})
        r2 = _make_room(2, {Direction.west: 1})
        graph = LayoutGraph(rooms={1: r1, 2: r2}, edges=(LayoutEdge(src=1, dst=2, direction=Direction.east),))

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(),
        )
        tree = SolveTree(root=root, graph=graph)
        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert isinstance(sol, TreeSolution)
        assert "root" in sol.node_solutions
        root_sol = sol.node_solutions["root"]
        assert 1 in root_sol.local_coordinates
        assert 2 in root_sol.local_coordinates

    def test_solve_nested_hierarchy_post_order(self) -> None:
        """Solves nested hierarchy bottom-up, passing child solutions to parent container."""
        # Child 1: rooms 10, 11
        # Child 2: rooms 20, 21
        # Root: parent container containing child 1 and child 2
        r10 = _make_room(10, {Direction.east: 11})
        r11 = _make_room(11, {Direction.west: 10, Direction.north: 20})
        r20 = _make_room(20, {Direction.south: 11, Direction.east: 21})
        r21 = _make_room(21, {Direction.west: 20})

        graph = LayoutGraph(
            rooms={10: r10, 11: r11, 20: r20, 21: r21},
            edges=(
                LayoutEdge(src=10, dst=11, direction=Direction.east),
                LayoutEdge(src=11, dst=20, direction=Direction.north),
                LayoutEdge(src=20, dst=21, direction=Direction.east),
            ),
        )

        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 11}),
            children=(),
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({20, 21}),
            children=(),
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(),
            children=(c1, c2),
        )
        tree = SolveTree(root=root, graph=graph)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "c1" in sol.node_solutions
        assert "c2" in sol.node_solutions
        assert "root" in sol.node_solutions
        root_sol = sol.node_solutions["root"]
        child_offsets = root_sol.metadata.get("child_offsets", {})
        assert "c1" in child_offsets
        assert "c2" in child_offsets

    def test_solve_with_detached_components(self) -> None:
        """Solves both root and additional independent components in tree.components."""
        r1 = _make_room(1)
        r2 = _make_room(2)
        graph = LayoutGraph(rooms={1: r1, 2: r2}, edges=())

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        comp = SolveTreeNode(
            node_id="comp",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({2}),
            children=(),
        )
        tree = SolveTree(root=root, components=(comp,), graph=graph)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "root" in sol.node_solutions
        assert "comp" in sol.node_solutions
        assert 1 in sol.node_solutions["root"].local_coordinates
        assert 2 in sol.node_solutions["comp"].local_coordinates

    def test_synthesizes_graph_when_none_provided(self) -> None:
        """Collects all rooms across root and components when layout graph is None."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({101, 102}),
            children=(),
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100}),
            children=(c1,),
        )
        comp = SolveTreeNode(
            node_id="comp",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({200}),
            children=(),
        )
        tree = SolveTree(root=root, components=(comp,), graph=None)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "c1" in sol.node_solutions
        assert "root" in sol.node_solutions
        assert "comp" in sol.node_solutions
        assert 100 in sol.node_solutions["root"].local_coordinates
        assert 101 in sol.node_solutions["c1"].local_coordinates
        assert 200 in sol.node_solutions["comp"].local_coordinates

    def test_custom_graph_override(self) -> None:
        """Solver instance graph overrides tree graph when explicitly passed."""
        r1 = _make_room(1)
        g_custom = LayoutGraph(rooms={1: r1}, edges=())
        g_tree = LayoutGraph(rooms={}, edges=())

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        tree = SolveTree(root=root, graph=g_tree)

        solver = HierarchicalTreeSolver(graph=g_custom)
        sol = solver.solve_tree(tree)
        assert 1 in sol.node_solutions["root"].local_coordinates

    def test_node_caching_behavior(self) -> None:
        """Returns cached node solution if the same node is encountered multiple times."""
        child = SolveTreeNode(
            node_id="shared_child",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({5}),
            children=(),
        )
        # Tree where root has the child, and components also references the same child node
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(),
            children=(child,),
        )
        tree = SolveTree(root=root, components=(child,), graph=None)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "shared_child" in sol.node_solutions
        assert "root" in sol.node_solutions
        assert len(sol.node_solutions) == 2

    def test_forwards_options_and_solver_name(self) -> None:
        """Forwards solver options and solver name to solve_cluster."""
        node = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        tree = SolveTree(root=node, graph=None)
        options = LayoutOptions(timeout_seconds=42)

        with patch("romutil.layout.milp.tree.solve_cluster") as mock_solve:
            mock_solve.return_value = NodeSolution(
                node_id="root",
                local_coordinates={1: Coord3D(0, 0, 0)},
            )
            solver = HierarchicalTreeSolver(solver_name="custom_solver")
            solver.solve_tree(tree, options=options)

            mock_solve.assert_called_once()
            _, kwargs = mock_solve.call_args
            assert kwargs["solver_name"] == "custom_solver"
            assert kwargs["options"] is options

    def test_propagates_solver_error(self) -> None:
        """Propagates LayoutSolverError when underlying cluster solver fails."""
        node = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        tree = SolveTree(root=node, graph=None)

        with patch("romutil.layout.milp.tree.solve_cluster") as mock_solve:
            mock_solve.side_effect = LayoutSolverError("Simulated failure")
            solver = HierarchicalTreeSolver()
            with pytest.raises(LayoutSolverError, match="Simulated failure"):
                solver.solve_tree(tree)


class TestWorldRootComponentPacking:
    """Unit and integration tests for sequential X-axis packing of disconnected components under WORLD_ROOT."""

    def test_pack_world_root_empty_children(self) -> None:
        """Constructs an empty node solution when WORLD_ROOT has no children."""
        node = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(),
        )
        sol = pack_world_root_components(node=node, child_solutions={})
        assert sol.node_id == "world_root"
        assert sol.local_coordinates == {}
        assert sol.metadata.get("child_offsets") == {}

    def test_pack_world_root_sequential_x_alignment(self) -> None:
        """Packs multiple child component solutions sequentially along X-axis with padding."""
        ch0 = SolveTreeNode(
            node_id="comp_0",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
        )
        ch1 = SolveTreeNode(
            node_id="comp_1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
        )
        ch2 = SolveTreeNode(
            node_id="comp_2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100}),
        )
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1, ch2),
        )

        sol_0 = NodeSolution(
            node_id="comp_0",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(10, 4, 1)},
            sub_boxes=(BoundingBox3D(0, 10, 0, 4, 0, 1),),
            metadata={
                "bounding_box": BoundingBox3D(0, 10, 0, 4, 0, 1),
                "all_room_coordinates": {1: Coord3D(0, 0, 0), 2: Coord3D(10, 4, 1)},
            },
        )
        sol_1 = NodeSolution(
            node_id="comp_1",
            local_coordinates={10: Coord3D(0, 0, 0), 11: Coord3D(5, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 5, 0, 2, 0, 0),),
            metadata={
                "bounding_box": BoundingBox3D(0, 5, 0, 2, 0, 0),
                "all_room_coordinates": {10: Coord3D(0, 0, 0), 11: Coord3D(5, 2, 0)},
            },
        )
        sol_2 = NodeSolution(
            node_id="comp_2",
            local_coordinates={100: Coord3D(0, 0, 0)},
            metadata={
                "bounding_box": BoundingBox3D(0, 0, 0, 0, 0, 0),
                "all_room_coordinates": {100: Coord3D(0, 0, 0)},
            },
        )

        child_solutions = {
            "comp_0": sol_0,
            "comp_1": sol_1,
            "comp_2": sol_2,
        }

        options = LayoutOptions(component_padding=3)
        sol = pack_world_root_components(node=root, child_solutions=child_solutions, options=options)

        offsets = sol.metadata["child_offsets"]
        assert offsets["comp_0"] == Coord3D(0, 0, 0)
        # comp_0 max_x is 10, so comp_1 starts at 10 + 3 - 0 = 13
        assert offsets["comp_1"] == Coord3D(13, 0, 0)
        # comp_1 max_x is 13 + 5 = 18, so comp_2 starts at 18 + 3 - 0 = 21
        assert offsets["comp_2"] == Coord3D(21, 0, 0)

        # Descendant room coordinates
        all_rooms = sol.metadata["all_room_coordinates"]
        assert all_rooms[1] == Coord3D(0, 0, 0)
        assert all_rooms[2] == Coord3D(10, 4, 1)
        assert all_rooms[10] == Coord3D(13, 0, 0)
        assert all_rooms[11] == Coord3D(18, 2, 0)
        assert all_rooms[100] == Coord3D(21, 0, 0)

        assert sol.metadata["status"] == SolverStatus.OPTIMAL
        bbox = sol.bounding_box
        assert bbox is not None
        assert bbox.min_x == 0
        assert bbox.max_x == 21

    def test_solve_cluster_delegates_world_root_to_packing(self) -> None:
        """solve_cluster bypasses MILP model formulation for WORLD_ROOT nodes."""
        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BLOCK, rooms=frozenset({1}))
        ch1 = SolveTreeNode(node_id="comp_1", node_type=TreeNodeType.BLOCK, rooms=frozenset({2}))
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1),
        )

        sol_0 = NodeSolution(
            node_id="comp_0",
            local_coordinates={1: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 4, 0, 0, 0, 0),),
        )
        sol_1 = NodeSolution(
            node_id="comp_1",
            local_coordinates={2: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 2, 0, 0, 0, 0),),
        )

        sol = solve_cluster(
            node=root,
            child_solutions={"comp_0": sol_0, "comp_1": sol_1},
            options=LayoutOptions(component_padding=2),
        )

        assert sol.node_id == "world_root"
        assert sol.metadata["child_offsets"]["comp_0"] == Coord3D(0, 0, 0)
        assert sol.metadata["child_offsets"]["comp_1"] == Coord3D(6, 0, 0)
        assert sol.metadata["status"] == SolverStatus.OPTIMAL

    def test_hierarchical_tree_solver_world_root_packing(self) -> None:
        """HierarchicalTreeSolver solves child components and packs WORLD_ROOT sequentially."""
        r1 = _make_room(1)
        r2 = _make_room(2)
        graph = LayoutGraph(rooms={1: r1, 2: r2}, edges=())

        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BICONNECTED_BLOCK, rooms=frozenset({1}))
        ch1 = SolveTreeNode(node_id="comp_1", node_type=TreeNodeType.BICONNECTED_BLOCK, rooms=frozenset({2}))
        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1),
        )
        tree = SolveTree(root=world_root, components=(ch0, ch1), graph=graph)

        solver = HierarchicalTreeSolver()
        tree_sol = solver.solve_tree(tree, options=LayoutOptions(component_padding=4))

        assert "world_root" in tree_sol.node_solutions
        root_sol = tree_sol.node_solutions["world_root"]
        assert root_sol.metadata["child_offsets"]["comp_0"] == Coord3D(0, 0, 0)
        assert root_sol.metadata["child_offsets"]["comp_1"] == Coord3D(4, 0, 0)
        assert root_sol.metadata["all_room_coordinates"][1] == Coord3D(0, 0, 0)
        assert root_sol.metadata["all_room_coordinates"][2] == Coord3D(4, 0, 0)

    def test_sewer_disconnected_rooms_compact_placement(self) -> None:
        """Verify sewer layout places disconnected rooms tightly adjacent without large spatial void."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.parser import parse_file
        from romutil.types import WorldGraph

        area_path = Path("tests/fixtures/areas/sewer.are")
        if not area_path.exists():
            pytest.skip("Fixture sewer.are not found")

        sewer_area = parse_file(area_path)
        world_graph = WorldGraph.from_area_data(sewer_area)
        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph)

        assert 7130 in layout.coordinates
        assert 7301 in layout.coordinates

        # Main component rooms
        main_xs = [c.x for v, c in layout.coordinates.items() if v not in (7130, 7301)]
        main_max_x = max(main_xs)

        # Disconnected rooms must be placed within a small constant padding of main component
        assert layout.coordinates[7130].x > main_max_x
        assert layout.coordinates[7301].x > main_max_x
        assert layout.coordinates[7130].x <= main_max_x + 5
        assert layout.coordinates[7301].x <= main_max_x + 5

    def test_pack_world_root_missing_child_solution_and_empty_rooms(self) -> None:
        """Handles child nodes with missing solutions or empty coordinates gracefully."""
        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BLOCK, rooms=frozenset())
        ch1 = SolveTreeNode(node_id="comp_1", node_type=TreeNodeType.BLOCK, rooms=frozenset())
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1),
        )

        # Neither ch0 nor ch1 is provided in child_solutions
        sol = pack_world_root_components(node=root, child_solutions={})
        assert sol.node_id == "world_root"
        assert sol.metadata["child_offsets"]["comp_0"] == Coord3D(0, 0, 0)
        assert sol.metadata["child_offsets"]["comp_1"] == Coord3D(2, 0, 0)
        assert sol.metadata["all_room_coordinates"] == {}
        assert sol.bounding_box is None

    def test_pack_world_root_all_room_coordinates_fallback(self) -> None:
        """Derives child bounding box from metadata all_room_coordinates when sub_boxes is empty."""
        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BLOCK, rooms=frozenset({1}))
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0,),
        )
        sol_0 = NodeSolution(
            node_id="comp_0",
            local_coordinates={},
            sub_boxes=(),
            metadata={"all_room_coordinates": {1: Coord3D(3, 4, 0)}},
        )
        sol = pack_world_root_components(node=root, child_solutions={"comp_0": sol_0})
        assert sol.bounding_box == BoundingBox3D(3, 3, 4, 4, 0, 0)

    def test_midgaard_sewers_composite_disconnected_rooms_placement(self) -> None:
        """Verify Midgaard + Sewers composite layout places rooms 3032, 7130, 7301 adjacent without void."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.models import merge_areas
        from romutil.parser import parse_file
        from romutil.types import LayoutOptions, WorldGraph

        midgaard_path = Path("tests/fixtures/areas/midgaard.are")
        sewer_path = Path("tests/fixtures/areas/sewer.are")
        if not midgaard_path.exists() or not sewer_path.exists():
            pytest.skip("Fixtures midgaard.are and/or sewer.are not found")

        midgaard = parse_file(midgaard_path)
        sewer = parse_file(sewer_path)
        composite = merge_areas([midgaard, sewer])
        world_graph = WorldGraph.from_area_data(composite)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph, LayoutOptions(timeout_seconds=60))

        disconnected = [3032, 7130, 7301]
        for r in disconnected:
            assert r in layout.coordinates

        main_xs = [c.x for v, c in layout.coordinates.items() if v not in disconnected]
        main_max_x = max(main_xs)

        for r in disconnected:
            assert layout.coordinates[r].x > main_max_x
            assert layout.coordinates[r].x <= main_max_x + 10
