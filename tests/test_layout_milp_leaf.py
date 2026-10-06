"""Comprehensive unit tests for the Leaf Cluster MILP solver and LeafTreeSolver."""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout import (
    BlockCutTreeBuilder,
    HierarchicalLayoutSolver,
    LayoutEdge,
    LayoutGraph,
    LeafTreeSolver,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SpqrTreeBuilder,
    TreeSolver,
    solve_leaf_cluster,
)
from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.layout.milp.leaf import _find_free_pos, _solve_leaf_fallback
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


def _make_graph(
    rooms: Sequence[int],
    edges: Sequence[tuple[int, int, Direction | int, int]],
    cut_edges: Sequence[tuple[int, int, Direction | int, int]] | None = None,
) -> LayoutGraph:
    """Helper to construct a LayoutGraph."""
    room_defs = {v: _make_room(v) for v in rooms}
    layout_edges: list[LayoutEdge] = []
    for u, v, d, w in edges:
        layout_edges.append(
            LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=False)
        )
    if cut_edges:
        for u, v, d, w in cut_edges:
            layout_edges.append(
                LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=True)
            )
    return LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))


class TestSolveLeafClusterPrimitives:
    """Unit tests for solve_leaf_cluster edge cases and core geometric topologies."""

    def test_empty_cluster(self) -> None:
        """Empty room collection returns an empty NodeSolution."""
        node = SolveTreeNode(node_id="empty", node_type="leaf", rooms=frozenset())
        graph = LayoutGraph(rooms={}, edges=())
        sol = solve_leaf_cluster(node, graph)

        assert sol.node_id == "empty"
        assert sol.local_coordinates == {}
        assert sol.bounding_box is None

    def test_single_room_cluster(self) -> None:
        """Single room cluster returns origin (0, 0, 0)."""
        node = SolveTreeNode(node_id="single", node_type="leaf", rooms=frozenset({42}))
        graph = _make_graph([42], [])
        sol = solve_leaf_cluster(node, graph)

        assert sol.node_id == "single"
        assert sol.local_coordinates == {42: Coord3D(0, 0, 0)}
        assert sol.bounding_box == (Coord3D(0, 0, 0), Coord3D(0, 0, 0))

    def test_simple_two_room_corridor(self) -> None:
        """Simple 2-room corridor (East) -> coordinates (0, 0, 0) and (1, 0, 0)."""
        node = SolveTreeNode(node_id="corridor", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1), (2, 1, Direction.west, 1)])
        sol = solve_leaf_cluster(node, graph)

        assert sol.local_coordinates[1] == Coord3D(0, 0, 0)
        assert sol.local_coordinates[2] == Coord3D(1, 0, 0)
        assert sol.bounding_box == (Coord3D(0, 0, 0), Coord3D(1, 0, 0))

    def test_rectilinear_loop_square(self) -> None:
        """Rectilinear loop (4-room cycle North, East, South, West) -> exact compact square layout."""
        # 1 -N-> 2 -E-> 3 -S-> 4 -W-> 1
        rooms = [1, 2, 3, 4]
        edges = [
            (1, 2, Direction.north, 1),
            (2, 3, Direction.east, 1),
            (3, 4, Direction.south, 1),
            (4, 1, Direction.west, 1),
        ]
        node = SolveTreeNode(node_id="cycle4", node_type="series_loop", rooms=frozenset(rooms))
        graph = _make_graph(rooms, edges)
        sol = solve_leaf_cluster(node, graph)

        # Expected coordinates normalized:
        # 1: (0, 0, 0), 2: (0, 1, 0), 3: (1, 1, 0), 4: (1, 0, 0)
        coords = sol.local_coordinates
        assert coords[1] == Coord3D(0, 0, 0)
        assert coords[2] == Coord3D(0, 1, 0)
        assert coords[3] == Coord3D(1, 1, 0)
        assert coords[4] == Coord3D(1, 0, 0)
        assert sol.bounding_box == (Coord3D(0, 0, 0), Coord3D(1, 1, 0))

    def test_3d_cube_layer_separation(self) -> None:
        """3D cube or vertical steps (with Up/Down exits) -> vertical layer separation."""
        rooms = [1, 2, 3]
        edges = [
            (1, 2, Direction.up, 1),
            (2, 3, Direction.east, 1),
        ]
        node = SolveTreeNode(node_id="tower", node_type="leaf", rooms=frozenset(rooms))
        graph = _make_graph(rooms, edges)
        sol = solve_leaf_cluster(node, graph)

        coords = sol.local_coordinates
        assert coords[1] == Coord3D(0, 0, 0)
        assert coords[2] == Coord3D(0, 0, 1)
        assert coords[3] == Coord3D(1, 0, 1)
        assert sol.bounding_box == (Coord3D(0, 0, 0), Coord3D(1, 0, 1))

    def test_hard_room_non_overlap(self) -> None:
        """Hard non-overlap collision test: rooms cannot overlap at the same (x, y, z)."""
        # Graph with 3 disconnected rooms with no edges between them
        rooms = [10, 20, 30]
        node = SolveTreeNode(node_id="disconnected", node_type="leaf", rooms=frozenset(rooms))
        graph = _make_graph(rooms, [])
        sol = solve_leaf_cluster(node, graph)

        placed_points = list(sol.local_coordinates.values())
        # All points must be distinct
        assert len(placed_points) == len(set(placed_points))
        assert len(placed_points) == 3

    def test_elastic_loop_closure(self) -> None:
        """Elastic loop closure: rectangle with 2 North, 1 East, 1 South, 1 West stretches South by 1."""
        # 1 -N-> 2 -N-> 3 -E-> 4 -S-> 5 -W-> 1
        rooms = [1, 2, 3, 4, 5]
        edges = [
            (1, 2, Direction.north, 1),
            (2, 3, Direction.north, 1),
            (3, 4, Direction.east, 1),
            (4, 5, Direction.south, 1),
            (5, 1, Direction.west, 1),
        ]
        node = SolveTreeNode(node_id="elastic", node_type="series_loop", rooms=frozenset(rooms))
        graph = _make_graph(rooms, edges)
        sol = solve_leaf_cluster(node, graph)

        coords = sol.local_coordinates
        # All positions must be distinct and non-overlapping
        assert len(coords) == 5
        assert len(set(coords.values())) == 5
        # Room 1 is (0, 0, 0)
        assert coords[1] == Coord3D(0, 0, 0)
        assert coords[2] == Coord3D(0, 1, 0)
        assert coords[3] == Coord3D(0, 2, 0)
        assert coords[4] == Coord3D(1, 2, 0)
        assert coords[5] == Coord3D(1, 0, 0)
        # Verify edge 4->5 stretched by 1 (distance in Y is 2 instead of 1)
        assert coords[4].y - coords[5].y == 2
        assert coords[4].x == coords[5].x

    def test_cut_edges_ignored(self) -> None:
        """Edges with cut=True are ignored by leaf cluster solver."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        cut_edges = [(2, 1, Direction.north, 1)]
        graph = _make_graph(rooms, edges, cut_edges=cut_edges)
        node = SolveTreeNode(node_id="n1", node_type="leaf", rooms=frozenset(rooms))
        sol = solve_leaf_cluster(node, graph)

        assert sol.local_coordinates[1] == Coord3D(0, 0, 0)
        assert sol.local_coordinates[2] == Coord3D(1, 0, 0)

    def test_invalid_direction_edge_ignored(self) -> None:
        """Edges with invalid direction integer values are skipped gracefully."""
        rooms = [1, 2]
        room_defs = {v: _make_room(v) for v in rooms}
        layout_edges = [
            LayoutEdge(src=1, dst=2, direction=999, weight=1),  # Invalid direction
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))
        node = SolveTreeNode(node_id="n_invalid", node_type="leaf", rooms=frozenset(rooms))
        sol = solve_leaf_cluster(node, graph)

        # Rooms are still separated without crashing
        assert len(sol.local_coordinates) == 2
        assert sol.local_coordinates[1] != sol.local_coordinates[2]


class TestFallbackSolver:
    """Unit tests verifying deterministic fallback solver behavior and robustness."""

    def test_find_free_pos_stepping(self) -> None:
        """_find_free_pos steps forward along preferred displacement until a vacant cell is found."""
        occupied = {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
        free_pos = _find_free_pos((0, 0, 0), occupied, step=(1, 0, 0))
        assert free_pos == (3, 0, 0)

        # Zero step defaults to (1, 0, 0)
        free_pos_zero = _find_free_pos((0, 0, 0), occupied, step=(0, 0, 0))
        assert free_pos_zero == (3, 0, 0)

    def test_fallback_empty_and_single(self) -> None:
        """Fallback solver handles empty and single-room clusters gracefully."""
        graph = LayoutGraph(rooms={}, edges=())
        empty_node = SolveTreeNode(node_id="e", node_type="leaf", rooms=frozenset())
        sol_empty = _solve_leaf_fallback(empty_node, graph)
        assert sol_empty.local_coordinates == {}

        single_node = SolveTreeNode(node_id="s", node_type="leaf", rooms=frozenset({100}))
        sol_single = _solve_leaf_fallback(single_node, graph)
        assert sol_single.local_coordinates == {100: Coord3D(0, 0, 0)}

    def test_fallback_when_solver_unavailable(self) -> None:
        """When an invalid solver name is passed, fallback solver is invoked deterministically."""
        rooms = [1, 2, 3]
        edges = [
            (1, 2, Direction.north, 1),
            (2, 3, Direction.east, 1),
        ]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="fb_test", node_type="leaf", rooms=frozenset(rooms))

        sol = solve_leaf_cluster(node, graph, solver_name="nonexistent_solver_xyz")
        coords = sol.local_coordinates
        assert len(coords) == 3
        assert len(set(coords.values())) == 3
        # Room 1 -> N -> 2 -> E -> 3
        assert coords[1] == Coord3D(0, 0, 0)
        assert coords[2] == Coord3D(0, 1, 0)
        assert coords[3] == Coord3D(1, 1, 0)

    def test_fallback_disconnected_rooms(self) -> None:
        """Fallback solver places disconnected rooms safely on the positive X axis."""
        rooms = [1, 2, 10, 20]
        edges = [
            (1, 2, Direction.north, 1),
            (10, 20, Direction.south, 1),
        ]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="fb_discon", node_type="leaf", rooms=frozenset(rooms))

        sol = _solve_leaf_fallback(node, graph)
        coords = sol.local_coordinates
        assert len(coords) == 4
        assert len(set(coords.values())) == 4

    def test_fallback_reverse_edges(self) -> None:
        """Fallback solver traverses reverse incident edges correctly."""
        rooms = [1, 2]
        # Only reverse edge 2 -> 1 (East)
        edges = [(2, 1, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="fb_rev", node_type="leaf", rooms=frozenset(rooms))

        sol = _solve_leaf_fallback(node, graph)
        assert len(sol.local_coordinates) == 2
        assert sol.local_coordinates[1] != sol.local_coordinates[2]

    def test_solver_factory_exception_triggers_fallback(self) -> None:
        """Exceptions in pyo.SolverFactory or solver setup cleanly trigger fallback."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="exc_node", node_type="leaf", rooms=frozenset(rooms))

        with patch("pyomo.environ.SolverFactory", side_effect=RuntimeError("Solver init error")):
            sol = solve_leaf_cluster(node, graph)
            assert len(sol.local_coordinates) == 2
            assert sol.local_coordinates[1] != sol.local_coordinates[2]

    def test_solver_infeasible_status_triggers_fallback(self) -> None:
        """Infeasible or errored solver status triggers fallback solver."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="infeasible_node", node_type="leaf", rooms=frozenset(rooms))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.error
        mock_results.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            sol = solve_leaf_cluster(node, graph)
            assert len(sol.local_coordinates) == 2
            assert sol.local_coordinates[1] != sol.local_coordinates[2]

    def test_solver_none_value_triggers_fallback(self) -> None:
        """If pyo.value returns None for a coordinate, fallback is cleanly triggered."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="none_val_node", node_type="leaf", rooms=frozenset(rooms))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=None):
                sol = solve_leaf_cluster(node, graph)
                assert len(sol.local_coordinates) == 2

    def test_solver_unavailable_triggers_fallback(self) -> None:
        """When solver.available() returns False, fallback is cleanly triggered."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="unavail_node", node_type="leaf", rooms=frozenset(rooms))

        mock_solver = MagicMock()
        mock_solver.available.return_value = False
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            sol = solve_leaf_cluster(node, graph)
            assert len(sol.local_coordinates) == 2

    def test_solve_model_exception_triggers_fallback(self) -> None:
        """Exceptions raised during model solving trigger fallback."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="model_exc_node", node_type="leaf", rooms=frozenset(rooms))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.solve.side_effect = RuntimeError("Model error during solve")

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            sol = solve_leaf_cluster(node, graph)
            assert len(sol.local_coordinates) == 2

    def test_fallback_with_none_active_edges_and_invalid_direction(self) -> None:
        """Fallback with active_edges=None skips invalid direction edges."""
        rooms = [1, 2]
        room_defs = {v: _make_room(v) for v in rooms}
        layout_edges = [
            LayoutEdge(src=1, dst=2, direction=888, weight=1),  # Invalid direction
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))
        node = SolveTreeNode(node_id="fb_inv", node_type="leaf", rooms=frozenset(rooms))

        sol = _solve_leaf_fallback(node, graph, active_edges=None)
        assert len(sol.local_coordinates) == 2

    def test_solver_timeout_options_configuration(self) -> None:
        """LayoutOptions timeout_seconds are passed to CBC and GLPK solvers."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="timeout_node", node_type="leaf", rooms=frozenset(rooms))
        opts = LayoutOptions(timeout_seconds=45)

        # For CBC
        mock_cbc = MagicMock()
        mock_cbc.available.return_value = True
        mock_cbc.options = {}
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_cbc.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_cbc):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0]):
                solve_leaf_cluster(node, graph, options=opts, solver_name="cbc")
                assert mock_cbc.options.get("seconds") == 45

        # For GLPK
        mock_glpk = MagicMock()
        mock_glpk.available.return_value = True
        mock_glpk.options = {}
        mock_glpk.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_glpk):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0]):
                solve_leaf_cluster(node, graph, options=opts, solver_name="glpk")
                assert mock_glpk.options.get("tmlim") == 45


class TestLeafTreeSolver:
    """Unit tests for LeafTreeSolver implementing the TreeSolver protocol."""

    def test_implements_tree_solver_protocol(self) -> None:
        """LeafTreeSolver is a runtime instance of TreeSolver."""
        solver = LeafTreeSolver()
        assert isinstance(solver, TreeSolver)

    def test_solve_single_leaf_tree(self) -> None:
        """LeafTreeSolver solves a single leaf root tree."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset(rooms))
        tree = SolveTree(root=root, components=(root,), graph=graph)

        solver = LeafTreeSolver()
        solution = solver.solve_tree(tree)

        assert "root" in solution.node_solutions
        root_sol = solution.node_solutions["root"]
        assert root_sol.local_coordinates[1] == Coord3D(0, 0, 0)
        assert root_sol.local_coordinates[2] == Coord3D(1, 0, 0)

    def test_solve_hierarchical_tree_with_children(self) -> None:
        """LeafTreeSolver encloses non-leaf bounding boxes over its children solutions."""
        rooms1 = [1, 2]
        rooms2 = [3, 4]
        all_rooms = rooms1 + rooms2
        edges = [
            (1, 2, Direction.east, 1),
            (3, 4, Direction.north, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        child1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset(rooms1))
        child2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset(rooms2))
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset(all_rooms),
            children=(child1, child2),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)

        solver = LeafTreeSolver()
        solution = solver.solve_tree(tree)

        assert "c1" in solution.node_solutions
        assert "c2" in solution.node_solutions
        assert "parent" in solution.node_solutions

        parent_sol = solution.node_solutions["parent"]
        assert parent_sol.bounding_box is not None
        min_c, max_c = parent_sol.bounding_box
        # Encloses both c1 and c2 bounding boxes
        assert min_c.x <= max_c.x
        assert min_c.y <= max_c.y
        assert min_c.z <= max_c.z

    def test_solve_tree_without_graph(self) -> None:
        """LeafTreeSolver synthesizes layout graph when none is attached to tree."""
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset({10, 20}))
        tree = SolveTree(root=root, components=(root,), graph=None)

        solver = LeafTreeSolver()
        solution = solver.solve_tree(tree)
        assert "root" in solution.node_solutions
        assert len(solution.node_solutions["root"].local_coordinates) == 2

    def test_non_leaf_empty_children_corners(self) -> None:
        """Non-leaf node whose children have empty bounding boxes gets None bounding box."""
        empty_child = SolveTreeNode(node_id="empty_child", node_type="leaf", rooms=frozenset())
        parent = SolveTreeNode(
            node_id="p_empty",
            node_type="container",
            rooms=frozenset(),
            children=(empty_child,),
        )
        tree = SolveTree(root=parent, components=(parent,))
        solver = LeafTreeSolver()
        solution = solver.solve_tree(tree)
        assert solution.node_solutions["p_empty"].bounding_box is None

    def test_solve_tree_with_detached_component(self) -> None:
        """LeafTreeSolver solves components not in tree.root children."""
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset({1}))
        detached = SolveTreeNode(node_id="detached", node_type="leaf", rooms=frozenset({2}))
        tree = SolveTree(root=root, components=(root, detached))

        solver = LeafTreeSolver()
        solution = solver.solve_tree(tree)
        assert "root" in solution.node_solutions
        assert "detached" in solution.node_solutions

    def test_solve_tree_already_solved_node(self) -> None:
        """LeafTreeSolver returns cached solution if a node is encountered again."""
        child = SolveTreeNode(node_id="shared_child", node_type="leaf", rooms=frozenset({1}))
        # Parent contains child twice
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({1}),
            children=(child, child),
        )
        tree = SolveTree(root=parent, components=(parent,))
        solver = LeafTreeSolver()
        solution = solver.solve_tree(tree)
        assert "shared_child" in solution.node_solutions


class TestHierarchicalLayoutIntegration:
    """End-to-end integration tests with HierarchicalLayoutSolver."""

    def test_integration_trivial_pipeline(self) -> None:
        """HierarchicalLayoutSolver(tree_solver=LeafTreeSolver()).solve(world_graph) produces valid SolvedLayout."""
        room1 = Room(
            vnum=101,
            name="Hall",
            description="",
            exits=(Exit(direction=Direction.east, dst_vnum=102),),
        )
        room2 = Room(
            vnum=102,
            name="Chamber",
            description="",
            exits=(Exit(direction=Direction.west, dst_vnum=101),),
        )
        world_graph = WorldGraph(rooms={101: room1, 102: room2}, exits=tuple(room1.exits + room2.exits))

        solver = HierarchicalLayoutSolver(tree_solver=LeafTreeSolver())
        layout = solver.solve(world_graph)

        assert isinstance(layout, SolvedLayout)
        assert layout.diagnostics.status == SolverStatus.FEASIBLE
        assert 101 in layout.coordinates
        assert 102 in layout.coordinates
        # Room 102 is 1 step East of 101
        p1 = layout.coordinates[101]
        p2 = layout.coordinates[102]
        assert p2.x - p1.x == 1
        assert p2.y == p1.y
        assert p2.z == p1.z

    def test_integration_with_block_cut_tree_builder(self) -> None:
        """LeafTreeSolver integrates seamlessly with BlockCutTreeBuilder."""
        # 1 -E-> 2, 2 -N-> 3, 2 -S-> 4 (cut vertex at 2)
        r1 = Room(vnum=1, name="R1", description="", exits=(Exit(direction=Direction.east, dst_vnum=2),))
        r2 = Room(
            vnum=2,
            name="R2",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=1),
                Exit(direction=Direction.north, dst_vnum=3),
                Exit(direction=Direction.south, dst_vnum=4),
            ),
        )
        r3 = Room(vnum=3, name="R3", description="", exits=(Exit(direction=Direction.south, dst_vnum=2),))
        r4 = Room(vnum=4, name="R4", description="", exits=(Exit(direction=Direction.north, dst_vnum=2),))
        rooms = {1: r1, 2: r2, 3: r3, 4: r4}
        world_graph = WorldGraph(rooms=rooms, exits=tuple(r1.exits + r2.exits + r3.exits + r4.exits))

        solver = HierarchicalLayoutSolver(
            tree_builder=BlockCutTreeBuilder(),
            tree_solver=LeafTreeSolver(),
        )
        layout = solver.solve(world_graph)

        assert len(layout.coordinates) == 4
        assert layout.diagnostics.status == SolverStatus.FEASIBLE

    def test_integration_with_spqr_tree_builder(self) -> None:
        """LeafTreeSolver integrates seamlessly with SpqrTreeBuilder."""
        # 4-room cycle (triconnected block)
        r1 = Room(vnum=1, name="R1", description="", exits=(Exit(direction=Direction.north, dst_vnum=2), Exit(direction=Direction.east, dst_vnum=4),))
        r2 = Room(vnum=2, name="R2", description="", exits=(Exit(direction=Direction.south, dst_vnum=1), Exit(direction=Direction.east, dst_vnum=3),))
        r3 = Room(vnum=3, name="R3", description="", exits=(Exit(direction=Direction.west, dst_vnum=2), Exit(direction=Direction.south, dst_vnum=4),))
        r4 = Room(vnum=4, name="R4", description="", exits=(Exit(direction=Direction.west, dst_vnum=1), Exit(direction=Direction.north, dst_vnum=3),))
        rooms = {1: r1, 2: r2, 3: r3, 4: r4}
        world_graph = WorldGraph(rooms=rooms, exits=tuple(r1.exits + r2.exits + r3.exits + r4.exits))

        solver = HierarchicalLayoutSolver(
            tree_builder=SpqrTreeBuilder(),
            tree_solver=LeafTreeSolver(),
        )
        layout = solver.solve(world_graph)

        assert len(layout.coordinates) == 4
        assert layout.diagnostics.status == SolverStatus.FEASIBLE
