"""Comprehensive unit and integration tests for parent container MILP solver and HierarchicalTreeSolver."""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout import (
    BlockCutTreeBuilder,
    HierarchicalLayoutSolver,
    HierarchicalTreeSolver,
    LayoutEdge,
    LayoutGraph,
    LeafTreeSolver,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SpqrTreeBuilder,
    TreeSolution,
    TreeSolver,
    build_block_spqr_tree,
    solve_leaf_cluster,
    solve_parent_cluster,
)
from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.layout.milp.parent import (
    _boxes_intersect,
    _get_child_bbox_corners,
    _point_in_box_bounds,
    _solve_parent_fallback,
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


class TestParentClusterMILP:
    """Unit tests for solve_parent_cluster core mathematical formulation and clearances."""

    def test_empty_parent_and_children(self) -> None:
        """Empty rooms and children returns empty NodeSolution."""
        node = SolveTreeNode(node_id="empty_p", node_type="container", rooms=frozenset())
        graph = LayoutGraph(rooms={}, edges=())
        sol = solve_parent_cluster(node, {}, graph)

        assert sol.node_id == "empty_p"
        assert sol.local_coordinates == {}
        assert sol.bounding_box is None
        assert sol.metadata == {"child_offsets": {}}

    def test_single_room_parent_no_children(self) -> None:
        """Single room parent with no children delegates to leaf solve."""
        node = SolveTreeNode(node_id="single_p", node_type="block", rooms=frozenset({10}))
        graph = _make_graph([10], [])
        sol = solve_parent_cluster(node, {}, graph)

        assert sol.local_coordinates == {10: Coord3D(0, 0, 0)}
        assert sol.bounding_box == (Coord3D(0, 0, 0), Coord3D(0, 0, 0))
        assert sol.metadata == {"child_offsets": {}}

    def test_1cut_gateway_articulation_alignment(self) -> None:
        """1-Cut gateway: parent square with child triangle attached at articulation point.

        Child placed outside parent, gateway aligned, zero collision.
        """
        # Parent square: 1 -E-> 2 -N-> 3 -W-> 4 -S-> 1
        # Articulation anchor: room 2
        # Child triangle: 2 -E-> 5 -N-> 6 -W-> 7 -S-> 2 (or triangle 2 -E-> 5, 5 -N-> 6, 6 -SW-> 2)
        parent_rooms = [1, 2, 3, 4]
        child_rooms = [2, 5, 6]
        all_rooms = [1, 2, 3, 4, 5, 6]

        parent_edges = [
            (1, 2, Direction.east, 1),
            (2, 3, Direction.north, 1),
            (3, 4, Direction.west, 1),
            (4, 1, Direction.south, 1),
        ]
        child_edges = [
            (2, 5, Direction.east, 1),
            (5, 6, Direction.north, 1),
            (6, 2, Direction.south, 1),  # stretched south-west link
        ]
        all_edges = parent_edges + child_edges
        graph = _make_graph(all_rooms, all_edges)

        child_node = SolveTreeNode(
            node_id="child_tri",
            node_type="leaf",
            rooms=frozenset(child_rooms),
            metadata={"cut_type": "articulation_point", "anchor_vnum": 2},
        )
        child_sol = solve_leaf_cluster(child_node, graph)

        parent_node = SolveTreeNode(
            node_id="parent_sq",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child_node,),
        )
        parent_sol = solve_parent_cluster(parent_node, {child_node.node_id: child_sol}, graph)

        # 1. Articulation anchor alignment:
        # In parent frame: parent_coords[2] == child_offset + child_coords[2]
        c_off = parent_sol.metadata["child_offsets"][child_node.node_id]
        c2 = child_sol.local_coordinates[2]
        p2 = parent_sol.local_coordinates[2]
        assert p2.x == c_off.x + c2.x
        assert p2.y == c_off.y + c2.y
        assert p2.z == c_off.z + c2.z

        # 2. Child placed outside parent square:
        # Transform all child rooms into parent frame and verify zero collision with other parent rooms
        placed_child_coords = {
            v: Coord3D(c_off.x + pt.x, c_off.y + pt.y, c_off.z + pt.z)
            for v, pt in child_sol.local_coordinates.items()
        }

        # Parent rooms 1, 3, 4 must not collide with child rooms 5, 6
        for pr in [1, 3, 4]:
            for cr in [5, 6]:
                assert parent_sol.local_coordinates[pr] != placed_child_coords[cr]

        # 3. Macro-box clearance: non-anchor parent rooms (1, 3, 4) cannot lie inside child's bounding box
        assert child_sol.bounding_box is not None
        c_min, c_max = child_sol.bounding_box
        child_box_in_parent = BoundingBox3D(
            min_x=c_off.x + c_min.x,
            max_x=c_off.x + c_max.x,
            min_y=c_off.y + c_min.y,
            max_y=c_off.y + c_max.y,
            min_z=c_off.z + c_min.z,
            max_z=c_off.z + c_max.z,
        )
        for pr in [1, 3, 4]:
            p_pt = parent_sol.local_coordinates[pr]
            assert not child_box_in_parent.contains_point(p_pt)

    def test_2cut_split_pair_side_by_side(self) -> None:
        """2-Cut split pair: parent container with 2 parallel child paths between poles.

        Children placed side-by-side with non-overlapping bounding boxes.
        """
        # Poles 1 and 2 in parent container
        # Child 1: path through room 3 (1 -N-> 3 -N-> 2)
        # Child 2: path through room 4 (1 -N-> 4 -N-> 2)
        # Doorway edges: (1, 3, North), (3, 2, North), (1, 4, East), (4, 2, North)
        poles = [1, 2]
        c1_rooms = [3]
        c2_rooms = [4]
        all_rooms = [1, 2, 3, 4]

        edges = [
            (1, 3, Direction.north, 1),
            (3, 2, Direction.north, 1),
            (1, 4, Direction.east, 1),
            (4, 2, Direction.north, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        child1 = SolveTreeNode(node_id="path_3", node_type="leaf", rooms=frozenset(c1_rooms))
        child2 = SolveTreeNode(node_id="path_4", node_type="leaf", rooms=frozenset(c2_rooms))

        sol1 = solve_leaf_cluster(child1, graph)
        sol2 = solve_leaf_cluster(child2, graph)

        parent_node = SolveTreeNode(
            node_id="split_container",
            node_type="split_pair_container",
            rooms=frozenset(poles),
            children=(child1, child2),
            metadata={"cut_type": "split_pair", "poles": (1, 2)},
        )

        parent_sol = solve_parent_cluster(
            parent_node,
            {child1.node_id: sol1, child2.node_id: sol2},
            graph,
        )

        off1 = parent_sol.metadata["child_offsets"][child1.node_id]
        off2 = parent_sol.metadata["child_offsets"][child2.node_id]

        assert sol1.bounding_box is not None and sol2.bounding_box is not None
        b1_min, b1_max = sol1.bounding_box
        b2_min, b2_max = sol2.bounding_box

        box1 = BoundingBox3D(
            min_x=off1.x + b1_min.x,
            max_x=off1.x + b1_max.x,
            min_y=off1.y + b1_min.y,
            max_y=off1.y + b1_max.y,
            min_z=off1.z + b1_min.z,
            max_z=off1.z + b1_max.z,
        )
        box2 = BoundingBox3D(
            min_x=off2.x + b2_min.x,
            max_x=off2.x + b2_max.x,
            min_y=off2.y + b2_min.y,
            max_y=off2.y + b2_max.y,
            min_z=off2.z + b2_min.z,
            max_z=off2.z + b2_max.z,
        )

        # Must not intersect
        assert not box1.intersects(box2)

        # Poles 1 and 2 in parent coordinates
        p1 = parent_sol.local_coordinates[1]
        p2 = parent_sol.local_coordinates[2]
        assert p1 != p2

    def test_2cut_split_pair_with_shared_poles(self) -> None:
        """2-Cut split pair where child subtrees contain the poles aligns poles precisely."""
        poles = (1, 2)
        c1_rooms = [1, 2, 3]
        c2_rooms = [1, 2, 4]
        all_rooms = [1, 2, 3, 4]

        edges = [
            (1, 3, Direction.north, 1),
            (3, 2, Direction.north, 1),
            (1, 4, Direction.north, 1),
            (4, 2, Direction.north, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        child1 = SolveTreeNode(node_id="spqr_c1", node_type="series_loop", rooms=frozenset(c1_rooms))
        child2 = SolveTreeNode(node_id="spqr_c2", node_type="series_loop", rooms=frozenset(c2_rooms))

        sol1 = solve_leaf_cluster(child1, graph)
        sol2 = solve_leaf_cluster(child2, graph)

        parent_node = SolveTreeNode(
            node_id="spqr_p",
            node_type="split_pair_container",
            rooms=frozenset({1, 2}),
            children=(child1, child2),
            metadata={"cut_type": "split_pair", "poles": poles},
        )

        parent_sol = solve_parent_cluster(
            parent_node,
            {child1.node_id: sol1, child2.node_id: sol2},
            graph,
        )

        off1 = parent_sol.metadata["child_offsets"][child1.node_id]
        off2 = parent_sol.metadata["child_offsets"][child2.node_id]

        p1 = parent_sol.local_coordinates[1]
        p2 = parent_sol.local_coordinates[2]

        # Verify pole 1 aligned with both children
        assert p1.x == off1.x + sol1.local_coordinates[1].x
        assert p1.y == off1.y + sol1.local_coordinates[1].y
        assert p1.z == off1.z + sol1.local_coordinates[1].z

        assert p1.x == off2.x + sol2.local_coordinates[1].x
        assert p1.y == off2.y + sol2.local_coordinates[1].y
        assert p1.z == off2.z + sol2.local_coordinates[1].z

        # Verify pole 2 aligned with both children
        assert p2.x == off1.x + sol1.local_coordinates[2].x
        assert p2.y == off1.y + sol1.local_coordinates[2].y
        assert p2.z == off1.z + sol1.local_coordinates[2].z

        assert p2.x == off2.x + sol2.local_coordinates[2].x
        assert p2.y == off2.y + sol2.local_coordinates[2].y
        assert p2.z == off2.z + sol2.local_coordinates[2].z

    def test_child_child_box_separation(self) -> None:
        """Child-child box separation: parent with 2 child subtrees -> bounding boxes strictly non-overlapping."""
        # Parent room 10
        # Child 1: rooms 1, 2 (East) attached to 10 via doorway (10 -E-> 1)
        # Child 2: rooms 3, 4 (North) attached to 10 via doorway (10 -N-> 3)
        parent_rooms = [10]
        c1_rooms = [1, 2]
        c2_rooms = [3, 4]
        all_rooms = [10, 1, 2, 3, 4]

        edges = [
            (10, 1, Direction.east, 1),
            (1, 2, Direction.east, 1),
            (10, 3, Direction.north, 1),
            (3, 4, Direction.north, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        child1 = SolveTreeNode(node_id="child1", node_type="leaf", rooms=frozenset(c1_rooms))
        child2 = SolveTreeNode(node_id="child2", node_type="leaf", rooms=frozenset(c2_rooms))

        sol1 = solve_leaf_cluster(child1, graph)
        sol2 = solve_leaf_cluster(child2, graph)

        parent_node = SolveTreeNode(
            node_id="parent_node",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child1, child2),
        )

        parent_sol = solve_parent_cluster(
            parent_node,
            {child1.node_id: sol1, child2.node_id: sol2},
            graph,
        )

        off1 = parent_sol.metadata["child_offsets"][child1.node_id]
        off2 = parent_sol.metadata["child_offsets"][child2.node_id]

        assert sol1.bounding_box is not None and sol2.bounding_box is not None
        b1_min, b1_max = sol1.bounding_box
        b2_min, b2_max = sol2.bounding_box

        box1 = BoundingBox3D(
            min_x=off1.x + b1_min.x,
            max_x=off1.x + b1_max.x,
            min_y=off1.y + b1_min.y,
            max_y=off1.y + b1_max.y,
            min_z=off1.z + b1_min.z,
            max_z=off1.z + b1_max.z,
        )
        box2 = BoundingBox3D(
            min_x=off2.x + b2_min.x,
            max_x=off2.x + b2_max.x,
            min_y=off2.y + b2_min.y,
            max_y=off2.y + b2_max.y,
            min_z=off2.z + b2_min.z,
            max_z=off2.z + b2_max.z,
        )

        assert not box1.intersects(box2)

    def test_multi_component_world_root_packing(self) -> None:
        """Multi-component world root (parent with 0 rooms, 2 child components) packed side-by-side without collision."""
        c1_rooms = [1, 2, 3, 4]
        c2_rooms = [5, 6, 7, 8]
        all_rooms = c1_rooms + c2_rooms

        edges = [
            # Comp 1: square
            (1, 2, Direction.east, 1),
            (2, 3, Direction.north, 1),
            (3, 4, Direction.west, 1),
            (4, 1, Direction.south, 1),
            # Comp 2: square
            (5, 6, Direction.east, 1),
            (6, 7, Direction.north, 1),
            (7, 8, Direction.west, 1),
            (8, 5, Direction.south, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        comp1 = SolveTreeNode(node_id="comp1", node_type="leaf", rooms=frozenset(c1_rooms))
        comp2 = SolveTreeNode(node_id="comp2", node_type="leaf", rooms=frozenset(c2_rooms))

        sol1 = solve_leaf_cluster(comp1, graph)
        sol2 = solve_leaf_cluster(comp2, graph)

        world_root = SolveTreeNode(
            node_id="world_root",
            node_type="world_root",
            rooms=frozenset(),
            children=(comp1, comp2),
        )

        root_sol = solve_parent_cluster(
            world_root,
            {comp1.node_id: sol1, comp2.node_id: sol2},
            graph,
        )

        assert root_sol.local_coordinates == {}
        assert root_sol.bounding_box is not None

        off1 = root_sol.metadata["child_offsets"][comp1.node_id]
        off2 = root_sol.metadata["child_offsets"][comp2.node_id]

        assert sol1.bounding_box is not None and sol2.bounding_box is not None
        b1_min, b1_max = sol1.bounding_box
        b2_min, b2_max = sol2.bounding_box

        box1 = BoundingBox3D(
            min_x=off1.x + b1_min.x,
            max_x=off1.x + b1_max.x,
            min_y=off1.y + b1_min.y,
            max_y=off1.y + b1_max.y,
            min_z=off1.z + b1_min.z,
            max_z=off1.z + b1_max.z,
        )
        box2 = BoundingBox3D(
            min_x=off2.x + b2_min.x,
            max_x=off2.x + b2_max.x,
            min_y=off2.y + b2_min.y,
            max_y=off2.y + b2_max.y,
            min_z=off2.z + b2_min.z,
            max_z=off2.z + b2_max.z,
        )

        assert not box1.intersects(box2)

        # Min corner of overall bounding box must be at (0, 0, 0)
        root_min, _ = root_sol.bounding_box
        assert root_min == Coord3D(0, 0, 0)

    def test_reverse_doorway_edge_alignment(self) -> None:
        """1-Cut doorway edge directed from child to parent (reverse) aligns correctly."""
        # Parent room 10
        # Child rooms [1, 2], edge from child 2 to parent 10: 2 -E-> 10
        parent_rooms = [10]
        child_rooms = [1, 2]
        edges = [
            (1, 2, Direction.east, 1),
            (2, 10, Direction.east, 1),  # child -> parent
        ]
        graph = _make_graph([10, 1, 2], edges)

        child = SolveTreeNode(node_id="c_rev", node_type="leaf", rooms=frozenset(child_rooms))
        child_sol = solve_leaf_cluster(child, graph)

        parent_node = SolveTreeNode(
            node_id="p_rev",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child,),
        )
        parent_sol = solve_parent_cluster(parent_node, {child.node_id: child_sol}, graph)

        off = parent_sol.metadata["child_offsets"][child.node_id]
        p10 = parent_sol.local_coordinates[10]
        c2 = child_sol.local_coordinates[2]

        # Room 10 is 1 step East of child room 2: p10.x - (off.x + c2.x) >= 1
        assert p10.x - (off.x + c2.x) >= 1


class TestParentFallbackAndRobustness:
    """Unit tests verifying deterministic fallback solver behavior, geometry helpers, and error handling."""

    def test_geometry_helpers(self) -> None:
        """Verify _boxes_intersect, _point_in_box_bounds, and _get_child_bbox_corners."""
        b1 = (0, 2, 0, 2, 0, 2)
        b2 = (3, 5, 0, 2, 0, 2)
        b3 = (1, 3, 1, 3, 1, 3)

        assert not _boxes_intersect(b1, b2)
        assert _boxes_intersect(b1, b3)

        assert _point_in_box_bounds((1, 1, 1), b1)
        assert not _point_in_box_bounds((3, 1, 1), b1)

        assert _get_child_bbox_corners(None) is None
        bb = BoundingBox3D(min_x=1, max_x=2, min_y=3, max_y=4, min_z=5, max_z=6)
        assert _get_child_bbox_corners(bb) == (1, 2, 3, 4, 5, 6)
        tup_bb = (Coord3D(1, 3, 5), Coord3D(2, 4, 6))
        assert _get_child_bbox_corners(tup_bb) == (1, 2, 3, 4, 5, 6)

    def test_fallback_empty_node(self) -> None:
        """Fallback on completely empty node returns clean empty NodeSolution."""
        node = SolveTreeNode(node_id="fb_empty", node_type="container", rooms=frozenset())
        graph = LayoutGraph(rooms={}, edges=())
        sol = _solve_parent_fallback(node, {}, graph)
        assert sol.local_coordinates == {}
        assert sol.bounding_box is None

    def test_fallback_when_solver_unavailable(self) -> None:
        """Invalid solver name cleanly triggers _solve_parent_fallback."""
        parent_rooms = [1, 2]
        child_rooms = [3, 4]
        edges = [
            (1, 2, Direction.east, 1),
            (2, 3, Direction.north, 1),
            (3, 4, Direction.north, 1),
        ]
        graph = _make_graph([1, 2, 3, 4], edges)
        child = SolveTreeNode(node_id="c_fb", node_type="leaf", rooms=frozenset(child_rooms))
        child_sol = solve_leaf_cluster(child, graph)

        parent = SolveTreeNode(
            node_id="p_fb",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child,),
        )
        sol = solve_parent_cluster(
            parent,
            {child.node_id: child_sol},
            graph,
            solver_name="nonexistent_solver_xyz",
        )
        assert len(sol.local_coordinates) == 2
        assert child.node_id in sol.metadata["child_offsets"]

    def test_fallback_solver_factory_exception(self) -> None:
        """Exception during pyo.SolverFactory triggers fallback cleanly."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_exc", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_exc", local_coordinates={3: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))

        parent = SolveTreeNode(node_id="p_exc", node_type="block", rooms=frozenset(parent_rooms), children=(child,))

        with patch("pyomo.environ.SolverFactory", side_effect=RuntimeError("Solver init failure")):
            sol = solve_parent_cluster(parent, {child.node_id: child_sol}, graph)
            assert len(sol.local_coordinates) == 2

    def test_fallback_infeasible_status(self) -> None:
        """Infeasible solver status cleanly triggers fallback."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_inf", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_inf", local_coordinates={3: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))

        parent = SolveTreeNode(node_id="p_inf", node_type="block", rooms=frozenset(parent_rooms), children=(child,))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.error
        mock_results.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            sol = solve_parent_cluster(parent, {child.node_id: child_sol}, graph)
            assert len(sol.local_coordinates) == 2

    def test_fallback_none_coordinate_value(self) -> None:
        """pyo.value returning None triggers fallback."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_none", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_none", local_coordinates={3: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))

        parent = SolveTreeNode(node_id="p_none", node_type="block", rooms=frozenset(parent_rooms), children=(child,))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=None):
                sol = solve_parent_cluster(parent, {child.node_id: child_sol}, graph)
                assert len(sol.local_coordinates) == 2

    def test_timeout_options_configuration(self) -> None:
        """Options timeout_seconds are passed to CBC and GLPK solvers."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_to", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_to", local_coordinates={3: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        parent = SolveTreeNode(node_id="p_to", node_type="block", rooms=frozenset(parent_rooms), children=(child,))
        opts = LayoutOptions(timeout_seconds=30)

        mock_cbc = MagicMock()
        mock_cbc.available.return_value = True
        mock_cbc.options = {}
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_cbc.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_cbc):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0, 2, 0, 0]):
                solve_parent_cluster(parent, {child.node_id: child_sol}, graph, options=opts, solver_name="cbc")
                assert mock_cbc.options.get("seconds") == 30

        mock_glpk = MagicMock()
        mock_glpk.available.return_value = True
        mock_glpk.options = {}
        mock_glpk.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_glpk):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0, 2, 0, 0]):
                solve_parent_cluster(parent, {child.node_id: child_sol}, graph, options=opts, solver_name="glpk")
                assert mock_glpk.options.get("tmlim") == 30

    def test_fallback_unconnected_overlapping_boxes_step_away(self) -> None:
        """Fallback separates multiple unconnected children whose default placements collide."""
        c1 = SolveTreeNode(node_id="fb_c1", node_type="leaf", rooms=frozenset({1}))
        c2 = SolveTreeNode(node_id="fb_c2", node_type="leaf", rooms=frozenset({2}))
        sol1 = NodeSolution(node_id="fb_c1", local_coordinates={1: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 1, 1)))
        sol2 = NodeSolution(node_id="fb_c2", local_coordinates={2: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 1, 1)))

        parent = SolveTreeNode(node_id="p_fb_boxes", node_type="container", rooms=frozenset(), children=(c1, c2))
        graph = LayoutGraph(rooms={}, edges=())

        sol = _solve_parent_fallback(parent, {c1.node_id: sol1, c2.node_id: sol2}, graph)
        off1 = sol.metadata["child_offsets"]["fb_c1"]
        off2 = sol.metadata["child_offsets"]["fb_c2"]

        box1 = (off1.x, off1.x + 1, off1.y, off1.y + 1, off1.z, off1.z + 1)
        box2 = (off2.x, off2.x + 1, off2.y, off2.y + 1, off2.z, off2.z + 1)
        assert not _boxes_intersect(box1, box2)


class TestHierarchicalTreeSolver:
    """Unit tests for HierarchicalTreeSolver implementing TreeSolver."""

    def test_implements_tree_solver_protocol(self) -> None:
        """HierarchicalTreeSolver is an instance of TreeSolver."""
        solver = HierarchicalTreeSolver()
        assert isinstance(solver, TreeSolver)

    def test_solve_single_leaf_tree(self) -> None:
        """HierarchicalTreeSolver solves a single leaf root tree."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset(rooms))
        tree = SolveTree(root=root, components=(root,), graph=graph)

        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)

        assert "root" in solution.node_solutions
        root_sol = solution.node_solutions["root"]
        assert root_sol.local_coordinates[1] == Coord3D(0, 0, 0)
        assert root_sol.local_coordinates[2] == Coord3D(1, 0, 0)

    def test_solve_nested_hierarchy(self) -> None:
        """HierarchicalTreeSolver bottom-up solves leaves first, then parent container."""
        rooms_leaf = [1, 2]
        rooms_parent = [10]
        all_rooms = rooms_leaf + rooms_parent
        edges = [
            (10, 1, Direction.east, 1),
            (1, 2, Direction.north, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        child = SolveTreeNode(node_id="child", node_type="leaf", rooms=frozenset(rooms_leaf))
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset(rooms_parent),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)

        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)

        assert "child" in solution.node_solutions
        assert "parent" in solution.node_solutions

        p_sol = solution.node_solutions["parent"]
        assert 10 in p_sol.local_coordinates
        assert "child" in p_sol.metadata["child_offsets"]

    def test_solve_tree_without_graph(self) -> None:
        """HierarchicalTreeSolver synthesizes layout graph when none is provided."""
        root = SolveTreeNode(node_id="root_no_graph", node_type="leaf", rooms=frozenset({100}))
        tree = SolveTree(root=root, components=(root,), graph=None)

        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)

        assert "root_no_graph" in solution.node_solutions
        assert solution.node_solutions["root_no_graph"].local_coordinates[100] == Coord3D(0, 0, 0)

    def test_solve_tree_detached_component(self) -> None:
        """HierarchicalTreeSolver solves detached components in tree.components."""
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset({1}))
        detached = SolveTreeNode(node_id="detached", node_type="leaf", rooms=frozenset({2}))
        tree = SolveTree(root=root, components=(root, detached))

        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)

        assert "root" in solution.node_solutions
        assert "detached" in solution.node_solutions


class TestHierarchicalLayoutIntegration:
    """End-to-end integration tests using HierarchicalLayoutSolver with SPQR and BlockCut tree builders."""

    def test_integration_spqr_and_hierarchical_solver(self) -> None:
        """HierarchicalLayoutSolver with SpqrTreeBuilder and HierarchicalTreeSolver solves complex nested hierarchy."""
        # 4-room cycle (triconnected block) with a 5th room attached via articulation point at 2
        # 1 -E-> 2 -N-> 3 -W-> 4 -S-> 1, plus 2 -E-> 5
        r1 = Room(vnum=1, name="R1", description="", exits=(Exit(direction=Direction.east, dst_vnum=2), Exit(direction=Direction.north, dst_vnum=4)))
        r2 = Room(vnum=2, name="R2", description="", exits=(Exit(direction=Direction.west, dst_vnum=1), Exit(direction=Direction.north, dst_vnum=3), Exit(direction=Direction.east, dst_vnum=5)))
        r3 = Room(vnum=3, name="R3", description="", exits=(Exit(direction=Direction.south, dst_vnum=2), Exit(direction=Direction.west, dst_vnum=4)))
        r4 = Room(vnum=4, name="R4", description="", exits=(Exit(direction=Direction.east, dst_vnum=3), Exit(direction=Direction.south, dst_vnum=1)))
        r5 = Room(vnum=5, name="R5", description="", exits=(Exit(direction=Direction.west, dst_vnum=2),))

        rooms = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5}
        all_exits = tuple(r1.exits + r2.exits + r3.exits + r4.exits + r5.exits)
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)

        solver = HierarchicalLayoutSolver(
            tree_builder=SpqrTreeBuilder(),
            tree_solver=HierarchicalTreeSolver(),
        )
        layout = solver.solve(world_graph)

        assert isinstance(layout, SolvedLayout)
        assert layout.diagnostics.status == SolverStatus.FEASIBLE
        assert len(layout.coordinates) == 5

    def test_integration_block_cut_and_hierarchical_solver(self) -> None:
        """HierarchicalLayoutSolver with BlockCutTreeBuilder and HierarchicalTreeSolver solves cut-vertex graph."""
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
            tree_solver=HierarchicalTreeSolver(),
        )
        layout = solver.solve(world_graph)

        assert len(layout.coordinates) == 4
        assert layout.diagnostics.status == SolverStatus.FEASIBLE


class TestParentComprehensiveEdgeCoverage:
    """Tests targeting specific fallback and MILP edge cases to achieve 100% coverage."""

    def test_fallback_invalid_parent_edge_direction(self) -> None:
        """Fallback skips parent edges with invalid directions."""
        parent_rooms = [1, 2]
        edges = [LayoutEdge(src=1, dst=2, direction=999, weight=1)]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in parent_rooms}, edges=tuple(edges))
        node = SolveTreeNode(node_id="fb_inv", node_type="block", rooms=frozenset(parent_rooms))
        sol = _solve_parent_fallback(node, {}, graph)
        assert len(sol.local_coordinates) == 2

    def test_fallback_multiple_disconnected_and_stepping(self) -> None:
        """Fallback handles multiple disconnected parent rooms and occupied cell collisions."""
        rooms = [1, 2, 3, 4]
        # Room 1 and 2 share edge, 3 and 4 share edge, but placed in occupied positions
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=4, direction=Direction.east, weight=1),
        ]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in rooms}, edges=tuple(edges))
        node = SolveTreeNode(node_id="fb_discon", node_type="block", rooms=frozenset(rooms))
        sol = _solve_parent_fallback(node, {}, graph)
        assert len(sol.local_coordinates) == 4

    def test_fallback_child_none_solution(self) -> None:
        """Fallback records (0, 0, 0) offset when child is missing from child_solutions."""
        c = SolveTreeNode(node_id="missing_c", node_type="leaf", rooms=frozenset({1}))
        parent = SolveTreeNode(node_id="p_miss", node_type="container", rooms=frozenset(), children=(c,))
        graph = LayoutGraph(rooms={}, edges=())
        sol = _solve_parent_fallback(parent, {}, graph)
        assert sol.metadata["child_offsets"]["missing_c"] == Coord3D(0, 0, 0)

    def test_fallback_split_pair_alignment(self) -> None:
        """Fallback aligns children under a split pair container with poles."""
        poles = (1, 2)
        c_rooms = [1, 2, 3]
        edges = [LayoutEdge(src=1, dst=2, direction=Direction.north, weight=2)]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in [1, 2, 3]}, edges=tuple(edges))

        child = SolveTreeNode(node_id="c_sp", node_type="leaf", rooms=frozenset(c_rooms))
        c_sol = NodeSolution(
            node_id="c_sp",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0), 3: Coord3D(1, 1, 0)},
            bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 2, 0)),
        )
        parent = SolveTreeNode(
            node_id="p_sp",
            node_type="split_pair_container",
            rooms=frozenset(poles),
            children=(child,),
            metadata={"cut_type": "split_pair", "poles": poles},
        )
        sol = _solve_parent_fallback(parent, {child.node_id: c_sol}, graph)
        assert child.node_id in sol.metadata["child_offsets"]

    def test_fallback_common_room_anchor_and_doorway_directions(self) -> None:
        """Fallback infers anchor from common room and traverses forward/reverse doorway edges."""
        # Common room without anchor_vnum
        parent = SolveTreeNode(
            node_id="p_com",
            node_type="block",
            rooms=frozenset({1, 2}),
            children=(
                SolveTreeNode(node_id="c_com", node_type="leaf", rooms=frozenset({2, 3})),
                SolveTreeNode(node_id="c_fwd", node_type="leaf", rooms=frozenset({4})),
                SolveTreeNode(node_id="c_rev", node_type="leaf", rooms=frozenset({5})),
            ),
        )
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, weight=1),
            LayoutEdge(src=1, dst=4, direction=Direction.east, weight=1),  # Forward doorway
            LayoutEdge(src=5, dst=2, direction=Direction.west, weight=1),  # Reverse doorway
            LayoutEdge(src=1, dst=4, direction=999, weight=1),             # Invalid forward
            LayoutEdge(src=5, dst=2, direction=999, weight=1),             # Invalid reverse
        ]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in range(1, 6)}, edges=tuple(edges))
        sol_com = NodeSolution(node_id="c_com", local_coordinates={2: Coord3D(0, 0, 0), 3: Coord3D(1, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 0, 0)))
        sol_fwd = NodeSolution(node_id="c_fwd", local_coordinates={4: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        sol_rev = NodeSolution(node_id="c_rev", local_coordinates={5: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))

        child_solutions = {
            "c_com": sol_com,
            "c_fwd": sol_fwd,
            "c_rev": sol_rev,
        }
        sol = _solve_parent_fallback(parent, child_solutions, graph)
        assert len(sol.metadata["child_offsets"]) == 3

    def test_fallback_collision_while_loop_step(self) -> None:
        """Fallback steps when placed child macro-boxes overlap."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({1}))
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({2}))
        # Both connected to parent room 10 via same doorway direction East
        parent = SolveTreeNode(
            node_id="p_col",
            node_type="block",
            rooms=frozenset({10}),
            children=(c1, c2),
        )
        edges = [
            LayoutEdge(src=10, dst=1, direction=Direction.east, weight=1),
            LayoutEdge(src=10, dst=2, direction=Direction.east, weight=1),
        ]
        graph = LayoutGraph(rooms={10: _make_room(10), 1: _make_room(1), 2: _make_room(2)}, edges=tuple(edges))
        sol1 = NodeSolution(node_id="c1", local_coordinates={1: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(2, 2, 2)))
        sol2 = NodeSolution(node_id="c2", local_coordinates={2: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(2, 2, 2)))

        sol = _solve_parent_fallback(parent, {"c1": sol1, "c2": sol2}, graph)
        off1 = sol.metadata["child_offsets"]["c1"]
        off2 = sol.metadata["child_offsets"]["c2"]
        box1 = (off1.x, off1.x + 2, off1.y, off1.y + 2, off1.z, off1.z + 2)
        box2 = (off2.x, off2.x + 2, off2.y, off2.y + 2, off2.z, off2.z + 2)
        assert not _boxes_intersect(box1, box2)

    def test_fallback_cand_stepping_in_occupied(self) -> None:
        """Fallback steps cand when candidate cell is already occupied."""
        # 1 -E-> 2, 1 -E-> 3 (where 3 attempts to step into (1, 0, 0) already occupied by 2)
        rooms = [1, 2, 3]
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=1, dst=3, direction=Direction.east, weight=1),
        ]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in rooms}, edges=tuple(edges))
        node = SolveTreeNode(node_id="fb_step", node_type="block", rooms=frozenset(rooms))
        sol = _solve_parent_fallback(node, {}, graph)
        assert len(sol.local_coordinates) == 3
        assert sol.local_coordinates[2] == Coord3D(1, 0, 0)
        assert sol.local_coordinates[3] == Coord3D(2, 0, 0)

    def test_fallback_doorway_cut_and_invalid_directions(self) -> None:
        """Fallback skips cut edges and invalid direction doorway edges."""
        parent = SolveTreeNode(
            node_id="p_dw_test",
            node_type="block",
            rooms=frozenset({10}),
            children=(
                SolveTreeNode(node_id="c_fwd", node_type="leaf", rooms=frozenset({1})),
                SolveTreeNode(node_id="c_rev", node_type="leaf", rooms=frozenset({2})),
            ),
        )
        edges = [
            LayoutEdge(src=10, dst=1, direction=Direction.east, weight=1, cut=True),  # Cut edge
            LayoutEdge(src=10, dst=1, direction=999, weight=1),  # Invalid direction forward
            LayoutEdge(src=10, dst=1, direction=Direction.east, weight=1),  # Valid forward
            LayoutEdge(src=2, dst=10, direction=999, weight=1),  # Invalid direction reverse
            LayoutEdge(src=2, dst=10, direction=Direction.east, weight=1),  # Valid reverse
        ]
        graph = LayoutGraph(rooms={10: _make_room(10), 1: _make_room(1), 2: _make_room(2)}, edges=tuple(edges))
        sol1 = NodeSolution(node_id="c_fwd", local_coordinates={1: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        sol2 = NodeSolution(node_id="c_rev", local_coordinates={2: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        sol = _solve_parent_fallback(parent, {"c_fwd": sol1, "c_rev": sol2}, graph)
        assert len(sol.metadata["child_offsets"]) == 2

    def test_milp_solver_unavailable_branch(self) -> None:
        """When solver.available() returns False in solve_parent_cluster, fallback is invoked."""
        child = SolveTreeNode(node_id="c_unavail", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_unavail", node_type="block", rooms=frozenset({1}), children=(child,))
        graph = _make_graph([1, 2], [])
        c_sol = NodeSolution(node_id="c_unavail", local_coordinates={2: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))

        mock_solver = MagicMock()
        mock_solver.available.return_value = False
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            sol = solve_parent_cluster(parent, {"c_unavail": c_sol}, graph)
            assert 1 in sol.local_coordinates

    def test_milp_common_room_anchor_inferred(self) -> None:
        """MILP infers articulation anchor when parent and child share 1 room without anchor_vnum in metadata."""
        child = SolveTreeNode(node_id="c_common", node_type="leaf", rooms=frozenset({2, 3}))
        parent = SolveTreeNode(node_id="p_common", node_type="block", rooms=frozenset({1, 2}), children=(child,))
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.north, weight=1),
        ]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in [1, 2, 3]}, edges=tuple(edges))
        c_sol = NodeSolution(node_id="c_common", local_coordinates={2: Coord3D(0, 0, 0), 3: Coord3D(0, 1, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 1, 0)))
        sol = solve_parent_cluster(parent, {"c_common": c_sol}, graph)
        assert 1 in sol.local_coordinates
        assert 2 in sol.local_coordinates

    def test_milp_invalid_direction_parent_and_doorway_edges(self) -> None:
        """MILP skips invalid direction edges in internal and doorway loops."""
        parent_rooms = [1, 2]
        edges = [
            LayoutEdge(src=1, dst=2, direction=999, weight=1),  # Invalid internal edge
            LayoutEdge(src=1, dst=3, direction=999, weight=1),  # Invalid doorway edge
            LayoutEdge(src=4, dst=2, direction=999, weight=1),  # Invalid reverse doorway
            LayoutEdge(src=1, dst=3, direction=Direction.east, weight=1, cut=True),  # Cut edge
            LayoutEdge(src=1, dst=3, direction=Direction.east, weight=1, cut=False), # Valid doorway
        ]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in [1, 2, 3, 4]}, edges=tuple(edges))
        child1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({3}))
        child2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({4}))
        c_sol1 = NodeSolution(node_id="c1", local_coordinates={3: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        c_sol2 = NodeSolution(node_id="c2", local_coordinates={4: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))

        parent = SolveTreeNode(
            node_id="p_edges",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child1, child2),
        )
        sol = solve_parent_cluster(parent, {"c1": c_sol1, "c2": c_sol2}, graph)
        assert len(sol.local_coordinates) == 2

    def test_milp_child_missing_solution_and_no_bbox(self) -> None:
        """MILP handles children without solutions or without bounding boxes."""
        c_no_sol = SolveTreeNode(node_id="c_no_sol", node_type="leaf", rooms=frozenset({10}))
        c_no_box = SolveTreeNode(node_id="c_no_box", node_type="leaf", rooms=frozenset())
        sol_no_box = NodeSolution(node_id="c_no_box", local_coordinates={}, bounding_box=None)

        parent = SolveTreeNode(
            node_id="p_no_box",
            node_type="block",
            rooms=frozenset({1}),
            children=(c_no_sol, c_no_box),
        )
        graph = _make_graph([1, 10], [])
        sol = solve_parent_cluster(parent, {"c_no_box": sol_no_box}, graph)
        assert 1 in sol.local_coordinates

    def test_milp_none_value_child_offset_triggers_fallback(self) -> None:
        """pyo.value returning None for child offset triggers fallback."""
        child = SolveTreeNode(node_id="c_none", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_none_c", node_type="block", rooms=frozenset({1}), children=(child,))
        c_sol = NodeSolution(node_id="c_none", local_coordinates={2: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        graph = _make_graph([1, 2], [])

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        # Return numbers for parent x, y, z, then None for child cx
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, None, 0, 0]):
                sol = solve_parent_cluster(parent, {"c_none": c_sol}, graph)
                assert 1 in sol.local_coordinates

    def test_milp_solve_exception_triggers_fallback(self) -> None:
        """Exception during solver.solve triggers fallback."""
        child = SolveTreeNode(node_id="c_exc_solve", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_exc_solve", node_type="block", rooms=frozenset({1}), children=(child,))
        c_sol = NodeSolution(node_id="c_exc_solve", local_coordinates={2: Coord3D(0, 0, 0)}, bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)))
        graph = _make_graph([1, 2], [])

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.solve.side_effect = RuntimeError("Solve crashed")

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            sol = solve_parent_cluster(parent, {"c_exc_solve": c_sol}, graph)
            assert 1 in sol.local_coordinates

    def test_hierarchical_solver_caching_and_nested_room_collection(self) -> None:
        """HierarchicalTreeSolver returns cached solutions and collects rooms across nested children when graph is None."""
        leaf1 = SolveTreeNode(node_id="l1", node_type="leaf", rooms=frozenset({1}))
        leaf2 = SolveTreeNode(node_id="l2", node_type="leaf", rooms=frozenset({2}))
        # Parent contains leaf1 twice to test cache branch
        parent = SolveTreeNode(node_id="p_nested", node_type="block", rooms=frozenset({3}), children=(leaf1, leaf1, leaf2))
        tree = SolveTree(root=parent, components=(parent,), graph=None)

        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)
        assert "l1" in solution.node_solutions
        assert "l2" in solution.node_solutions
        assert "p_nested" in solution.node_solutions
