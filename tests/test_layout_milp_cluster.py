"""Comprehensive unit and integration tests for MILP cluster solver and HierarchicalTreeSolver."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout import (
    ArticulationPointCut,
    BlockCutTreeBuilder,
    BridgeCut,
    CutType,
    HierarchicalClusterSolver,
    HierarchicalLayoutSolver,
    HierarchicalTreeSolver,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    SpqrTreeBuilder,
    TreeNodeType,
    TreeSolution,
    TreeSolver,
    build_block_spqr_tree,
    solve_cluster,
)
from romutil.layout.milp.formulation import ClusterFormulation
from romutil.layout.milp.inventory import (
    ClusterInventory,
    GatewayCutHandler,
    InterChildBridgeHandler,
    SplitPairHandler,
    derive_child_sub_boxes,
    get_child_boundary_ports,
    inventory_cluster,
)
from romutil.layout.milp.solver import solve_and_extract
from romutil.parser import parse_file
from romutil.layout.geometry import (
    BoundingBox3D,
    Segment3D,
    point_in_segment_interior,
    segment_box_penetration,
    segment_room_penetration,
    segments_intersect_3d,
)
from romutil.models import Direction, Exit, Room
from romutil.types import (
    Coord3D,
    LayoutOptions,
    LayoutSolverError,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


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
    """Unit tests for solve_cluster core mathematical formulation and clearances."""

    def test_empty_parent_and_children(self) -> None:
        """Empty rooms and children returns empty NodeSolution."""
        node = SolveTreeNode(node_id="empty_p", node_type="container", rooms=frozenset())
        graph = LayoutGraph(rooms={}, edges=())
        sol = solve_cluster(node, {}, graph)

        assert sol.node_id == "empty_p"
        assert sol.local_coordinates == {}
        assert sol.bounding_box is None
        assert sol.metadata == {"child_offsets": {}}

    def test_single_room_parent_no_children(self) -> None:
        """Single room parent with no children delegates to leaf solve."""
        node = SolveTreeNode(node_id="single_p", node_type="block", rooms=frozenset({10}))
        graph = _make_graph([10], [])
        sol = solve_cluster(node, {}, graph)

        assert sol.local_coordinates == {10: Coord3D(0, 0, 0)}
        assert sol.bounding_box == BoundingBox3D(0, 0, 0, 0, 0, 0)
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
            (6, 2, Direction.southwest, 1),  # closing south-west link
        ]
        all_edges = parent_edges + child_edges
        graph = _make_graph(all_rooms, all_edges)

        child_node = SolveTreeNode(
            node_id="child_tri",
            node_type="leaf",
            rooms=frozenset(child_rooms),
            metadata={"cut_type": "articulation_point", "anchor_vnum": 2},
        )
        child_sol = solve_cluster(child_node, graph)

        parent_node = SolveTreeNode(
            node_id="parent_sq",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child_node,),
        )
        parent_sol = solve_cluster(parent_node, {child_node.node_id: child_sol}, graph)

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

        # 3. Macro-box clearance: non-anchor parent rooms (1, 3, 4) cannot lie inside child's sub-boxes
        child_sub_boxes = derive_child_sub_boxes(child_node, child_sol, frozenset(get_child_boundary_ports(child_node)))
        assert child_sub_boxes
        for pr in [1, 3, 4]:
            p_pt = parent_sol.local_coordinates[pr]
            for sb in child_sub_boxes:
                sb_in_parent = BoundingBox3D(
                    min_x=c_off.x + sb.min_x,
                    max_x=c_off.x + sb.max_x,
                    min_y=c_off.y + sb.min_y,
                    max_y=c_off.y + sb.max_y,
                    min_z=c_off.z + sb.min_z,
                    max_z=c_off.z + sb.max_z,
                )
                assert not sb_in_parent.contains_point(p_pt)

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

        sol1 = solve_cluster(child1, graph)
        sol2 = solve_cluster(child2, graph)

        parent_node = SolveTreeNode(
            node_id="split_container",
            node_type="split_pair_container",
            rooms=frozenset(poles),
            children=(child1, child2),
            metadata={"cut_type": "split_pair", "poles": (1, 2)},
        )

        parent_sol = solve_cluster(
            parent_node,
            {child1.node_id: sol1, child2.node_id: sol2},
            graph,
        )

        off1 = parent_sol.metadata["child_offsets"][child1.node_id]
        off2 = parent_sol.metadata["child_offsets"][child2.node_id]

        assert sol1.bounding_box is not None and sol2.bounding_box is not None
        b1 = sol1.bounding_box
        b2 = sol2.bounding_box

        box1 = BoundingBox3D(
            min_x=off1.x + b1.min_x,
            max_x=off1.x + b1.max_x,
            min_y=off1.y + b1.min_y,
            max_y=off1.y + b1.max_y,
            min_z=off1.z + b1.min_z,
            max_z=off1.z + b1.max_z,
        )
        box2 = BoundingBox3D(
            min_x=off2.x + b2.min_x,
            max_x=off2.x + b2.max_x,
            min_y=off2.y + b2.min_y,
            max_y=off2.y + b2.max_y,
            min_z=off2.z + b2.min_z,
            max_z=off2.z + b2.max_z,
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

        sol1 = solve_cluster(child1, graph)
        sol2 = solve_cluster(child2, graph)

        parent_node = SolveTreeNode(
            node_id="spqr_p",
            node_type="split_pair_container",
            rooms=frozenset({1, 2}),
            children=(child1, child2),
            metadata={"cut_type": "split_pair", "poles": poles},
        )

        parent_sol = solve_cluster(
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

        sol1 = solve_cluster(child1, graph)
        sol2 = solve_cluster(child2, graph)

        parent_node = SolveTreeNode(
            node_id="parent_node",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child1, child2),
        )

        parent_sol = solve_cluster(
            parent_node,
            {child1.node_id: sol1, child2.node_id: sol2},
            graph,
        )

        off1 = parent_sol.metadata["child_offsets"][child1.node_id]
        off2 = parent_sol.metadata["child_offsets"][child2.node_id]

        assert sol1.bounding_box is not None and sol2.bounding_box is not None
        b1 = sol1.bounding_box
        b2 = sol2.bounding_box

        box1 = BoundingBox3D(
            min_x=off1.x + b1.min_x,
            max_x=off1.x + b1.max_x,
            min_y=off1.y + b1.min_y,
            max_y=off1.y + b1.max_y,
            min_z=off1.z + b1.min_z,
            max_z=off1.z + b1.max_z,
        )
        box2 = BoundingBox3D(
            min_x=off2.x + b2.min_x,
            max_x=off2.x + b2.max_x,
            min_y=off2.y + b2.min_y,
            max_y=off2.y + b2.max_y,
            min_z=off2.z + b2.min_z,
            max_z=off2.z + b2.max_z,
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

        sol1 = solve_cluster(comp1, graph)
        sol2 = solve_cluster(comp2, graph)

        world_root = SolveTreeNode(
            node_id="world_root",
            node_type="world_root",
            rooms=frozenset(),
            children=(comp1, comp2),
        )

        root_sol = solve_cluster(
            world_root,
            {comp1.node_id: sol1, comp2.node_id: sol2},
            graph,
        )

        assert root_sol.local_coordinates == {}
        assert root_sol.bounding_box is not None

        off1 = root_sol.metadata["child_offsets"][comp1.node_id]
        off2 = root_sol.metadata["child_offsets"][comp2.node_id]

        assert sol1.bounding_box is not None and sol2.bounding_box is not None
        b1 = sol1.bounding_box
        b2 = sol2.bounding_box

        box1 = BoundingBox3D(
            min_x=off1.x + b1.min_x,
            max_x=off1.x + b1.max_x,
            min_y=off1.y + b1.min_y,
            max_y=off1.y + b1.max_y,
            min_z=off1.z + b1.min_z,
            max_z=off1.z + b1.max_z,
        )
        box2 = BoundingBox3D(
            min_x=off2.x + b2.min_x,
            max_x=off2.x + b2.max_x,
            min_y=off2.y + b2.min_y,
            max_y=off2.y + b2.max_y,
            min_z=off2.z + b2.min_z,
            max_z=off2.z + b2.max_z,
        )

        assert not box1.intersects(box2)

        # Min corner of overall bounding box must be at (0, 0, 0)
        assert root_sol.bounding_box is not None
        assert (root_sol.bounding_box.min_x, root_sol.bounding_box.min_y, root_sol.bounding_box.min_z) == (0, 0, 0)

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
        child_sol = solve_cluster(child, graph)

        parent_node = SolveTreeNode(
            node_id="p_rev",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child,),
        )
        parent_sol = solve_cluster(parent_node, {child.node_id: child_sol}, graph)

        off = parent_sol.metadata["child_offsets"][child.node_id]
        p10 = parent_sol.local_coordinates[10]
        c2 = child_sol.local_coordinates[2]

        # Room 10 is 1 step East of child room 2: p10.x - (off.x + c2.x) >= 1
        assert p10.x - (off.x + c2.x) >= 1


class TestParentFallbackAndRobustness:
    """Unit tests verifying fail-fast behavior, solver fallback, and error handling."""

    def test_solve_empty_node(self) -> None:
        """Solving completely empty node returns clean empty NodeSolution."""
        node = SolveTreeNode(node_id="empty_test", node_type="container", rooms=frozenset())
        graph = LayoutGraph(rooms={}, edges=())
        sol = solve_cluster(node, {}, graph)
        assert sol.local_coordinates == {}
        assert sol.bounding_box is None

    def test_solve_single_room_no_children(self) -> None:
        """Single room container with no children places room at origin directly."""
        node = SolveTreeNode(node_id="single_test", node_type="leaf", rooms=frozenset({10}))
        graph = LayoutGraph(rooms={10: _make_room(10)}, edges=())
        sol = solve_cluster(node, {}, graph)
        assert sol.local_coordinates == {10: Coord3D(0, 0, 0)}

    def test_solver_unavailable_raises_error(self) -> None:
        """Invalid solver name raises LayoutSolverError."""
        parent_rooms = [1, 2]
        child_rooms = [3, 4]
        edges = [
            (1, 2, Direction.east, 1),
            (2, 3, Direction.north, 1),
            (3, 4, Direction.north, 1),
        ]
        graph = _make_graph([1, 2, 3, 4], edges)
        child = SolveTreeNode(node_id="c_fb", node_type="leaf", rooms=frozenset(child_rooms))
        child_sol = solve_cluster(child, graph)

        parent = SolveTreeNode(
            node_id="p_fb",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child,),
        )
        with pytest.raises(LayoutSolverError):
            solve_cluster(
                parent,
                {child.node_id: child_sol},
                graph,
                solver_name="nonexistent_solver_xyz",
            )

    def test_solver_factory_exception_raises_error(self) -> None:
        """Exception during pyo.SolverFactory raises LayoutSolverError."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_exc", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_exc", local_coordinates={3: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        parent = SolveTreeNode(node_id="p_exc", node_type="block", rooms=frozenset(parent_rooms), children=(child,))

        with patch("pyomo.environ.SolverFactory", side_effect=RuntimeError("Solver init failure")):
            with pytest.raises(LayoutSolverError):
                solve_cluster(parent, {child.node_id: child_sol}, graph)

    def test_solve_cluster_unexpected_exception_wrapping(self) -> None:
        """Unexpected non-LayoutSolverError exceptions in solve_cluster are wrapped in LayoutSolverError."""
        node = SolveTreeNode(node_id="p_unexp", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        with patch("romutil.layout.milp.cluster.formulate_cluster_model", side_effect=RuntimeError("Simulated formulation error")):
            with pytest.raises(LayoutSolverError, match="MILP solver failed for cluster 'p_unexp': Simulated formulation error"):
                solve_cluster(node, {}, graph)

    def test_solver_infeasible_status_raises_error(self) -> None:
        """Infeasible solver status cleanly raises LayoutSolverError."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_inf", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_inf", local_coordinates={3: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        parent = SolveTreeNode(node_id="p_inf", node_type="block", rooms=frozenset(parent_rooms), children=(child,))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.error
        mock_results.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError):
                solve_cluster(parent, {child.node_id: child_sol}, graph)

    def test_solver_none_coordinate_value_raises_error(self) -> None:
        """pyo.value returning None raises LayoutSolverError."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_none", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_none", local_coordinates={3: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        parent = SolveTreeNode(node_id="p_none", node_type="block", rooms=frozenset(parent_rooms), children=(child,))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=None):
                with pytest.raises(LayoutSolverError):
                    solve_cluster(parent, {child.node_id: child_sol}, graph)

    def test_timeout_options_configuration(self) -> None:
        """Options timeout_seconds are passed to CBC and GLPK solvers."""
        parent_rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(parent_rooms, edges)
        child = SolveTreeNode(node_id="c_to", node_type="leaf", rooms=frozenset({3}))
        child_sol = NodeSolution(node_id="c_to", local_coordinates={3: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))
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
                solve_cluster(parent, {child.node_id: child_sol}, graph, options=opts, solver_name="cbc")
                assert mock_cbc.options.get("seconds") == 30

        mock_glpk = MagicMock()
        mock_glpk.available.return_value = True
        mock_glpk.options = {}
        mock_glpk.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_glpk):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0, 2, 0, 0]):
                solve_cluster(parent, {child.node_id: child_sol}, graph, options=opts, solver_name="glpk")
                assert mock_glpk.options.get("tmlim") == 30

    def test_unconnected_overlapping_boxes_separated_by_milp(self) -> None:
        """MILP separates multiple unconnected children whose default placements collide."""
        c1 = SolveTreeNode(node_id="fb_c1", node_type="leaf", rooms=frozenset({1}))
        c2 = SolveTreeNode(node_id="fb_c2", node_type="leaf", rooms=frozenset({2}))
        sol1 = NodeSolution(node_id="fb_c1", local_coordinates={1: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 1),))
        sol2 = NodeSolution(node_id="fb_c2", local_coordinates={2: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 1),))

        parent = SolveTreeNode(node_id="p_fb_boxes", node_type="container", rooms=frozenset(), children=(c1, c2))
        graph = LayoutGraph(rooms={}, edges=())

        sol = solve_cluster(parent, {c1.node_id: sol1, c2.node_id: sol2}, graph)
        off1 = sol.metadata["child_offsets"]["fb_c1"]
        off2 = sol.metadata["child_offsets"]["fb_c2"]

        box1 = (off1.x, off1.x + 1, off1.y, off1.y + 1, off1.z, off1.z + 1)
        box2 = (off2.x, off2.x + 1, off2.y, off2.y + 1, off2.z, off2.z + 1)
        overlap = not (
            box1[1] < box2[0] or box1[0] > box2[1]
            or box1[3] < box2[2] or box1[2] > box2[3]
            or box1[5] < box2[4] or box1[4] > box2[5]
        )
        assert not overlap

    def test_stretched_gateway_corridor_avoids_piercing_third_party_room(self) -> None:
        """A stretched gateway corridor avoids piercing a third-party parent room along its line of sight."""
        parent_rooms = [1, 2]
        child_rooms = [3]
        all_rooms = [1, 2, 3]

        edges = [(1, 3, Direction.east, 4)]
        graph = _make_graph(all_rooms, edges)

        child_node = SolveTreeNode(
            node_id="child_box",
            node_type="leaf",
            rooms=frozenset(child_rooms),
        )
        child_sol = NodeSolution(
            node_id="child_box",
            local_coordinates={3: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        parent_node = SolveTreeNode(
            node_id="parent_cluster",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child_node,),
        )

        sol = solve_cluster(parent_node, {child_node.node_id: child_sol}, graph)

        p1 = sol.local_coordinates[1]
        p2 = sol.local_coordinates[2]
        c_off = sol.metadata["child_offsets"][child_node.node_id]
        p3 = Coord3D(c_off.x, c_off.y, c_off.z)

        gateway_seg = Segment3D(p1, p3)

        assert not point_in_segment_interior(p2, gateway_seg)
        assert not segment_room_penetration(gateway_seg, p2)

    def test_gateway_corridor_avoids_piercing_sibling_child_box(self) -> None:
        """A gateway corridor entering child k avoids piercing sibling child macro-box j."""
        parent_rooms = [1]
        child_k_rooms = [2]
        child_j_rooms = [3, 4]
        all_rooms = [1, 2, 3, 4]

        edges = [
            (1, 2, Direction.east, 4),
            (3, 4, Direction.north, 1),
        ]
        graph = _make_graph(all_rooms, edges)

        child_k = SolveTreeNode(node_id="child_k", node_type="leaf", rooms=frozenset(child_k_rooms))
        child_k_sol = NodeSolution(
            node_id="child_k",
            local_coordinates={2: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        child_j = SolveTreeNode(node_id="child_j", node_type="leaf", rooms=frozenset(child_j_rooms))
        child_j_sol = NodeSolution(
            node_id="child_j",
            local_coordinates={3: Coord3D(0, 0, 0), 4: Coord3D(0, 1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 1, 0, 0),),
        )

        parent_node = SolveTreeNode(
            node_id="parent_siblings",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child_k, child_j),
        )

        sol = solve_cluster(
            parent_node,
            {child_k.node_id: child_k_sol, child_j.node_id: child_j_sol},
            graph,
        )

        p1 = sol.local_coordinates[1]
        off_k = sol.metadata["child_offsets"][child_k.node_id]
        p2 = Coord3D(off_k.x, off_k.y, off_k.z)

        gateway_seg = Segment3D(p1, p2)

        off_j = sol.metadata["child_offsets"][child_j.node_id]
        box_j = BoundingBox3D(
            min_x=off_j.x,
            max_x=off_j.x,
            min_y=off_j.y,
            max_y=off_j.y + 1,
            min_z=off_j.z,
            max_z=off_j.z,
        )

        assert not segment_box_penetration(gateway_seg, box_j)

    def test_crossing_perpendicular_corridors_separated_in_3d(self) -> None:
        """Perpendicular non-incident doorway corridors separate onto distinct Z coordinates or non-overlapping spans."""
        parent_rooms = [1, 3]
        child_1_rooms = [2]
        child_2_rooms = [4]
        all_rooms = [1, 2, 3, 4]

        edges = [
            (1, 2, Direction.east, 4),
            (3, 4, Direction.north, 4),
        ]
        graph = _make_graph(all_rooms, edges)

        child_1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset(child_1_rooms))
        child_1_sol = NodeSolution(
            node_id="c1",
            local_coordinates={2: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        child_2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset(child_2_rooms))
        child_2_sol = NodeSolution(
            node_id="c2",
            local_coordinates={4: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        parent_node = SolveTreeNode(
            node_id="p_cross",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child_1, child_2),
        )

        sol = solve_cluster(
            parent_node,
            {child_1.node_id: child_1_sol, child_2.node_id: child_2_sol},
            graph,
        )

        p1 = sol.local_coordinates[1]
        off1 = sol.metadata["child_offsets"][child_1.node_id]
        p2 = Coord3D(off1.x, off1.y, off1.z)

        p3 = sol.local_coordinates[3]
        off2 = sol.metadata["child_offsets"][child_2.node_id]
        p4 = Coord3D(off2.x, off2.y, off2.z)

        seg1 = Segment3D(p1, p2)
        seg2 = Segment3D(p3, p4)

        assert not segments_intersect_3d(seg1, seg2)

    def test_crossing_doorway_and_internal_corridor_separated_in_3d(self) -> None:
        """Perpendicular doorway corridor and internal parent street separate cleanly."""
        parent_rooms = [1, 3, 4]
        child_rooms = [2]
        all_rooms = [1, 2, 3, 4]

        edges = [
            (3, 4, Direction.north, 4),
            (1, 2, Direction.east, 4),
        ]
        graph = _make_graph(all_rooms, edges)

        child = SolveTreeNode(node_id="c_gw", node_type="leaf", rooms=frozenset(child_rooms))
        child_sol = NodeSolution(
            node_id="c_gw",
            local_coordinates={2: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        parent_node = SolveTreeNode(
            node_id="p_gw_int",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child,),
        )

        sol = solve_cluster(parent_node, {child.node_id: child_sol}, graph)

        p1 = sol.local_coordinates[1]
        off = sol.metadata["child_offsets"][child.node_id]
        p2 = Coord3D(off.x, off.y, off.z)

        p3 = sol.local_coordinates[3]
        p4 = sol.local_coordinates[4]

        gw_seg = Segment3D(p1, p2)
        int_seg = Segment3D(p3, p4)

        assert not segments_intersect_3d(gw_seg, int_seg)

    def test_milp_corridor_clearance_and_sibling_box_avoidance(self) -> None:
        """MILP solver places child boxes without corridor piercing."""
        parent_rooms = [1, 2]
        all_rooms = [1, 2, 3, 4]

        edges = [
            (2, 1, Direction.north, 1),
            (1, 3, Direction.east, 3),
            (2, 4, Direction.east, 3),
        ]
        graph = _make_graph(all_rooms, edges)

        c1 = SolveTreeNode(node_id="fb_c1", node_type="leaf", rooms=frozenset({3}))
        c2 = SolveTreeNode(node_id="fb_c2", node_type="leaf", rooms=frozenset({4}))
        sol1 = NodeSolution(node_id="fb_c1", local_coordinates={3: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))
        sol2 = NodeSolution(node_id="fb_c2", local_coordinates={4: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        parent = SolveTreeNode(node_id="p_fb_corr", node_type="block", rooms=frozenset(parent_rooms), children=(c1, c2))

        sol = solve_cluster(parent, {c1.node_id: sol1, c2.node_id: sol2}, graph)

        p1 = sol.local_coordinates[1]
        p2 = sol.local_coordinates[2]
        off1 = sol.metadata["child_offsets"]["fb_c1"]
        off2 = sol.metadata["child_offsets"]["fb_c2"]

        seg1 = Segment3D(p1, Coord3D(off1.x, off1.y, off1.z))
        seg2 = Segment3D(p2, Coord3D(off2.x, off2.y, off2.z))

        assert not point_in_segment_interior(p2, seg1)
        assert not point_in_segment_interior(p1, seg2)
        assert not segments_intersect_3d(seg1, seg2)

    def test_gateway_corridor_directions_west_south_vertical(self) -> None:
        """Gateway corridors along West, South, and Up/Down directions enforce clearance."""
        for d in [Direction.west, Direction.south, Direction.up, Direction.down]:
            parent_rooms = [1, 2]
            child_rooms = [3]
            all_rooms = [1, 2, 3]
            edges = [(1, 3, d, 4)]
            graph = _make_graph(all_rooms, edges)

            child_node = SolveTreeNode(node_id=f"c_{d.name}", node_type="leaf", rooms=frozenset(child_rooms))
            child_sol = NodeSolution(
                node_id=f"c_{d.name}",
                local_coordinates={3: Coord3D(0, 0, 0)},
                sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
            )

            parent_node = SolveTreeNode(
                node_id=f"p_{d.name}",
                node_type="block",
                rooms=frozenset(parent_rooms),
                children=(child_node,),
            )

            sol = solve_cluster(parent_node, {child_node.node_id: child_sol}, graph)
            p1 = sol.local_coordinates[1]
            p2 = sol.local_coordinates[2]
            c_off = sol.metadata["child_offsets"][child_node.node_id]
            p3 = Coord3D(c_off.x, c_off.y, c_off.z)

            seg = Segment3D(p1, p3)
            assert not point_in_segment_interior(p2, seg)

    def test_gateway_corridor_sibling_box_other_directions(self) -> None:
        """Gateway corridor vs sibling box for West, South, Up, Down directions."""
        for d in [Direction.west, Direction.south, Direction.up, Direction.down]:
            parent_rooms = [1]
            edges = [(1, 2, d, 4)]
            graph = _make_graph([1, 2, 3], edges)

            child_k = SolveTreeNode(node_id="ck", node_type="leaf", rooms=frozenset({2}))
            child_k_sol = NodeSolution(
                node_id="ck",
                local_coordinates={2: Coord3D(0, 0, 0)},
                sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
            )

            child_j = SolveTreeNode(node_id="cj", node_type="leaf", rooms=frozenset({3}))
            child_j_sol = NodeSolution(
                node_id="cj",
                local_coordinates={3: Coord3D(0, 0, 0)},
                sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
            )

            parent_node = SolveTreeNode(
                node_id="p_sib_dir",
                node_type="block",
                rooms=frozenset(parent_rooms),
                children=(child_k, child_j),
            )

            sol = solve_cluster(parent_node, {"ck": child_k_sol, "cj": child_j_sol}, graph)
            p1 = sol.local_coordinates[1]
            off_k = sol.metadata["child_offsets"]["ck"]
            p2 = Coord3D(off_k.x, off_k.y, off_k.z)
            seg = Segment3D(p1, p2)

            off_j = sol.metadata["child_offsets"]["cj"]
            box_j = BoundingBox3D(off_j.x, off_j.x, off_j.y, off_j.y, off_j.z, off_j.z)
            assert not segment_box_penetration(seg, box_j)

    def test_crossing_perpendicular_vertical_corridors(self) -> None:
        """Crossing separation between vertical and horizontal corridors."""
        parent_rooms = [1, 3]
        edges = [
            (1, 2, Direction.east, 4),
            (3, 4, Direction.up, 4),
        ]
        graph = _make_graph([1, 2, 3, 4], edges)

        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({2}))
        c1_sol = NodeSolution(node_id="c1", local_coordinates={2: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({4}))
        c2_sol = NodeSolution(node_id="c2", local_coordinates={4: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        parent = SolveTreeNode(node_id="p_v_cross", node_type="block", rooms=frozenset(parent_rooms), children=(c1, c2))
        sol = solve_cluster(parent, {"c1": c1_sol, "c2": c2_sol}, graph)

        p1 = sol.local_coordinates[1]
        off1 = sol.metadata["child_offsets"]["c1"]
        p2 = Coord3D(off1.x, off1.y, off1.z)

        p3 = sol.local_coordinates[3]
        off2 = sol.metadata["child_offsets"]["c2"]
        p4 = Coord3D(off2.x, off2.y, off2.z)

        seg1 = Segment3D(p1, p2)
        seg2 = Segment3D(p3, p4)
        assert not segments_intersect_3d(seg1, seg2)


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
        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
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
        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)


class TestParentComprehensiveEdgeCoverage:
    """Tests targeting specific solver edge cases and error handling to ensure fail-fast semantics."""

    def test_hierarchical_cluster_solver_delegate(self) -> None:
        """HierarchicalClusterSolver wraps solve_cluster cleanly."""
        solver = HierarchicalClusterSolver(solver_name="cbc")
        node = SolveTreeNode(node_id="pcs_single", node_type="leaf", rooms=frozenset({1}))
        graph = LayoutGraph(rooms={1: _make_room(1)}, edges=())
        sol = solver.solve(node, {}, graph)
        assert sol.local_coordinates == {1: Coord3D(0, 0, 0)}

    def test_milp_solver_unavailable_branch(self) -> None:
        """When solver.available() returns False in solve_cluster, LayoutSolverError is raised."""
        child = SolveTreeNode(node_id="c_unavail", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_unavail", node_type="block", rooms=frozenset({1}), children=(child,))
        graph = _make_graph([1, 2], [])
        c_sol = NodeSolution(node_id="c_unavail", local_coordinates={2: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        mock_solver = MagicMock()
        mock_solver.available.return_value = False
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError):
                solve_cluster(parent, {"c_unavail": c_sol}, graph)

    def test_milp_common_room_anchor_inferred(self) -> None:
        """MILP infers articulation anchor when parent and child share 1 room without anchor_vnum in metadata."""
        child = SolveTreeNode(node_id="c_common", node_type="leaf", rooms=frozenset({2, 3}))
        parent = SolveTreeNode(node_id="p_common", node_type="block", rooms=frozenset({1, 2}), children=(child,))
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.north, weight=1),
        ]
        graph = LayoutGraph(rooms={v: _make_room(v) for v in [1, 2, 3]}, edges=tuple(edges))
        c_sol = NodeSolution(node_id="c_common", local_coordinates={2: Coord3D(0, 0, 0), 3: Coord3D(0, 1, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 1, 0, 0),))
        sol = solve_cluster(parent, {"c_common": c_sol}, graph)
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
        c_sol1 = NodeSolution(node_id="c1", local_coordinates={3: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))
        c_sol2 = NodeSolution(node_id="c2", local_coordinates={4: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))

        parent = SolveTreeNode(
            node_id="p_edges",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child1, child2),
        )
        sol = solve_cluster(parent, {"c1": c_sol1, "c2": c_sol2}, graph)
        assert len(sol.local_coordinates) == 2

    def test_milp_child_missing_solution_and_no_bbox(self) -> None:
        """MILP handles children without solutions or without bounding boxes."""
        c_no_sol = SolveTreeNode(node_id="c_no_sol", node_type="leaf", rooms=frozenset({10}))
        c_no_box = SolveTreeNode(node_id="c_no_box", node_type="leaf", rooms=frozenset())
        sol_no_box = NodeSolution(node_id="c_no_box", local_coordinates={})

        parent = SolveTreeNode(
            node_id="p_no_box",
            node_type="block",
            rooms=frozenset({1}),
            children=(c_no_sol, c_no_box),
        )
        graph = _make_graph([1, 10], [])
        sol = solve_cluster(parent, {"c_no_box": sol_no_box}, graph)
        assert 1 in sol.local_coordinates

    def test_milp_none_value_child_offset_raises_error(self) -> None:
        """pyo.value returning None for child offset raises LayoutSolverError."""
        child = SolveTreeNode(node_id="c_none", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_none_c", node_type="block", rooms=frozenset({1}), children=(child,))
        c_sol = NodeSolution(node_id="c_none", local_coordinates={2: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))
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
                with pytest.raises(LayoutSolverError):
                    solve_cluster(parent, {"c_none": c_sol}, graph)

    def test_milp_solve_exception_raises_error(self) -> None:
        """Exception during solver.solve raises LayoutSolverError."""
        child = SolveTreeNode(node_id="c_exc_solve", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_exc_solve", node_type="block", rooms=frozenset({1}), children=(child,))
        c_sol = NodeSolution(node_id="c_exc_solve", local_coordinates={2: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))
        graph = _make_graph([1, 2], [])

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.solve.side_effect = RuntimeError("Solve crashed")

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError):
                solve_cluster(parent, {"c_exc_solve": c_sol}, graph)

    def test_parent_room_placed_in_concave_child_void(self) -> None:
        """Integration test demonstrating parent room placement inside child concavity.

        When an L-shaped child cluster has empty cells inside its bounding box,
        concave sub-box decomposition allows a parent room to be placed inside the
        concave void without artificial separation conflict or layout dilation.
        """
        # Child cluster: rooms 1, 2, 3 forming an L-shape
        # 1 -E-> 2, 1 -N-> 3
        child_rooms = [1, 2, 3]
        child_edges = [
            (1, 2, Direction.east, 1),
            (1, 3, Direction.north, 1),
        ]
        graph_child = _make_graph(child_rooms, child_edges)
        child_node = SolveTreeNode(
            node_id="child_l",
            node_type="leaf",
            rooms=frozenset(child_rooms),
            metadata={"cut_type": "articulation_point", "anchor_vnum": 2},
        )
        child_sol = solve_cluster(child_node, graph_child)

        # Excluding docking gateway anchor room 2 leaves rooms 1 and 3 in keep-out sub-boxes
        assert child_sol.bounding_box == BoundingBox3D(0, 1, 0, 1, 0, 0)
        child_sub_boxes = derive_child_sub_boxes(child_node, child_sol, frozenset(get_child_boundary_ports(child_node)))
        assert len(child_sub_boxes) == 1

        # Parent cluster: rooms 2 and 4 with edge 2 -N-> 4
        # Room 2 is the gateway anchor. Room 4 is a non-anchor parent room that wants to
        # occupy (1, 1, 0), which lies inside child's bounding box but outside both sub-boxes.
        parent_rooms = [2, 4]
        parent_edges = [
            (2, 4, Direction.north, 1),
        ]
        all_rooms = [1, 2, 3, 4]
        all_edges = child_edges + parent_edges
        graph_parent = _make_graph(all_rooms, all_edges)
        parent_node = SolveTreeNode(
            node_id="parent_p",
            node_type="block",
            rooms=frozenset(parent_rooms),
            children=(child_node,),
        )
        parent_sol = solve_cluster(parent_node, {child_node.node_id: child_sol}, graph_parent)

        # Verify parent solved with optimal status
        assert parent_sol.metadata.get("status") == SolverStatus.OPTIMAL

        c_off = parent_sol.metadata["child_offsets"]["child_l"]
        placed_child_coords = {
            v: Coord3D(c_off.x + pt.x, c_off.y + pt.y, c_off.z + pt.z)
            for v, pt in child_sol.local_coordinates.items()
        }
        p_r4 = parent_sol.local_coordinates[4]

        # 1. Zero collision with any child room
        for cv, cpt in placed_child_coords.items():
            assert p_r4 != cpt

        # 2. Room 4 is inside the overall convex bounding box of the child
        child_bbox = child_sol.bounding_box
        assert child_bbox is not None
        child_full_box = BoundingBox3D(
            min_x=c_off.x + child_bbox.min_x,
            max_x=c_off.x + child_bbox.max_x,
            min_y=c_off.y + child_bbox.min_y,
            max_y=c_off.y + child_bbox.max_y,
            min_z=c_off.z + child_bbox.min_z,
            max_z=c_off.z + child_bbox.max_z,
        )
        assert child_full_box.contains_point(p_r4)

        # 3. Room 4 is outside all decomposed sub-boxes of the child
        for sb in child_sub_boxes:
            sb_in_parent = BoundingBox3D(
                min_x=c_off.x + sb.min_x,
                max_x=c_off.x + sb.max_x,
                min_y=c_off.y + sb.min_y,
                max_y=c_off.y + sb.max_y,
                min_z=c_off.z + sb.min_z,
                max_z=c_off.z + sb.max_z,
            )
            assert not sb_in_parent.contains_point(p_r4)

        # 4. Edge length from 2 to 4 is exactly 1 (no dilation)
        p_r2 = parent_sol.local_coordinates[2]
        dist = abs(p_r4.x - p_r2.x) + abs(p_r4.y - p_r2.y) + abs(p_r4.z - p_r2.z)
        assert dist == 1

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


class TestDirectlyObservableCutAlignment:
    """Tests verifying directly observable split-pair, articulation, and bridge alignments."""

    def test_r_node_parent_aligns_s_node_split_child(self) -> None:
        """R-node parent aligns S-node child at split poles without falling through."""
        # Parent R-node contains rooms 1, 2, 3
        # S-node child attaches at poles (1, 2) and contains rooms 1, 2, 4
        s_child = SolveTreeNode(
            node_id="s_child",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 4}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        r_parent = SolveTreeNode(
            node_id="r_parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2, 3}),
            children=(s_child,),
        )

        child_sol = NodeSolution(
            node_id="s_child",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(0, 2, 0),
                4: Coord3D(0, 1, 0),
            },
            sub_boxes=(BoundingBox3D(0, 0, 0, 2, 0, 0),),
        )

        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 3, Direction.east, 1),
                (3, 2, Direction.north, 2),
            ],
        )

        parent_sol = solve_cluster(r_parent, {"s_child": child_sol}, graph)

        child_offsets = parent_sol.metadata.get("child_offsets", {})
        assert "s_child" in child_offsets
        c_offset = child_offsets["s_child"]

        # Pole 1 in parent must align with child pole 1: model.x[1] == cx + pc1.x
        p1_coord = parent_sol.local_coordinates[1]
        assert p1_coord.x == c_offset.x + 0
        assert p1_coord.y == c_offset.y + 0
        assert p1_coord.z == c_offset.z + 0

    def test_parent_aligns_articulation_point_child_via_contract(self) -> None:
        """Parent container aligns articulation child using typed ArticulationPointCut."""
        art_child = SolveTreeNode(
            node_id="art_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2, 5}),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 2,
                "cut_contract": ArticulationPointCut(anchor_vnum=2),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_art",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
            children=(art_child,),
        )

        child_sol = NodeSolution(
            node_id="art_child",
            local_coordinates={
                2: Coord3D(1, 1, 0),
                5: Coord3D(2, 1, 0),
            },
            sub_boxes=(BoundingBox3D(1, 2, 1, 1, 0, 0),),
        )

        graph = _make_graph(
            rooms=[1, 2, 5],
            edges=[(1, 2, Direction.north, 1)],
        )

        parent_sol = solve_cluster(parent, {"art_child": child_sol}, graph)
        c_offset = parent_sol.metadata["child_offsets"]["art_child"]
        p2_coord = parent_sol.local_coordinates[2]

        assert p2_coord.x == c_offset.x + 1
        assert p2_coord.y == c_offset.y + 1
        assert p2_coord.z == c_offset.z + 0

    def test_parent_aligns_bridge_child_via_contract(self) -> None:
        """Parent container aligns bridge child using typed BridgeCut."""
        bridge_child = SolveTreeNode(
            node_id="br_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
            metadata={
                "cut_type": CutType.BRIDGE,
                "bridge_edge": (2, 10),
                "direction": int(Direction.north),
                "weight": 2,
                "cut_contract": BridgeCut(bridge_edge=(2, 10), direction=int(Direction.north), weight=2),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_br",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
            children=(bridge_child,),
        )

        child_sol = NodeSolution(
            node_id="br_child",
            local_coordinates={
                10: Coord3D(0, 0, 0),
                11: Coord3D(1, 0, 0),
            },
            sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 0),),
        )

        graph = _make_graph(
            rooms=[1, 2, 10, 11],
            edges=[(1, 2, Direction.east, 1)],
        )

        parent_sol = solve_cluster(parent, {"br_child": child_sol}, graph)
        c_offset = parent_sol.metadata["child_offsets"]["br_child"]
        p2_coord = parent_sol.local_coordinates[2]
        c10_world = Coord3D(c_offset.x + 0, c_offset.y + 0, c_offset.z + 0)

        # Bridge constraint: (cx + c10.x) - x[2] == L * dx(0) -> x aligns
        # (cy + c10.y) - y[2] == L * dy(1) >= 2 -> child is north of room 2
        assert c10_world.x == p2_coord.x
        assert c10_world.y >= p2_coord.y + 2

    def test_split_pair_rigid_boundary_ports(self) -> None:
        """Split-pair child maintains rigid boundary port coordinates under parent layout."""
        # Split-pair container with poles 1 and 2
        # Child contains rooms 1, 2, 3 with nominal span 2 along North
        s_child = SolveTreeNode(
            node_id="s_child",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 3}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        child_sol = NodeSolution(
            node_id="s_child",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(0, 2, 0),
                3: Coord3D(0, 1, 0),
            },
            sub_boxes=(BoundingBox3D(0, 0, 0, 2, 0, 0),),
        )

        parent = SolveTreeNode(
            node_id="split_parent",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2, 4, 5}),
            children=(s_child,),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )

        # Parent has intermediate path forcing distance between 1 and 2 to be at least 5
        graph = _make_graph(
            rooms=[1, 2, 3, 4, 5],
            edges=[
                (1, 4, Direction.east, 1),
                (4, 5, Direction.north, 5),
                (5, 2, Direction.west, 1),
            ],
        )

        parent_sol = solve_cluster(parent, {"s_child": child_sol}, graph)
        p1 = parent_sol.local_coordinates[1]
        p2 = parent_sol.local_coordinates[2]
        c_off = parent_sol.metadata["child_offsets"]["s_child"]

        # Pole 1 is pinned at child origin
        assert p1.x == c_off.x + child_sol.local_coordinates[1].x
        assert p1.y == c_off.y + child_sol.local_coordinates[1].y
        assert p1.z == c_off.z + child_sol.local_coordinates[1].z

        # Pole 2 stretches along north axis beyond nominal child span (2 -> >= 5)
        assert p2.x == p1.x
        assert p2.z == p1.z
        assert p2.y >= p1.y + 5

    def test_child_child_bridge_elastic_edge(self) -> None:
        """Child-child bridge edge positions sibling child relative to aligned child."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({20, 21}),
            metadata={
                "cut_type": CutType.BRIDGE,
                "bridge_edge": (11, 20),
                "direction": int(Direction.north),
                "weight": 3,
                "cut_contract": BridgeCut(bridge_edge=(11, 20), direction=int(Direction.north), weight=3),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_two_children",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(),
            children=(c1, c2),
        )

        c1_sol = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 0),),
        )
        c2_sol = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0), 21: Coord3D(0, 1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 1, 0, 0),),
        )

        graph = _make_graph(
            rooms=[10, 11, 20, 21],
            edges=[(10, 11, Direction.east, 1), (20, 21, Direction.north, 1)],
        )

        parent_sol = solve_cluster(parent, {"c1": c1_sol, "c2": c2_sol}, graph)
        off1 = parent_sol.metadata["child_offsets"]["c1"]
        off2 = parent_sol.metadata["child_offsets"]["c2"]

        pos_11 = Coord3D(off1.x + 1, off1.y + 0, off1.z + 0)
        pos_20 = Coord3D(off2.x + 0, off2.y + 0, off2.z + 0)

        # Bridge North with length >= 3
        assert pos_20.x == pos_11.x
        assert pos_20.y >= pos_11.y + 3
        assert pos_20.z == pos_11.z

    def test_split_pair_sibling_alignment_without_parent_rooms(self) -> None:
        """Multiple split-pair children align at shared pole in container without parent rooms."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100, 101}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (100, 101),
                "cut_contract": SplitPairCut(poles=(100, 101)),
            },
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100, 101}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (100, 101),
                "cut_contract": SplitPairCut(poles=(100, 101)),
            },
        )
        parent = SolveTreeNode(
            node_id="split_container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(c1, c2),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (100, 101),
                "cut_contract": SplitPairCut(poles=(100, 101)),
            },
        )
        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={100: Coord3D(0, 0, 0), 101: Coord3D(1, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={100: Coord3D(0, 0, 0), 101: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 2, 0, 0),),
        )
        graph = _make_graph(rooms=[100, 101], edges=[(100, 101, Direction.east, 1)])
        parent_sol = solve_cluster(parent, {"c1": sol1, "c2": sol2}, graph)
        off1 = parent_sol.metadata["child_offsets"]["c1"]
        off2 = parent_sol.metadata["child_offsets"]["c2"]
        # Both children align at shared pole 100
        assert off1.x + sol1.local_coordinates[100].x == off2.x + sol2.local_coordinates[100].x

    def test_split_pair_boundary_port_external_parent_edges(self) -> None:
        """Parent rooms connect to child boundary port when pole is not in parent room set."""
        child = SolveTreeNode(
            node_id="c_split",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({200, 201, 202}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (200, 201),
                "cut_contract": SplitPairCut(poles=(200, 201)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({300, 301}),
            children=(child,),
        )
        child_sol = NodeSolution(
            node_id="c_split",
            local_coordinates={
                200: Coord3D(0, 0, 0),
                201: Coord3D(2, 0, 0),
                202: Coord3D(1, 1, 0),
            },
            sub_boxes=(BoundingBox3D(0, 2, 0, 1, 0, 0),),
        )
        graph = _make_graph(
            rooms=[200, 201, 202, 300, 301],
            edges=[
                (300, 200, Direction.east, 1),
                (201, 301, Direction.east, 1),
            ],
        )
        parent_sol = solve_cluster(parent, {"c_split": child_sol}, graph)
        c_off = parent_sol.metadata["child_offsets"]["c_split"]
        p300 = parent_sol.local_coordinates[300]
        p301 = parent_sol.local_coordinates[301]
        port200 = Coord3D(c_off.x + 0, c_off.y + 0, c_off.z + 0)
        port201 = Coord3D(c_off.x + 2, c_off.y + 0, c_off.z + 0)
        assert port200.x >= p300.x + 1
        assert p301.x >= port201.x + 1

    def test_tree_solver_canonical_export(self) -> None:
        """romutil.layout.milp.tree canonically exports HierarchicalTreeSolver."""
        from romutil.layout.milp.tree import HierarchicalTreeSolver as TreeSolverClass
        assert TreeSolverClass is HierarchicalTreeSolver

    def test_inter_child_split_pair_boundary_port_edges(self) -> None:
        """Inter-child edge connects boundary ports of two split-pair children."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (10, 11),
                "cut_contract": SplitPairCut(poles=(10, 11)),
            },
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({20, 21}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (20, 21),
                "cut_contract": SplitPairCut(poles=(20, 21)),
            },
        )
        parent = SolveTreeNode(
            node_id="p_split_bridge",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(c1, c2),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (10, 21),
                "cut_contract": SplitPairCut(poles=(10, 21)),
            },
        )
        c1_sol = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 0),),
        )
        c2_sol = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0), 21: Coord3D(1, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 0),),
        )
        graph = _make_graph(
            rooms=[10, 11, 20, 21],
            edges=[(11, 20, Direction.east, 2)],
        )
        parent_sol = solve_cluster(parent, {"c1": c1_sol, "c2": c2_sol}, graph)
        off1 = parent_sol.metadata["child_offsets"]["c1"]
        off2 = parent_sol.metadata["child_offsets"]["c2"]
        p11 = Coord3D(off1.x + 1, off1.y, off1.z)
        p20 = Coord3D(off2.x + 0, off2.y, off2.z)
        assert p20.x >= p11.x + 2

    def test_child_bounding_box_strictly_excludes_boundary_ports(self) -> None:
        """Child keep-out sub-boxes strictly exclude boundary ports within parent context."""
        split_child = SolveTreeNode(
            node_id="split_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2, 3}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        graph = _make_graph(
            rooms=[1, 2, 3],
            edges=[
                (1, 3, Direction.east, 1),
                (3, 2, Direction.east, 1),
            ],
        )
        child_sol = solve_cluster(split_child, {}, graph)
        # Pole 1 and Pole 2 are excluded from derived keep-out sub-boxes; only interior room 3 determines keep-out box
        sub_boxes = derive_child_sub_boxes(split_child, child_sol, frozenset({1, 2}))
        assert len(sub_boxes) == 1
        loc3 = child_sol.local_coordinates[3]
        assert sub_boxes[0] == BoundingBox3D(loc3.x, loc3.x, loc3.y, loc3.y, loc3.z, loc3.z)

    def test_child_bounding_box_none_when_all_rooms_are_boundary_ports(self) -> None:
        """Child keep-out sub-boxes are empty when all constituent rooms are boundary gateway ports."""
        pole_only_child = SolveTreeNode(
            node_id="pole_only",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 20}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (10, 20),
                "cut_contract": SplitPairCut(poles=(10, 20)),
            },
        )
        graph = _make_graph(rooms=[10, 20], edges=[(10, 20, Direction.north, 1)])
        sol = solve_cluster(pole_only_child, {}, graph)
        sub_boxes = derive_child_sub_boxes(pole_only_child, sol, frozenset({10, 20}))
        assert sub_boxes == ()

    def test_equated_sibling_children_do_not_add_conflicting_directional_edge(self) -> None:
        """Sibling children equated at a shared pole do not accumulate conflicting directional active edges."""
        # Two children sharing pole 100, where an edge exists between their local ports
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100, 101}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (100, 101),
                "cut_contract": SplitPairCut(poles=(100, 101)),
            },
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100, 102}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (100, 102),
                "cut_contract": SplitPairCut(poles=(100, 102)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({100}),
            children=(c1, c2),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (100, 101),
            },
        )
        c1_sol = NodeSolution(
            node_id="c1",
            local_coordinates={100: Coord3D(0, 0, 0), 101: Coord3D(2, 0, 0)},
            sub_boxes=(BoundingBox3D(2, 2, 0, 0, 0, 0),),
        )
        c2_sol = NodeSolution(
            node_id="c2",
            local_coordinates={100: Coord3D(0, 0, 0), 102: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 2, 2, 0, 0),),
        )
        graph = _make_graph(
            rooms=[100, 101, 102],
            edges=[
                (100, 101, Direction.east, 2),
                (100, 102, Direction.north, 2),
            ],
        )
        sol = solve_cluster(parent, {"c1": c1_sol, "c2": c2_sol}, graph)
        assert sol.metadata["child_offsets"]["c1"] == sol.metadata["child_offsets"]["c2"]


class TestClusterPrimitives:
    """Unit tests for solve_cluster edge cases and core geometric topologies."""

    def test_empty_cluster(self) -> None:
        """Empty room collection returns an empty NodeSolution."""
        node = SolveTreeNode(node_id="empty", node_type="leaf", rooms=frozenset())
        graph = LayoutGraph(rooms={}, edges=())
        sol = solve_cluster(node, {}, graph)

        assert sol.node_id == "empty"
        assert sol.local_coordinates == {}
        assert sol.bounding_box is None

    def test_single_room_cluster(self) -> None:
        """Single room cluster returns origin (0, 0, 0)."""
        node = SolveTreeNode(node_id="single", node_type="leaf", rooms=frozenset({42}))
        graph = _make_graph([42], [])
        sol = solve_cluster(node, {}, graph)

        assert sol.node_id == "single"
        assert sol.local_coordinates == {42: Coord3D(0, 0, 0)}
        assert sol.bounding_box == BoundingBox3D(0, 0, 0, 0, 0, 0)

    def test_solve_cluster_default_arguments(self) -> None:
        """Calling solve_cluster without child_solutions or graph uses sensible empty defaults."""
        node = SolveTreeNode(node_id="single_default", node_type="leaf", rooms=frozenset({10}))
        sol = solve_cluster(node)
        assert sol.local_coordinates == {10: Coord3D(0, 0, 0)}

    def test_solve_cluster_graph_as_second_positional_arg(self) -> None:
        """Calling solve_cluster(node, graph) positionally delegates graph properly."""
        node = SolveTreeNode(node_id="single_pos", node_type="leaf", rooms=frozenset({20}))
        graph = _make_graph([20], [])
        sol = solve_cluster(node, graph)
        assert sol.local_coordinates == {20: Coord3D(0, 0, 0)}

    def test_simple_two_room_corridor(self) -> None:
        """Simple 2-room corridor (East) -> coordinates (0, 0, 0) and (1, 0, 0)."""
        node = SolveTreeNode(node_id="corridor", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1), (2, 1, Direction.west, 1)])
        sol = solve_cluster(node, {}, graph)

        assert sol.local_coordinates[1] == Coord3D(0, 0, 0)
        assert sol.local_coordinates[2] == Coord3D(1, 0, 0)
        assert sol.bounding_box == BoundingBox3D(0, 1, 0, 0, 0, 0)

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
        sol = solve_cluster(node, {}, graph)

        # Expected coordinates normalized:
        # 1: (0, 0, 0), 2: (0, 1, 0), 3: (1, 1, 0), 4: (1, 0, 0)
        coords = sol.local_coordinates
        assert coords[1] == Coord3D(0, 0, 0)
        assert coords[2] == Coord3D(0, 1, 0)
        assert coords[3] == Coord3D(1, 1, 0)
        assert coords[4] == Coord3D(1, 0, 0)
        assert sol.bounding_box == BoundingBox3D(0, 1, 0, 1, 0, 0)

    def test_3d_cube_layer_separation(self) -> None:
        """3D cube or vertical steps (with Up/Down exits) -> vertical layer separation."""
        rooms = [1, 2, 3]
        edges = [
            (1, 2, Direction.up, 1),
            (2, 3, Direction.east, 1),
        ]
        node = SolveTreeNode(node_id="tower", node_type="leaf", rooms=frozenset(rooms))
        graph = _make_graph(rooms, edges)
        sol = solve_cluster(node, {}, graph)

        coords = sol.local_coordinates
        assert coords[1] == Coord3D(0, 0, 0)
        assert coords[2] == Coord3D(0, 0, 1)
        assert coords[3] == Coord3D(1, 0, 1)
        assert sol.bounding_box == BoundingBox3D(0, 1, 0, 0, 0, 1)

    def test_hard_room_non_overlap(self) -> None:
        """Hard non-overlap collision test: rooms cannot overlap at the same (x, y, z)."""
        # Graph with 3 disconnected rooms with no edges between them
        rooms = [10, 20, 30]
        node = SolveTreeNode(node_id="disconnected", node_type="leaf", rooms=frozenset(rooms))
        graph = _make_graph(rooms, [])
        sol = solve_cluster(node, {}, graph)

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
        sol = solve_cluster(node, {}, graph)

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
        """Edges with cut=True are ignored by cluster solver."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        cut_edges = [(2, 1, Direction.north, 1)]
        graph = _make_graph(rooms, edges, cut_edges=cut_edges)
        node = SolveTreeNode(node_id="n1", node_type="leaf", rooms=frozenset(rooms))
        sol = solve_cluster(node, {}, graph)

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
        sol = solve_cluster(node, {}, graph)

        # Rooms are still separated without crashing
        assert len(sol.local_coordinates) == 2
        assert sol.local_coordinates[1] != sol.local_coordinates[2]


class TestClusterFailFast:
    """Unit tests verifying fail-fast behavior and LayoutSolverError on solver failure."""

    def test_solve_empty_and_single_room(self) -> None:
        """Solver handles empty and single-room clusters cleanly."""
        graph = LayoutGraph(rooms={}, edges=())
        empty_node = SolveTreeNode(node_id="e", node_type="leaf", rooms=frozenset())
        sol_empty = solve_cluster(empty_node, {}, graph)
        assert sol_empty.local_coordinates == {}

        single_node = SolveTreeNode(node_id="s", node_type="leaf", rooms=frozenset({100}))
        sol_single = solve_cluster(single_node, {}, graph)
        assert sol_single.local_coordinates == {100: Coord3D(0, 0, 0)}

    def test_solver_unavailable_raises_error(self) -> None:
        """When an invalid solver name is passed, LayoutSolverError is raised."""
        rooms = [1, 2, 3]
        edges = [
            (1, 2, Direction.north, 1),
            (2, 3, Direction.east, 1),
        ]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="fb_test", node_type="leaf", rooms=frozenset(rooms))

        with pytest.raises(LayoutSolverError):
            solve_cluster(node, {}, graph, solver_name="nonexistent_solver_xyz")

    def test_solver_factory_exception_raises_error(self) -> None:
        """Exceptions in pyo.SolverFactory or solver setup raise LayoutSolverError."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="exc_node", node_type="leaf", rooms=frozenset(rooms))

        with patch("pyomo.environ.SolverFactory", side_effect=RuntimeError("Solver init error")):
            with pytest.raises(LayoutSolverError):
                solve_cluster(node, {}, graph)

    def test_solver_infeasible_status_raises_error(self) -> None:
        """Infeasible or errored solver status raises LayoutSolverError."""
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
            with pytest.raises(LayoutSolverError):
                solve_cluster(node, {}, graph)

    def test_solver_none_value_raises_error(self) -> None:
        """If pyo.value returns None for a coordinate, LayoutSolverError is raised."""
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
                with pytest.raises(LayoutSolverError):
                    solve_cluster(node, {}, graph)

    def test_solver_unavailable_flag_raises_error(self) -> None:
        """When solver.available() returns False, LayoutSolverError is raised."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="unavail_node", node_type="leaf", rooms=frozenset(rooms))

        mock_solver = MagicMock()
        mock_solver.available.return_value = False
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError):
                solve_cluster(node, {}, graph)

    def test_solve_model_exception_raises_error(self) -> None:
        """Exceptions raised during model solving raise LayoutSolverError."""
        rooms = [1, 2]
        edges = [(1, 2, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="model_exc_node", node_type="leaf", rooms=frozenset(rooms))

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.solve.side_effect = RuntimeError("Model error during solve")

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError):
                solve_cluster(node, {}, graph)

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
                solve_cluster(node, {}, graph, options=opts, solver_name="cbc")
                assert mock_cbc.options.get("seconds") == 45

        # For GLPK
        mock_glpk = MagicMock()
        mock_glpk.available.return_value = True
        mock_glpk.options = {}
        mock_glpk.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_glpk):
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0]):
                solve_cluster(node, {}, graph, options=opts, solver_name="glpk")
                assert mock_glpk.options.get("tmlim") == 45


class TestHierarchicalTreeSolverAdditionalCases:
    """Additional edge cases for HierarchicalTreeSolver."""

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
        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)
        assert solution.node_solutions["p_empty"].bounding_box is None

    def test_solve_tree_already_solved_node(self) -> None:
        """HierarchicalTreeSolver returns cached solution if a node is encountered again."""
        child = SolveTreeNode(node_id="shared_child", node_type="leaf", rooms=frozenset({1}))
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({1}),
            children=(child, child),
        )
        tree = SolveTree(root=parent, components=(parent,))
        solver = HierarchicalTreeSolver()
        solution = solver.solve_tree(tree)
        assert "shared_child" in solution.node_solutions


class TestSplitPairBoundaryPortAlignment:
    """Unit and integration tests for split-pair dual boundary port alignment and transverse slack."""

    def test_thalos_split_pair_boundary_port_strictly_on_axis(self) -> None:
        """Split-pair boundary port exit 5243 <-> 5241 in Thalos solves strictly on-axis."""
        from pathlib import Path
        from romutil.parser import Parser

        thalos_path = Path("tests/fixtures/areas/thalos.are")
        area = Parser().parse(thalos_path.read_text(encoding="latin-1"))
        world_graph = WorldGraph.from_area_data(area)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph, LayoutOptions(timeout_seconds=30))

        assert layout.diagnostics.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert 5243 in layout.coordinates
        assert 5241 in layout.coordinates

        c5243 = layout.coordinates[5243]
        c5241 = layout.coordinates[5241]

        # Room 5243 connects East to 5241
        assert c5241.x - c5243.x >= 1
        assert c5241.y == c5243.y
        assert c5241.z == c5243.z

        # Connected corridor 5201 -> 5239 -> 5240 -> 5241 is collinear along North
        c5201 = layout.coordinates[5201]
        c5239 = layout.coordinates[5239]
        c5240 = layout.coordinates[5240]

        assert c5201.x == c5239.x == c5240.x == c5241.x
        assert c5201.z == c5239.z == c5240.z == c5241.z
        assert c5201.y < c5239.y < c5240.y < c5241.y

        # Rooms 5209, 5210, 5211 solve strictly on-axis with 5214 along North-South
        assert 5209 in layout.coordinates
        assert 5210 in layout.coordinates
        assert 5211 in layout.coordinates
        assert 5214 in layout.coordinates
        c5209 = layout.coordinates[5209]
        c5210 = layout.coordinates[5210]
        c5211 = layout.coordinates[5211]
        c5214 = layout.coordinates[5214]
        assert c5209.x == c5210.x == c5211.x == c5214.x
        assert c5209.z == c5210.z == c5211.z == c5214.z
        assert c5209.y > c5210.y > c5211.y > c5214.y

        # Entrance corridors 5214 -> 5215 and 5209 -> 5224 solve strictly horizontally with 0 shear
        assert 5215 in layout.coordinates
        assert 5224 in layout.coordinates
        c5215 = layout.coordinates[5215]
        c5224 = layout.coordinates[5224]
        assert c5215.y == c5214.y
        assert c5215.z == c5214.z
        assert c5215.x < c5214.x
        assert c5224.y == c5209.y
        assert c5224.z == c5209.z
        assert c5224.x < c5209.x

        # Market and watchtower rooms assemble cleanly with zero collisions
        market_watchtower_rooms = [5215, 5247, 5217, 5208, 5207, 5220, 5225, 5222, 5224, 5246]
        market_coords = [layout.coordinates[r] for r in market_watchtower_rooms if r in layout.coordinates]
        assert len(market_coords) == len(set(market_coords))

        # Split-pair container rooms 5250 and 5205 solve coplanar at z=2, 1 step west of 5203 and 5204
        assert 5203 in layout.coordinates
        assert 5204 in layout.coordinates
        assert 5205 in layout.coordinates
        assert 5250 in layout.coordinates
        assert layout.coordinates[5203] == Coord3D(2, 8, 2)
        assert layout.coordinates[5204] == Coord3D(2, 9, 2)
        assert layout.coordinates[5250] == Coord3D(1, 8, 2)
        assert layout.coordinates[5205] == Coord3D(1, 9, 2)

    def test_symmetric_split_pair_boundary_active_edges(self) -> None:
        """Both boundary poles of a split-pair child generate symmetric elastic active edges without rigid welds."""
        p0, p1 = 10, 20
        child = SolveTreeNode(
            node_id="c_symm",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 20, 30, 40}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 20}),
            children=(child,),
        )
        c_sol = NodeSolution(
            node_id="c_symm",
            local_coordinates={
                10: Coord3D(0, 0, 0),
                30: Coord3D(2, 0, 0),
                40: Coord3D(2, 4, 0),
                20: Coord3D(0, 4, 0),
            },
            sub_boxes=(BoundingBox3D(0, 2, 0, 4, 0, 0),),
        )
        graph = _make_graph(
            rooms=[10, 20, 30, 40],
            edges=[
                (10, 30, Direction.east, 2),
                (20, 40, Direction.east, 2),
            ],
        )

        handler = SplitPairHandler(node=parent, child_solutions={"c_symm": c_sol}, graph=graph)
        res = handler.process()

        assert len(res.corridor_edges) == 2
        edge_10 = next(e for e in res.corridor_edges if e.src == 10)
        edge_20 = next(e for e in res.corridor_edges if e.src == 20)
        assert edge_10.dst == ("c_symm", 30)
        assert edge_20.dst == ("c_symm", 40)
        assert edge_10.allow_transverse_slack is True
        assert edge_20.allow_transverse_slack is True
        assert edge_10.min_length == 2
        assert edge_20.min_length == 2

        # Symmetrical: neither pole is rigidly welded
        assert res.gateway_anchors["c_symm"] == frozenset({10, 20})
        assert "c_symm" in res.aligned_children

    def test_dual_pole_corridor_stretching_and_transverse_slack(self) -> None:
        """Both split-pair boundary corridors stretch elastically and accommodate transverse slack."""
        p0, p1 = 1, 2
        child = SolveTreeNode(
            node_id="c_elastic",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 11, 12}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="p_elastic",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        child_sol = NodeSolution(
            node_id="c_elastic",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                11: Coord3D(2, 0, 0),
                12: Coord3D(2, 2, 0),
                2: Coord3D(0, 2, 0),
            },
            sub_boxes=(BoundingBox3D(2, 2, 0, 2, 0, 0),),
        )
        graph = _make_graph(
            rooms=[1, 2, 11, 12],
            edges=[
                (1, 2, Direction.north, 4),
                (1, 11, Direction.east, 1),
                (2, 12, Direction.east, 1),
            ],
        )

        parent_sol = solve_cluster(parent, {"c_elastic": child_sol}, graph)
        p1_coord = parent_sol.local_coordinates[1]
        p2_coord = parent_sol.local_coordinates[2]
        c_off = parent_sol.metadata["child_offsets"]["c_elastic"]

        assert p1_coord.x == p2_coord.x
        assert p2_coord.y - p1_coord.y == 4
        # Both corridors connect elastically with transverse slack
        assert "c_elastic" in parent_sol.metadata["child_offsets"]

    def test_split_pair_transverse_boundary_port_alignment(self) -> None:
        """Split-pair child boundary port with transverse exit aligns strictly on-axis."""
        # Pole 1 and Pole 2 in parent. Child contains interior room 4.
        # Inside child: 1 -> 4 is North (len 2), 4 -> 2 is East (len 1).
        # Pole 2 to 4 is transverse to the split pair's primary Y axis.
        s_child = SolveTreeNode(
            node_id="s_child",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 4}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        r_parent = SolveTreeNode(
            node_id="r_parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2, 3}),
            children=(s_child,),
        )

        child_sol = NodeSolution(
            node_id="s_child",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                4: Coord3D(0, 2, 0),
                2: Coord3D(1, 2, 0),
            },
            sub_boxes=(BoundingBox3D(0, 1, 0, 2, 0, 0),),
        )

        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 3, Direction.east, 2),
                (3, 2, Direction.north, 4),
                (1, 4, Direction.north, 2),
                (4, 2, Direction.east, 1),
            ],
        )

        parent_sol = solve_cluster(r_parent, {"s_child": child_sol}, graph)

        child_offsets = parent_sol.metadata.get("child_offsets", {})
        assert "s_child" in child_offsets
        c_offset = child_offsets["s_child"]

        # Room 4 in global parent coordinate system:
        c4_x = c_offset.x + 0
        c4_y = c_offset.y + 2
        c4_z = c_offset.z + 0

        p2_coord = parent_sol.local_coordinates[2]
        # Exit between 4 and 2 is East: c2.x >= c4.x + 1, and strictly on-axis in Y and Z
        assert p2_coord.x - c4_x >= 1
        assert p2_coord.y == c4_y
        assert p2_coord.z == c4_z

    def test_split_pair_anchor_selection_falls_back_to_p0_when_no_transverse_bias(self) -> None:
        """When split-pair poles have symmetric orientation, anchoring falls back to p0."""
        # Both exits are longitudinal along North
        s_child = SolveTreeNode(
            node_id="s_symm",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 4}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        r_parent = SolveTreeNode(
            node_id="r_parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(s_child,),
        )

        child_sol = NodeSolution(
            node_id="s_symm",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                4: Coord3D(0, 1, 0),
                2: Coord3D(0, 2, 0),
            },
            sub_boxes=(BoundingBox3D(0, 0, 0, 2, 0, 0),),
        )

        graph = _make_graph(
            rooms=[1, 2, 4],
            edges=[
                (1, 4, Direction.north, 1),
                (4, 2, Direction.north, 1),
            ],
        )

        parent_sol = solve_cluster(r_parent, {"s_symm": child_sol}, graph)
        child_offsets = parent_sol.metadata.get("child_offsets", {})
        assert "s_symm" in child_offsets
        c_offset = child_offsets["s_symm"]

        # Anchored at pole 1 (p0):
        p1_coord = parent_sol.local_coordinates[1]
        assert p1_coord.x == c_offset.x + 0
        assert p1_coord.y == c_offset.y + 0
        assert p1_coord.z == c_offset.z + 0

    def test_typed_child_handlers_interface_dispatch(self) -> None:
        """Typed child handlers classify interfaces and feed solve_cluster correctly."""
        p0, p1 = 1, 2
        sp_child = SolveTreeNode(
            node_id="sp_child",
            node_type="leaf",
            rooms=frozenset({1, 2, 5}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        gw_child = SolveTreeNode(
            node_id="gw_child",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(1, 10), direction=Direction.south, weight=2)},
        )
        parent = SolveTreeNode(
            node_id="multi_child_p",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(sp_child, gw_child),
        )
        sols = {
            "sp_child": NodeSolution(
                "sp_child",
                {1: Coord3D(0, 0, 0), 2: Coord3D(3, 0, 0), 5: Coord3D(1, 1, 0)},
                (BoundingBox3D(0, 3, 0, 1, 0, 0),),
            ),
            "gw_child": NodeSolution(
                "gw_child",
                {10: Coord3D(0, 0, 0)},
                (BoundingBox3D(0, 0, 0, 0, 0, 0),),
            ),
        }
        graph = _make_graph([1, 2, 5, 10], [(1, 2, Direction.east, 3), (1, 10, Direction.south, 2)])

        inv = ClusterInventory.from_node(parent, sols, graph)
        assert len(inv.children) == 2
        assert any(e.edge_id.startswith("gw_gw_child") for e in inv.active_edges)
        assert any(e.edge_id.startswith("sp_") for e in inv.active_edges)

        sol = solve_cluster(parent, sols, graph)
        assert "sp_child" in sol.metadata["child_offsets"]
        assert "gw_child" in sol.metadata["child_offsets"]

    def test_diagonal_split_pair_solves_to_optimal(self) -> None:
        """A non-collinear split pair with diagonal poles solves to optimal status without coordinate collisions."""
        # 4-room square district 1 <-> 11 <-> 2 <-> 21 <-> 1
        # Connected to outer bypass 1 -> 100 -> 300 -> 301 -> 302 -> 200 -> 2
        r1 = Room(
            vnum=1,
            name="R1",
            description="",
            exits=(
                Exit(direction=Direction.east, dst_vnum=11),
                Exit(direction=Direction.north, dst_vnum=21),
                Exit(direction=Direction.south, dst_vnum=100),
            ),
        )
        r11 = Room(
            vnum=11,
            name="R11",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=1),
                Exit(direction=Direction.north, dst_vnum=2),
            ),
        )
        r2 = Room(
            vnum=2,
            name="R2",
            description="",
            exits=(
                Exit(direction=Direction.south, dst_vnum=11),
                Exit(direction=Direction.west, dst_vnum=21),
                Exit(direction=Direction.east, dst_vnum=200),
            ),
        )
        r21 = Room(
            vnum=21,
            name="R21",
            description="",
            exits=(
                Exit(direction=Direction.east, dst_vnum=2),
                Exit(direction=Direction.south, dst_vnum=1),
            ),
        )
        r100 = Room(
            vnum=100,
            name="R100",
            description="",
            exits=(
                Exit(direction=Direction.north, dst_vnum=1),
                Exit(direction=Direction.east, dst_vnum=300),
            ),
        )
        r300 = Room(
            vnum=300,
            name="R300",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=100),
                Exit(direction=Direction.east, dst_vnum=301),
            ),
        )
        r301 = Room(
            vnum=301,
            name="R301",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=300),
                Exit(direction=Direction.north, dst_vnum=302),
            ),
        )
        r302 = Room(
            vnum=302,
            name="R302",
            description="",
            exits=(
                Exit(direction=Direction.south, dst_vnum=301),
                Exit(direction=Direction.north, dst_vnum=200),
            ),
        )
        r200 = Room(
            vnum=200,
            name="R200",
            description="",
            exits=(
                Exit(direction=Direction.south, dst_vnum=302),
                Exit(direction=Direction.west, dst_vnum=2),
            ),
        )

        rooms = {
            1: r1,
            11: r11,
            2: r2,
            21: r21,
            100: r100,
            300: r300,
            301: r301,
            302: r302,
            200: r200,
        }
        all_exits = tuple(
            exit
            for room in rooms.values()
            for exit in room.exits
        )
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph)

        assert layout.diagnostics.status == SolverStatus.OPTIMAL
        assert len(set(layout.coordinates.values())) == len(layout.coordinates) == 9

        # Relative coordinate assertions for square district
        c1 = layout.coordinates[1]
        c11 = layout.coordinates[11]
        c2 = layout.coordinates[2]
        c21 = layout.coordinates[21]

        assert c11.x - c1.x >= 1
        assert c11.y == c1.y
        assert c2.y - c11.y >= 1
        assert c2.x == c11.x
        assert c21.y - c1.y >= 1
        assert c21.x == c1.x
        assert c2.x - c21.x >= 1
        assert c2.y == c21.y

        # Diagonal pole relationship: delta x >= 1 and delta y >= 1
        assert c2.x - c1.x >= 1
        assert c2.y - c1.y >= 1

        # Relative coordinate assertions for outer bypass
        c100 = layout.coordinates[100]
        c300 = layout.coordinates[300]
        c301 = layout.coordinates[301]
        c302 = layout.coordinates[302]
        c200 = layout.coordinates[200]

        assert c100.y < c1.y
        assert c300.x > c100.x
        assert c301.x > c300.x
        assert c302.y > c301.y
        assert c200.y > c302.y
        assert c200.x > c2.x

    def test_on_axis_stretch_preferred_over_transverse_shear(self) -> None:
        """Solver strictly chooses edge stretch over diagonal transverse shear under quadratic penalty."""
        s_child = SolveTreeNode(
            node_id="s_child",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 4}),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        r_parent = SolveTreeNode(
            node_id="r_parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2, 3}),
            children=(s_child,),
        )
        child_sol = NodeSolution(
            node_id="s_child",
            local_coordinates={1: Coord3D(0, 0, 0), 4: Coord3D(0, 2, 0), 2: Coord3D(1, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 2, 0, 0),),
        )
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 3, Direction.east, 2),
                (3, 2, Direction.north, 4),
                (1, 4, Direction.north, 2),
                (4, 2, Direction.east, 1),
            ],
        )

        inv = inventory_cluster(r_parent, {"s_child": child_sol}, graph)

        # 1. Artificially low transverse penalty: taking 1 unit of transverse shear is cheaper than stretching
        formulation_cheap = ClusterFormulation(inv, w_transverse=5.0)
        formulation_cheap.apply_clearances()
        solve_and_extract(formulation_cheap, options=LayoutOptions())
        sp_cheap = [formulation_cheap.model.s_transverse_p[k].value for k in formulation_cheap.model.s_transverse_p]
        assert any(v > 0 for v in sp_cheap)

        # 2. Quadratic transverse penalty: edge stretching is strictly cheaper than transverse shear
        formulation_quad = ClusterFormulation(inv)
        formulation_quad.apply_clearances()
        solve_and_extract(formulation_quad, options=LayoutOptions())
        sp_quad = [formulation_quad.model.s_transverse_p[k].value for k in formulation_quad.model.s_transverse_p]
        sm_quad = [formulation_quad.model.s_transverse_m[k].value for k in formulation_quad.model.s_transverse_m]
        assert all(v == 0.0 for v in sp_quad)
        assert all(v == 0.0 for v in sm_quad)

    def test_midgaard_park_entrance_and_small_path_on_axis(self) -> None:
        """Park Entrance (3105) and Small Path (3108) in midgaard.are align strictly on-axis with zero transverse shear."""
        area = parse_file(Path("tests/fixtures/areas/midgaard.are"))
        world_graph = WorldGraph.from_area_data(area)
        solver = HierarchicalLayoutSolver.production()
        result = solver.solve(world_graph)
        coords = result.coordinates

        p3105 = coords[3105]
        p3108 = coords[3108]
        # Exit 3105 -> 3108 is South (Direction.south: dx=0, dy=-1, dz=0)
        assert p3105.x == p3108.x
        assert p3108.y < p3105.y
        assert p3105.z == p3108.z


class TestSafe2CutAllowlistEndToEnd:
    """End-to-end regression tests for safe 2-cut allowlist contract and combined child fusion."""

    def test_case_2_rigid_span_mismatch_outer_parent_zero_collisions_zero_tears(self) -> None:
        """Outer parent with rigid span mismatch solves with zero room collisions and zero corridor tears."""
        # Outer parent rigid cluster on rooms 100, 101, 102, 103 with cross diagonal (3-connected)
        # Connecting to poles 1, 2.
        # Branch 1: Rigid K4 pinning span(1, 2) = 1
        # Branch 2: Longer series loop requiring span >= 3
        rooms_data = {
            # Outer Parent R-node
            100: {Direction.north: 101, Direction.west: 103, Direction.northwest: 102, Direction.east: 1},
            101: {Direction.south: 100, Direction.west: 102, Direction.southwest: 103, Direction.east: 2},
            102: {Direction.east: 101, Direction.south: 103, Direction.southeast: 100},
            103: {Direction.north: 102, Direction.east: 100, Direction.northeast: 101},
            # Pole 1
            1: {
                Direction.west: 100,
                Direction.north: 2,
                Direction.east: 3,
                Direction.northeast: 4,
                Direction.south: 20,
            },
            # Pole 2
            2: {
                Direction.west: 101,
                Direction.south: 1,
                Direction.east: 4,
                Direction.southeast: 3,
                Direction.north: 23,
            },
            # Branch 1 internal
            3: {Direction.west: 1, Direction.north: 4, Direction.northwest: 2},
            4: {Direction.south: 3, Direction.west: 2, Direction.southwest: 1},
            # Branch 2 internal
            20: {Direction.north: 1, Direction.east: 21},
            21: {Direction.west: 20, Direction.north: 22},
            22: {Direction.south: 21, Direction.west: 23},
            23: {Direction.east: 22, Direction.south: 2},
        }

        rooms = {
            vnum: Room(
                vnum=vnum,
                name=f"Room {vnum}",
                description="",
                exits=tuple(Exit(direction=d, dst_vnum=dst) for d, dst in exits.items()),
            )
            for vnum, exits in rooms_data.items()
        }
        all_exits = tuple(e for r in rooms.values() for e in r.exits)
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph)

        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(set(layout.coordinates.values())) == len(layout.coordinates) == len(rooms)

        # Verify zero corridor tears for all cardinal exits
        for r in rooms.values():
            c_src = layout.coordinates[r.vnum]
            for e in r.exits:
                c_dst = layout.coordinates[e.dst_vnum]
                dx = c_dst.x - c_src.x
                dy = c_dst.y - c_src.y
                dz = c_dst.z - c_src.z
                if e.direction == Direction.north:
                    assert dx == 0 and dy > 0 and dz == 0
                elif e.direction == Direction.south:
                    assert dx == 0 and dy < 0 and dz == 0
                elif e.direction == Direction.east:
                    assert dx > 0 and dy == 0 and dz == 0
                elif e.direction == Direction.west:
                    assert dx < 0 and dy == 0 and dz == 0

    def test_case_3_concentric_arch_enclosure_outer_parent_zero_collisions_zero_tears(self) -> None:
        """Outer parent with concentric arch enclosure solves with zero room collisions and zero corridor tears."""
        # Outer parent rigid cluster on rooms 100, 101, 102, 103 (3-connected)
        # Connecting to poles 1, 2.
        # Inner branch: 1 -> 3 -> 30 -> 2
        # Outer arch: 1 -> 4 -> 5 -> 6 -> 7 -> 2 wrapping around inner branch
        rooms_data = {
            # Outer Parent R-node
            100: {Direction.north: 101, Direction.west: 103, Direction.northwest: 102, Direction.east: 1},
            101: {Direction.south: 100, Direction.west: 102, Direction.southwest: 103, Direction.east: 2},
            102: {Direction.east: 101, Direction.south: 103, Direction.southeast: 100},
            103: {Direction.north: 102, Direction.east: 100, Direction.northeast: 101},
            # Pole 1
            1: {Direction.west: 100, Direction.north: 2, Direction.east: 3, Direction.south: 4},
            # Pole 2
            2: {Direction.west: 101, Direction.south: 1, Direction.east: 30, Direction.north: 7},
            # Inner branch
            3: {Direction.west: 1, Direction.north: 30},
            30: {Direction.south: 3, Direction.west: 2},
            # Outer arch
            4: {Direction.north: 1, Direction.east: 5},
            5: {Direction.west: 4, Direction.north: 6},
            6: {Direction.south: 5, Direction.west: 7},
            7: {Direction.east: 6, Direction.south: 2},
        }

        rooms = {
            vnum: Room(
                vnum=vnum,
                name=f"Room {vnum}",
                description="",
                exits=tuple(Exit(direction=d, dst_vnum=dst) for d, dst in exits.items()),
            )
            for vnum, exits in rooms_data.items()
        }
        all_exits = tuple(e for r in rooms.values() for e in r.exits)
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph)

        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(set(layout.coordinates.values())) == len(layout.coordinates) == len(rooms)

        # Verify zero corridor tears for all cardinal exits
        for r in rooms.values():
            c_src = layout.coordinates[r.vnum]
            for e in r.exits:
                c_dst = layout.coordinates[e.dst_vnum]
                dx = c_dst.x - c_src.x
                dy = c_dst.y - c_src.y
                dz = c_dst.z - c_src.z
                if e.direction == Direction.north:
                    assert dx == 0 and dy > 0 and dz == 0
                elif e.direction == Direction.south:
                    assert dx == 0 and dy < 0 and dz == 0
                elif e.direction == Direction.east:
                    assert dx > 0 and dy == 0 and dz == 0
                elif e.direction == Direction.west:
                    assert dx < 0 and dy == 0 and dz == 0
