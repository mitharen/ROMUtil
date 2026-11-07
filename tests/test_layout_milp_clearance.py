"""Unit tests for child-child geometric clearance and separation across shared boundary ports."""

from __future__ import annotations

import dataclasses

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    ArticulationPointCut,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTreeNode,
    SplitPairCut,
)
from romutil.layout.milp import (
    ClusterFormulation,
    detect_cluster_collisions,
    formulate_cluster_model,
    inventory_cluster,
)
from romutil.layout.milp.clearance import (
    ClearanceConstraintBuilder,
    ClearanceViolation,
    VIOLATION_TIER_PRIORITY,
)
from romutil.layout.milp.inventory import ActiveEdge
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.models import Direction, Room
from romutil.types import (
    Coord3D,
    LayoutOptions,
    SolverStatus,
)
from unittest.mock import MagicMock, patch

import pytest


def _make_graph(
    rooms: list[int],
    edges: list[tuple[int, int, Direction | int, int]],
    cut_edges: list[tuple[int, int, Direction | int, int]] | None = None,
) -> LayoutGraph:
    """Construct a LayoutGraph for testing."""
    room_defs = {
        v: Room(vnum=v, name=f"Room {v}", description="", exits=())
        for v in rooms
    }
    layout_edges: list[LayoutEdge] = [
        LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=False)
        for u, v, d, w in edges
    ]
    if cut_edges:
        for u, v, d, w in cut_edges:
            layout_edges.append(
                LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=True)
            )
    return LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))


class TestChildChildClearance:
    """Tests for pairwise child-child sub-box clearance across shared boundary ports and poles."""

    def test_child_child_separation_shared_boundary_port(self) -> None:
        """Child clusters sharing a boundary port have their interior sub-boxes separated."""
        # Child 1: Split pair cluster possessing interior room 10 and boundary port 1
        c1 = SolveTreeNode(
            node_id="c_spqr",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        # Child 2: Articulation cluster possessing interior room 20 and anchored at boundary port 1
        c2 = SolveTreeNode(
            node_id="c_artic",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c_spqr",
            local_coordinates={10: Coord3D(0, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 1, 1, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c_artic",
            local_coordinates={20: Coord3D(1, 0, 0)},
            port_coordinates={1: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(1, 1, 0, 0, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_spqr": sol1, "c_artic": sol2},
            graph=graph,
        )
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_child_child_separation()

        # Both sub-boxes must be registered for disjunctive separation
        separated = formulation.clearance_builder.separated_pairs
        assert ("child_child", ("c_artic", 0), ("c_spqr", 0)) in separated or (
            "child_child",
            ("c_spqr", 0),
            ("c_artic", 0),
        ) in separated
        assert hasattr(formulation.model, "child_child_br")

    def test_child_child_separation_shared_split_pair_poles(self) -> None:
        """Parallel split-pair child clusters sharing poles have their interior sub-boxes separated."""
        c1 = SolveTreeNode(
            node_id="c_branch_a",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        c2 = SolveTreeNode(
            node_id="c_branch_b",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        parent = SolveTreeNode(
            node_id="p_split",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c_branch_a",
            local_coordinates={10: Coord3D(0, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 1, 1, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c_branch_b",
            local_coordinates={20: Coord3D(1, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(1, 1, 1, 1, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_branch_a": sol1, "c_branch_b": sol2},
            graph=graph,
        )
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_child_child_separation()

        separated = formulation.clearance_builder.separated_pairs
        assert ("child_child", ("c_branch_a", 0), ("c_branch_b", 0)) in separated or (
            "child_child",
            ("c_branch_b", 0),
            ("c_branch_a", 0),
        ) in separated
        assert hasattr(formulation.model, "child_child_br")

    def test_detect_cluster_collisions_shared_boundary_port_overlap(self) -> None:
        """Overlapping child sub-boxes sharing a boundary port are detected as collisions."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0), 2: Coord3D(0, 1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )
        formulation = formulate_cluster_model(inventory=inv)

        # Offsets place both child sub-boxes at identical world coordinates
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (0, 2, 0)},
            raw_child_offsets={"c1": (2, 2, 0), "c2": (2, 2, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        child_child_viols = [v for v in violations if v.category == "child_child"]
        assert len(child_child_viols) == 1
        assert child_child_viols[0].item_a == ("c1", 0)
        assert child_child_viols[0].item_b == ("c2", 0)

    def test_detect_cluster_collisions_shared_boundary_port_disjoint(self) -> None:
        """Non-overlapping child sub-boxes sharing a boundary port produce zero child-child violations."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0), 2: Coord3D(0, 1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )
        formulation = formulate_cluster_model(inventory=inv)

        # Offsets place child sub-boxes disjointly
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (0, 2, 0)},
            raw_child_offsets={"c1": (2, 2, 0), "c2": (10, 10, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        child_child_viols = [v for v in violations if v.category == "child_child"]
        assert len(child_child_viols) == 0

    def test_child_child_separation_empty_sub_boxes_ignored(self) -> None:
        """Child clusters without sub-boxes do not generate separation constraints."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset())
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset())
        parent = SolveTreeNode(
            node_id="p_empty",
            node_type="container",
            rooms=frozenset(),
            children=(c1, c2),
        )

        sol1 = NodeSolution("c1", {}, sub_boxes=())
        sol2 = NodeSolution("c2", {}, sub_boxes=())

        graph = _make_graph([], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_child_child_separation()

        assert len(formulation.clearance_builder.separated_pairs) == 0
        assert not hasattr(formulation.model, "child_child_br")

class TestChildSubBoxCorridorClearance:
    """Unit tests for child keep-out sub-box corridor clearance and collision isolation."""

    def test_corridor_child_universal_obstacle_in_formulation(self) -> None:
        """Every child sub-box is formulated as an active obstacle against corridor interiors."""
        child = SolveTreeNode(node_id="c_branch", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_adapter", node_type="container", rooms=frozenset({100}), children=(child,))
        sol = NodeSolution(
            node_id="c_branch",
            local_coordinates={200: Coord3D(0, 5, 0)},
            port_coordinates={100: Coord3D(0, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 1, 4, 6, 0, 0),  # Sub-box 0: Enclosing interior room 200
            ),
        )

        graph = _make_graph([50, 100, 200], [(50, 100, Direction.north, 1)])
        inv = inventory_cluster(node=parent, child_solutions={"c_branch": sol}, graph=graph)
        inv = dataclasses.replace(inv, boundary_ports=frozenset({50}))
        ae = ActiveEdge(
            edge_id="int_50_100",
            src=50,
            dst=100,
            direction=(0, 1, 0),
            min_length=1,
            associated_rooms=frozenset({50, 100}),
            associated_children=frozenset(),
        )
        inv = dataclasses.replace(inv, active_edges=(ae,))
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_corridor_child_separation()

        # Every child sub-box is an active obstacle in the formulation with zero exceptions
        separated = formulation.clearance_builder.separated_pairs
        assert ("corridor_child", 0, ("c_branch", 0)) in separated

    def test_unrelated_corridor_traversing_child_interior_detected_and_separated(self) -> None:
        """Unrelated corridor traversing across a child cluster's interior is detected and separated."""
        child = SolveTreeNode(node_id="c_house", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_root", node_type="container", rooms=frozenset({10, 20}), children=(child,))
        sol = NodeSolution(
            node_id="c_house",
            local_coordinates={200: Coord3D(5, 0, 0)},
            sub_boxes=(
                BoundingBox3D(4, 6, -1, 1, 0, 0),  # Sub-box 0: Encloses room 200
            ),
        )

        graph = _make_graph([10, 20, 200], [(10, 20, Direction.east, 10)])
        inv = inventory_cluster(node=parent, child_solutions={"c_house": sol}, graph=graph)
        ae = ActiveEdge(
            edge_id="ae_street",
            src=10,
            dst=20,
            direction=(1, 0, 0),
            min_length=10,
            associated_rooms=frozenset({10, 20}),
            associated_children=frozenset(),
        )
        inv = dataclasses.replace(inv, active_edges=(ae,))
        formulation = ClusterFormulation(inventory=inv)

        # Extraction where corridor runs from (0, 0, 0) to (10, 0, 0), plowing through [4, 6]
        extraction = SolvedClusterExtraction(
            raw_parent_coords={10: (0, 0, 0), 20: (10, 0, 0)},
            raw_child_offsets={"c_house": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 1
        assert cchild_violations[0].item_b == ("c_house", 0)

        # Cut injection injects corridor_child separation cut
        cuts = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts == 1
        assert hasattr(formulation.model, "corridor_child_br")
        assert ("corridor_child", 0, ("c_house", 0)) in formulation.clearance_builder.separated_pairs


class TestClearanceCutPriorityAndLimits:
    """Unit tests for ClearanceConstraintBuilder cut limits and category priority tiering."""

    def test_violation_tier_priority_mapping(self) -> None:
        """Category tier priority reflects structural hierarchy from macroscopic to corridor crossings."""
        assert VIOLATION_TIER_PRIORITY["parent_room"] == 0
        assert VIOLATION_TIER_PRIORITY["child_child"] == 1
        assert VIOLATION_TIER_PRIORITY["room_child"] == 1
        assert VIOLATION_TIER_PRIORITY["corridor_room"] == 2
        assert VIOLATION_TIER_PRIORITY["corridor_child"] == 2
        assert VIOLATION_TIER_PRIORITY["corridor_crossing"] == 3

    def test_priority_tiering_injects_highest_priority_violations_first(self) -> None:
        """inject_clearance_cuts prioritizes higher tiers (0-1) over lower tiers (2-3) within budget."""
        formulation = MagicMock()
        formulation.options = LayoutOptions(max_cuts_per_pass=3)
        builder = ClearanceConstraintBuilder(formulation)

        violations = [
            ClearanceViolation(category="corridor_crossing", item_a=1, item_b=2),
            ClearanceViolation(category="corridor_room", item_a=0, item_b=10),
            ClearanceViolation(category="child_child", item_a=("c1", 0), item_b=("c2", 0)),
            ClearanceViolation(category="room_child", item_a=30, item_b=("c1", 0)),
            ClearanceViolation(category="parent_room", item_a=100, item_b=101),
        ]

        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            injected_count = builder.inject_clearance_cuts(violations)
            assert injected_count == 3
            # Tier 0 (parent_room), Tier 1 (child_child, room_child) should be injected
            assert ("parent_room", 100, 101) in builder.separated_pairs
            assert ("child_child", ("c1", 0), ("c2", 0)) in builder.separated_pairs
            assert ("room_child", 30, ("c1", 0)) in builder.separated_pairs
            # Lower tier violations (corridor_room, corridor_crossing) should NOT be injected in this pass
            assert ("corridor_room", 0, 10) not in builder.separated_pairs
            assert ("corridor_crossing", 1, 2) not in builder.separated_pairs
            assert mock_sep.call_count == 3

    def test_max_cuts_per_pass_budget_limit_and_sequential_batches(self) -> None:
        """max_cuts_per_pass truncates lower priority violations and permits subsequent batch injection."""
        formulation = MagicMock()
        formulation.options = LayoutOptions(max_cuts_per_pass=2)
        builder = ClearanceConstraintBuilder(formulation)

        violations = [
            ClearanceViolation(category="parent_room", item_a=1, item_b=2),
            ClearanceViolation(category="parent_room", item_a=3, item_b=4),
            ClearanceViolation(category="parent_room", item_a=5, item_b=6),
        ]

        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation"):
            # Pass 1: budget of 2 cuts from formulation options
            cuts1 = builder.inject_clearance_cuts(violations)
            assert cuts1 == 2
            assert len(builder.separated_pairs) == 2
            assert ("parent_room", 1, 2) in builder.separated_pairs
            assert ("parent_room", 3, 4) in builder.separated_pairs

            # Pass 2: remaining unseparated violation injected with explicit override limit
            cuts2 = builder.inject_clearance_cuts(violations, max_cuts_per_pass=1)
            assert cuts2 == 1
            assert len(builder.separated_pairs) == 3
            assert ("parent_room", 5, 6) in builder.separated_pairs

            # Pass 3: all candidate violations are already separated
            cuts3 = builder.inject_clearance_cuts(violations)
            assert cuts3 == 0

    def test_inject_clearance_cuts_empty_and_zero_budget(self) -> None:
        """inject_clearance_cuts returns zero when violations is empty or budget <= 0."""
        formulation = MagicMock()
        builder = ClearanceConstraintBuilder(formulation)
        assert builder.inject_clearance_cuts([]) == 0

        v = [ClearanceViolation(category="parent_room", item_a=1, item_b=2)]
        assert builder.inject_clearance_cuts(v, max_cuts_per_pass=0) == 0
        assert builder.inject_clearance_cuts(v, max_cuts_per_pass=-1) == 0
        assert len(builder.separated_pairs) == 0

    def test_inject_clearance_cuts_deduplicates_in_same_batch(self) -> None:
        """Duplicate violations in the same input batch count once against budget."""
        formulation = MagicMock()
        builder = ClearanceConstraintBuilder(formulation)
        violations = [
            ClearanceViolation(category="parent_room", item_a=1, item_b=2),
            ClearanceViolation(category="parent_room", item_a=2, item_b=1),
            ClearanceViolation(category="child_child", item_a=("c1", 0), item_b=("c2", 0)),
            ClearanceViolation(category="child_child", item_a=("c2", 0), item_b=("c1", 0)),
        ]
        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation"):
            cuts = builder.inject_clearance_cuts(violations, max_cuts_per_pass=10)
            assert cuts == 2
            assert len(builder.separated_pairs) == 2

    def test_get_corridor_boxes_typed_entities(self) -> None:
        """get_corridor_interior_box and get_corridor_full_box resolve CorridorClearanceBox and BoundaryExitClearanceBox."""
        from romutil.layout.milp.clearance import (
            BoundaryExitClearanceBox,
            CorridorClearanceBox,
        )

        formulation = MagicMock()
        formulation.inventory.active_edges = ["edge0", "edge1"]
        formulation.inventory.boundary_exits = ["exit0", "exit1"]
        formulation.get_corridor_interior_box.return_value = (0, 0, 0, 1, 1, 1)
        formulation.get_corridor_full_box.return_value = (0, 0, 0, 2, 2, 2)
        formulation.get_boundary_exit_interior_box.return_value = (3, 3, 3, 4, 4, 4)
        formulation.get_boundary_exit_full_box.return_value = (3, 3, 3, 5, 5, 5)

        builder = ClearanceConstraintBuilder(formulation)

        c_box = CorridorClearanceBox(edge_index=1, edge_id="e1")
        b_box = BoundaryExitClearanceBox(exit_index=0, edge_id="be0")

        assert builder.get_corridor_interior_box(c_box) == (0, 0, 0, 1, 1, 1)
        formulation.get_corridor_interior_box.assert_called_with("edge1")

        assert builder.get_corridor_full_box(c_box) == (0, 0, 0, 2, 2, 2)
        formulation.get_corridor_full_box.assert_called_with("edge1")

        assert builder.get_corridor_interior_box(b_box) == (3, 3, 3, 4, 4, 4)
        formulation.get_boundary_exit_interior_box.assert_called_with("exit0")

        assert builder.get_corridor_full_box(b_box) == (3, 3, 3, 5, 5, 5)
        formulation.get_boundary_exit_full_box.assert_called_with("exit0")

    def test_inject_clearance_cuts_typed_entities(self) -> None:
        """inject_clearance_cuts sorts and injects violations using typed clearance box objects."""
        from romutil.layout.milp.clearance import (
            BoundaryExitClearanceBox,
            ChildSubClearanceBox,
            CorridorClearanceBox,
            RoomClearanceBox,
        )

        formulation = MagicMock()
        builder = ClearanceConstraintBuilder(formulation)
        violations = [
            ClearanceViolation(
                category="room_child",
                item_a=RoomClearanceBox(10),
                item_b=ChildSubClearanceBox("child_1", 0),
            ),
            ClearanceViolation(
                category="corridor_room",
                item_a=CorridorClearanceBox(0, "e0"),
                item_b=RoomClearanceBox(20),
            ),
            ClearanceViolation(
                category="corridor_crossing",
                item_a=BoundaryExitClearanceBox(0, "b0"),
                item_b=CorridorClearanceBox(1, "e1"),
            ),
        ]
        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            cuts = builder.inject_clearance_cuts(violations, max_cuts_per_pass=10)
            assert cuts == 3
            assert len(builder.separated_pairs) == 3
            assert mock_sep.call_count == 3
