"""Tests for the 1D positional constraint DAG and pre-injection clearance cut filtering."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
import pyomo.environ as pyo
import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import NodeSolution, SolveTreeNode
from romutil.layout.milp.clearance import ClearanceConstraintBuilder, ClearanceViolation
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.formulation import ClusterFormulation
from romutil.layout.milp.inventory import (
    ActiveEdge,
    ChildChildAlignment,
    ClusterInventory,
)
from romutil.layout.milp.spatial_dag import (
    ALL_DIRECTIONS,
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    DIR_AHEAD,
    DIR_BEHIND,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    PositionalConstraintDAG,
    SpatialBoundExpr,
    get_entity_spatial_bounds,
    resolve_endpoint_node_offset,
)
from romutil.types import Coord3D, LayoutOptions


class TestPositionalConstraintDAG:
    """Unit tests for the 1D positional difference constraint DAG core logic."""

    def test_init_and_lower_bounds(self) -> None:
        """Initial DAG has 0 distance to self and None for disconnected pairs."""
        dag = PositionalConstraintDAG()
        assert dag.get_lower_bound(AXIS_X, 1, 1) == 0
        assert dag.get_lower_bound(AXIS_X, 1, 2) is None

        dag.add_edge(AXIS_X, 1, 2, 5)
        assert dag.get_lower_bound(AXIS_X, 1, 2) == 5
        assert dag.get_lower_bound(AXIS_X, 2, 1) is None
        # Other axes remain untouched
        assert dag.get_lower_bound(AXIS_Y, 1, 2) is None
        assert dag.get_lower_bound(AXIS_Z, 1, 2) is None

    def test_longest_path_propagation(self) -> None:
        """Adding edges propagates transitive lower bounds along directed paths."""
        dag = PositionalConstraintDAG()
        assert dag.add_edge(AXIS_X, "A", "B", 4)
        assert dag.add_edge(AXIS_X, "B", "C", 6)
        assert dag.get_lower_bound(AXIS_X, "A", "C") == 10

        assert dag.add_edge(AXIS_X, "C", "D", 2)
        assert dag.get_lower_bound(AXIS_X, "A", "D") == 12

        # A tighter direct edge improves the lower bound
        assert dag.add_edge(AXIS_X, "A", "D", 15)
        assert dag.get_lower_bound(AXIS_X, "A", "D") == 15

    def test_cycle_and_contradiction_detection(self) -> None:
        """Edges that would introduce a positive cycle are rejected."""
        dag = PositionalConstraintDAG()
        # A -(5)-> B enforces coord(B) - coord(A) >= 5
        assert dag.add_edge(AXIS_X, "A", "B", 5)

        # Candidate edge B -(-5)-> A means coord(A) - coord(B) >= -5 (cycle weight 5 + (-5) = 0 <= 0: allowed)
        assert dag.can_add_edge(AXIS_X, "B", "A", -5)
        # Candidate edge B -(-4)-> A means coord(A) - coord(B) >= -4 (cycle weight 5 - 4 = 1 > 0: positive cycle!)
        assert not dag.can_add_edge(AXIS_X, "B", "A", -4)
        assert not dag.add_edge(AXIS_X, "B", "A", -4)

        # Enforcing coord(A) - coord(B) >= 1 is strictly contradicted
        assert not dag.can_add_edge(AXIS_X, "B", "A", 1)
        assert not dag.add_edge(AXIS_X, "B", "A", 1)

    def test_self_loops(self) -> None:
        """Self-loops with positive weight are contradicted; non-positive are trivial."""
        dag = PositionalConstraintDAG()
        assert not dag.can_add_edge(AXIS_X, 10, 10, 1)
        assert not dag.add_edge(AXIS_X, 10, 10, 1)

        assert dag.can_add_edge(AXIS_X, 10, 10, 0)
        assert dag.add_edge(AXIS_X, 10, 10, 0)

        assert dag.can_add_edge(AXIS_X, 10, 10, -5)
        assert dag.add_edge(AXIS_X, 10, 10, -5)

    def test_equality_constraints(self) -> None:
        """Bidirectional equality edges enforce exact coordinate displacement."""
        dag = PositionalConstraintDAG()
        # coord(B) - coord(A) == 3
        assert dag.add_equality(AXIS_Y, "A", "B", 3)
        assert dag.get_lower_bound(AXIS_Y, "A", "B") == 3
        assert dag.get_lower_bound(AXIS_Y, "B", "A") == -3

        # Requiring coord(B) - coord(A) >= 4 contradicts coord(B) - coord(A) <= 3
        assert not dag.can_add_edge(AXIS_Y, "A", "B", 4)
        # Requiring coord(A) - coord(B) >= -2 contradicts coord(A) - coord(B) <= -3
        assert not dag.can_add_edge(AXIS_Y, "B", "A", -2)


class TestSpatialBoundResolution:
    """Tests for mapping layout entities into affine spatial bound expressions."""

    def test_resolve_endpoint_node_offset(self) -> None:
        """Room integers resolve with zero offset; child ports resolve with local offset."""
        # Parent room integer
        node, off = resolve_endpoint_node_offset(101, None)
        assert node == 101
        assert off == (0, 0, 0)

        # Child port reference (cid, room_id)
        mock_sol = NodeSolution(
            node_id="child_1",
            local_coordinates={201: Coord3D(3, -4, 1)},
            port_coordinates={},
            sub_boxes=(),
        )
        mock_inv = MagicMock()
        mock_inv.child_solutions = {"child_1": mock_sol}

        node_c, off_c = resolve_endpoint_node_offset(("child_1", 201), mock_inv)
        assert node_c == "child_1"
        assert off_c == (3, -4, 1)

    def test_get_entity_spatial_bounds_parent_room(self) -> None:
        """Parent room entities resolve to zero-span point expressions."""
        bounds = get_entity_spatial_bounds(105, "parent_room", None)
        assert bounds is not None
        for b in bounds:
            assert b.min_node == 105
            assert b.max_node == 105
            assert b.min_const == 0
            assert b.max_const == 0

    def test_get_entity_spatial_bounds_child_sub_box(self) -> None:
        """Child sub-boxes resolve to child origin variable with constant local bounding box offsets."""
        mock_inv = MagicMock()
        mock_inv.child_sub_boxes = {
            "c_spqr": [BoundingBox3D(1, 4, -2, 5, 0, 0)]
        }
        bounds = get_entity_spatial_bounds(("c_spqr", 0), "child_child", mock_inv)
        assert bounds is not None
        assert bounds[0] == SpatialBoundExpr("c_spqr", 1, "c_spqr", 4)
        assert bounds[1] == SpatialBoundExpr("c_spqr", -2, "c_spqr", 5)
        assert bounds[2] == SpatialBoundExpr("c_spqr", 0, "c_spqr", 0)

    def test_get_entity_spatial_bounds_corridor_interior_and_full(self) -> None:
        """Corridor interior bounds exclude endpoints; full bounds include endpoints."""
        ae = ActiveEdge(
            edge_id="e1",
            src=10,
            dst=20,
            direction=(1, 0, 0),
            min_length=5,
            associated_rooms=frozenset({10, 20}),
            associated_children=frozenset(),
        )
        mock_inv = MagicMock()
        mock_inv.active_edges = [ae]

        # Interior bounds (corridor_room): min_x = src + 1, max_x = dst - 1
        int_bounds = get_entity_spatial_bounds(0, "corridor_room", mock_inv, is_item_a=True)
        assert int_bounds is not None
        assert int_bounds[0] == SpatialBoundExpr(10, 1, 20, -1)
        assert int_bounds[1] == SpatialBoundExpr(10, 0, 10, 0)

        # Full bounds (corridor_crossing): min_x = src, max_x = dst
        full_bounds = get_entity_spatial_bounds(0, "corridor_crossing", mock_inv, is_item_a=True)
        assert full_bounds is not None
        assert full_bounds[0] == SpatialBoundExpr(10, 0, 20, 0)
        assert full_bounds[1] == SpatialBoundExpr(10, 0, 10, 0)

    def test_get_entity_spatial_bounds_typed_entities(self) -> None:
        """Typed clearance entity dataclasses resolve identically to raw index keys."""
        from romutil.layout.milp.clearance import (
            BoundaryExitClearanceBox,
            ChildSubClearanceBox,
            CorridorClearanceBox,
            RoomClearanceBox,
        )
        from romutil.layout.milp.inventory import BoundaryExit

        ae = ActiveEdge(
            edge_id="e1",
            src=10,
            dst=20,
            direction=(1, 0, 0),
            min_length=5,
            associated_rooms=frozenset({10, 20}),
            associated_children=frozenset(),
        )
        be = BoundaryExit(
            port=10,
            direction=(0, 1, 0),
            length=3,
            target_vnum=999,
        )
        mock_inv = MagicMock()
        mock_inv.active_edges = [ae]
        mock_inv.boundary_exits = [be]
        mock_inv.child_sub_boxes = {
            "c_spqr": [BoundingBox3D(1, 4, -2, 5, 0, 0)]
        }

        # RoomClearanceBox
        r_box = RoomClearanceBox(vnum=105)
        bounds_r = get_entity_spatial_bounds(r_box, "parent_room", None)
        assert bounds_r is not None
        assert bounds_r[0] == SpatialBoundExpr(105, 0, 105, 0)

        # ChildSubClearanceBox
        cs_box = ChildSubClearanceBox(child_id="c_spqr", box_index=0)
        bounds_cs = get_entity_spatial_bounds(cs_box, "child_child", mock_inv)
        assert bounds_cs is not None
        assert bounds_cs[0] == SpatialBoundExpr("c_spqr", 1, "c_spqr", 4)

        # CorridorClearanceBox
        cor_box = CorridorClearanceBox(edge_index=0, edge_id="e1")
        bounds_cor = get_entity_spatial_bounds(cor_box, "corridor_room", mock_inv, is_item_a=True)
        assert bounds_cor is not None
        assert bounds_cor[0] == SpatialBoundExpr(10, 1, 20, -1)

        # BoundaryExitClearanceBox
        bexit_box = BoundaryExitClearanceBox(exit_index=0, edge_id="be1")
        bounds_bexit = get_entity_spatial_bounds(bexit_box, "corridor_room", mock_inv, is_item_a=True)
        assert bounds_bexit is not None
        assert bounds_bexit[1] == SpatialBoundExpr(10, 1, 10, 3)


class TestCandidateDirectionFiltering:
    """Tests for pre-injection filtering of candidate separation directions against the DAG."""

    def test_active_edge_seeds_dag_and_prunes_contradicted_directions(self) -> None:
        """Active edge from 10 to 20 along +X allows only left separation (20 to right of 10)."""
        ae = ActiveEdge(
            edge_id="e_x",
            src=10,
            dst=20,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({10, 20}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        mock_inv = MagicMock()
        mock_inv.active_edges = [ae]
        mock_inv.child_alignments = ()

        mock_form = MagicMock()
        mock_form.inventory = mock_inv

        dag = PositionalConstraintDAG.from_formulation(mock_form)
        valid_dirs, pruned = dag.filter_candidate_directions(10, 20, "parent_room", mock_inv)

        # Direction 'r': 10 > 20 along X requires x(10) - x(20) >= 1 (contradicts x(20) - x(10) >= 2)
        assert DIR_RIGHT not in valid_dirs
        assert DIR_RIGHT in pruned

        # Direction 'l': 20 > 10 along X requires x(20) - x(10) >= 1 (consistent with min_length 2)
        assert DIR_LEFT in valid_dirs

        # Transverse axes Y and Z remain unconstrained by X-aligned edge, allowing transverse separation
        assert DIR_AHEAD in valid_dirs
        assert DIR_BEHIND in valid_dirs
        assert DIR_UP in valid_dirs
        assert DIR_DOWN in valid_dirs

    def test_unsatisfiable_collision_detection(self) -> None:
        """When two items are rigidly constrained to exact coordinate equality, all directions are pruned."""
        # Rooms 10 and 20 constrained to exact coordinate equality on all 3 axes
        dag = PositionalConstraintDAG()
        dag.add_equality(AXIS_X, 10, 20, 0)
        dag.add_equality(AXIS_Y, 10, 20, 0)
        dag.add_equality(AXIS_Z, 10, 20, 0)

        valid_dirs, pruned = dag.filter_candidate_directions(10, 20, "parent_room", None)
        assert valid_dirs == []
        assert len(pruned) == 6


class TestClearanceConstraintBuilderDAGIntegration:
    """Integration tests verifying ClearanceConstraintBuilder behavior with the positional DAG."""

    def test_inject_clearance_cuts_filters_contradicted_directions(self) -> None:
        """inject_clearance_cuts filters contradicted directions and passes allowed_directions to builder."""
        ae = ActiveEdge(
            edge_id="e_x",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        mock_inv = MagicMock()
        mock_inv.active_edges = [ae]
        mock_inv.child_alignments = ()

        formulation = MagicMock()
        formulation.inventory = mock_inv
        formulation.options = LayoutOptions(max_cuts_per_pass=5)
        formulation.m_big = 100

        builder = ClearanceConstraintBuilder(formulation)
        violations = [ClearanceViolation(category="parent_room", item_a=1, item_b=2)]

        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            cuts_injected = builder.inject_clearance_cuts(violations)
            assert cuts_injected == 1
            assert ("parent_room", 1, 2) in builder.separated_pairs
            assert mock_sep.call_count == 1
            # Check allowed_directions prunes DIR_RIGHT
            args, kwargs = mock_sep.call_args
            allowed_dirs = kwargs.get("allowed_directions")
            assert allowed_dirs is not None
            assert DIR_RIGHT not in allowed_dirs[(1, 2)]
            assert DIR_LEFT in allowed_dirs[(1, 2)]

        assert len(builder.unsatisfiable_collisions) == 0

    def test_inject_clearance_cuts_prunes_unsatisfiable_collision(self) -> None:
        """inject_clearance_cuts classifies topologically deadlocked collisions as UNSATISFIABLE_COLLISION."""
        mock_inv = MagicMock()
        mock_inv.active_edges = ()
        mock_inv.child_alignments = ()

        formulation = MagicMock()
        formulation.inventory = mock_inv
        formulation.options = LayoutOptions(max_cuts_per_pass=5)
        formulation.m_big = 100

        builder = ClearanceConstraintBuilder(formulation)
        # Artificially enforce equality on all 3 axes between 1 and 2
        builder.spatial_dag.add_equality(AXIS_X, 1, 2, 0)
        builder.spatial_dag.add_equality(AXIS_Y, 1, 2, 0)
        builder.spatial_dag.add_equality(AXIS_Z, 1, 2, 0)

        viol = ClearanceViolation(category="parent_room", item_a=1, item_b=2)
        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            cuts_injected = builder.inject_clearance_cuts([viol])
            assert cuts_injected == 0
            assert mock_sep.call_count == 0

        assert len(builder.unsatisfiable_collisions) == 1
        assert builder.unsatisfiable_collisions[0] == viol
        assert ("parent_room", 1, 2) in builder.separated_pairs

    def test_inject_clearance_cuts_mixed_resolvable_and_unresolvable(self) -> None:
        """Resolvable violations are injected even if an unresolvable collision is present."""
        mock_inv = MagicMock()
        mock_inv.active_edges = ()
        mock_inv.child_alignments = ()

        formulation = MagicMock()
        formulation.inventory = mock_inv
        formulation.options = LayoutOptions(max_cuts_per_pass=5)
        formulation.m_big = 100

        builder = ClearanceConstraintBuilder(formulation)
        # Deadlock pair (1, 2)
        builder.spatial_dag.add_equality(AXIS_X, 1, 2, 0)
        builder.spatial_dag.add_equality(AXIS_Y, 1, 2, 0)
        builder.spatial_dag.add_equality(AXIS_Z, 1, 2, 0)

        # Resolvable pair (3, 4) has no constraints
        viol_deadlock = ClearanceViolation(category="parent_room", item_a=1, item_b=2)
        viol_resolvable = ClearanceViolation(category="parent_room", item_a=3, item_b=4)

        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            cuts_injected = builder.inject_clearance_cuts([viol_deadlock, viol_resolvable])
            assert cuts_injected == 1
            assert mock_sep.call_count == 1
            args, kwargs = mock_sep.call_args
            allowed_dirs = kwargs.get("allowed_directions")
            assert (3, 4) in allowed_dirs

        assert len(builder.unsatisfiable_collisions) == 1
        assert builder.unsatisfiable_collisions[0] == viol_deadlock
        assert ("parent_room", 3, 4) in builder.separated_pairs

    def test_commit_direction_propagates_to_subsequent_candidates(self) -> None:
        """Fixing or committing a single valid direction updates the DAG for subsequent pairs."""
        dag = PositionalConstraintDAG()
        # Item A (room 1) and Item B (room 2)
        # Commit direction 'l': 2 is to the right of 1 (x(2) - x(1) >= 1)
        assert dag.commit_direction(1, 2, "parent_room", None, DIR_LEFT)
        assert dag.get_lower_bound(AXIS_X, 1, 2) == 1

        # Now check candidate separation between 2 and 3 where 3 has active edge to 1
        dag.add_edge(AXIS_X, 3, 1, 2)  # x(1) - x(3) >= 2
        # Then x(2) - x(3) >= 3
        assert dag.get_lower_bound(AXIS_X, 3, 2) == 3
        # Candidate 3 > 2 along X requires x(3) - x(2) >= 1, which contradicts x(2) - x(3) >= 3
        valid_dirs, pruned = dag.filter_candidate_directions(3, 2, "parent_room", None)
        assert DIR_RIGHT not in valid_dirs

    def test_add_disjunctive_box_separation_with_allowed_directions_pyomo(self) -> None:
        """add_disjunctive_box_separation fixes inactive directions to 0 on a real Pyomo model."""
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2], bounds=(-100, 100))
        model.y = pyo.Var([1, 2], bounds=(-100, 100))
        model.z = pyo.Var([1, 2], bounds=(-100, 100))

        def get_box(u: int) -> tuple[Any, Any, Any, Any, Any, Any]:
            return (model.x[u], model.x[u], model.y[u], model.y[u], model.z[u], model.z[u])

        pair = (1, 2)
        allowed = {pair: ["l", "u"]}

        add_disjunctive_box_separation(
            model=model,
            pairs=[pair],
            get_box_a=get_box,
            get_box_b=get_box,
            prefix="test_sep",
            m_big=200,
            allowed_directions=allowed,
        )

        # Variables r, a, b, d should be fixed to 0
        assert model.test_sep_br[0].fixed
        assert model.test_sep_br[0].value == 0
        assert model.test_sep_ba[0].fixed
        assert model.test_sep_ba[0].value == 0
        assert model.test_sep_bb[0].fixed
        assert model.test_sep_bb[0].value == 0
        assert model.test_sep_bd[0].fixed
        assert model.test_sep_bd[0].value == 0

        # Variables l and u should NOT be fixed to 0
        assert not model.test_sep_bl[0].fixed
        assert not model.test_sep_bu[0].fixed

        # Only sep_l and sep_u constraints were added
        assert len(model.test_sep_l) == 1
        assert len(model.test_sep_u) == 1
        assert len(model.test_sep_r) == 0
        assert len(model.test_sep_a) == 0
        assert len(model.test_sep_b) == 0
        assert len(model.test_sep_d) == 0
        assert len(model.test_sep_sum) == 1

    def test_ushape_concave_corridor_deadlock_pruned(self) -> None:
        """A piercing corridor with min_length preventing separation from a concave wing is detected as deadlock."""
        import dataclasses
        from romutil.types import Coord3D, SolverStatus
        from romutil.layout.milp.clearance import detect_cluster_collisions
        from romutil.layout.milp.formulation import formulate_cluster_model
        from romutil.layout.milp.inventory import inventory_cluster
        from romutil.layout.milp.solver import SolvedClusterExtraction
        from tests.test_layout_milp_clearance import _make_graph

        child = SolveTreeNode(node_id="c_ushape", node_type="leaf", rooms=frozenset({10, 20}))
        parent = SolveTreeNode(node_id="p_root", node_type="container", rooms=frozenset({1, 2}), children=(child,))
        sol = NodeSolution(
            node_id="c_ushape",
            local_coordinates={
                10: Coord3D(1, 1, 0),
                20: Coord3D(5, 1, 0),
            },
            sub_boxes=(
                BoundingBox3D(0, 1, 0, 2, 0, 0),
                BoundingBox3D(4, 6, 0, 2, 0, 0),
            ),
        )
        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c_ushape": sol}, graph=graph)
        inv = dataclasses.replace(
            inv,
            active_edges=(
                ActiveEdge(
                    edge_id="ae_piercing",
                    src=("c_ushape", 10),
                    dst=2,
                    direction=(1, 0, 0),
                    min_length=7,
                    associated_rooms=frozenset({10, 2}),
                    associated_children=frozenset({"c_ushape"}),
                ),
            ),
        )
        formulation = formulate_cluster_model(inventory=inv)
        extraction = SolvedClusterExtraction(
            raw_parent_coords={2: (8, 1, 0)},
            raw_child_offsets={"c_ushape": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations = detect_cluster_collisions(formulation, extraction)
        assert len(violations) == 1

        cuts_injected = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_injected == 0
        assert len(formulation.clearance_builder.unsatisfiable_collisions) == 1
        assert formulation.clearance_builder.unsatisfiable_collisions[0] == violations[0]
