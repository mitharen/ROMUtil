"""Comprehensive unit and integration tests for component decomposition and incremental solving (Task 1e).

Verifies:
1. Connected Component Partitioning:
   - Decomposing disconnected room graphs into independent weakly connected components.
   - Multi-component (2, 3+ components) partitioned and packed with non-overlapping bounding boxes.
   - 1D horizontal shelf packing along the X-axis with configurable padding >= 2.
   - Padding clamping for values < 2.
2. Formulation & Performance Acceptance Metrics:
   - Maximum decision variables per MILP instance reduces from |V| to max_i |V(C_i)|.
   - Independent subproblems solve without monolithic model inflation or cross-component overlap constraint generation.
3. Edge Cases & Robustness:
   - Single-component pass-through (k=1) preserves direct solve path.
   - Isolated rooms (components with 0 exits).
   - Multi-elevation 3D components.
   - Component solve with solver timeout / partial recovery (maxTimeLimit with feasible coords).
   - Component solve with infeasible / failure fallback.
   - Dummy room anchoring and inclusion in component bounding boxes.
   - Straight hallway collapse and restoration within decomposed components.
   - Helper functions: compute_bounding_box, decompose_components.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock
import pyomo.opt
import pytest

from romutil.graph import (
    compute_bounding_box,
    decompose_components,
    graph,
    solve_layout,
    _has_feasible_coordinates,
)
from romutil.models import AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.solver import solve


class MockModelCoords:
    """Mock Pyomo model variables for coordinates and cuts."""

    def __init__(self, coord_map, cut_map=None):
        self.x = {v: MagicMock(value=c[0]) for v, c in coord_map.items()}
        self.y = {v: MagicMock(value=c[1]) for v, c in coord_map.items()}
        self.z = {v: MagicMock(value=c[2]) for v, c in coord_map.items()}
        cut_map = cut_map or {}
        self.cut = {i: MagicMock(value=v) for i, v in cut_map.items()}


class MockSolverResults:
    """Mock Pyomo solver results object."""

    def __init__(self, termination_condition):
        self.solver = MagicMock()
        self.solver.termination_condition = termination_condition
        self.solver.__str__.return_value = f"MockSolverResult({termination_condition})"


class TestHelperFunctions:
    """Unit tests for compute_bounding_box and decompose_components."""

    def test_compute_bounding_box_empty(self):
        """Empty iterable returns (0, 0, 0, 0, 0, 0)."""
        assert compute_bounding_box([]) == (0, 0, 0, 0, 0, 0)

    def test_compute_bounding_box_none_coordinates(self):
        """Rooms with None coordinates are ignored; returns zeroes if all None."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        assert compute_bounding_box([r1, r2]) == (0, 0, 0, 0, 0, 0)

    def test_compute_bounding_box_valid_rooms(self):
        """Correctly computes min/max extents across X, Y, Z."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r1.x, r1.y, r1.z = 2, 5, -1
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r2.x, r2.y, r2.z = 10, -3, 4
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))
        r3.x, r3.y, r3.z = None, None, None

        bbox = compute_bounding_box([r1, r2, r3])
        assert bbox == (2, 10, -3, 5, -1, 4)

    def test_decompose_components_empty(self):
        """Empty rdb produces empty components list."""
        assert decompose_components({}, []) == []

    def test_decompose_components_single_component(self):
        """Connected rooms form a single component."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        e1 = Exit(ExitDef(direction=1, dst_vnum=2), source=1)
        e2 = Exit(ExitDef(direction=3, dst_vnum=1), source=2)
        r1.exits = [e1]
        r2.exits = [e2]
        comps = decompose_components({1: r1, 2: r2}, [e1, e2])
        assert len(comps) == 1
        assert comps[0] == {1, 2}

    def test_decompose_components_ignores_dummy_rooms(self):
        """Dummy rooms are not nodes in the component graph and do not bridge components."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        dummy = Room(RoomDef(vnum=99, name="External", description=""))
        dummy.dummy = True

        # Both r1 and r2 have exits pointing to dummy room 99
        e1 = Exit(ExitDef(direction=1, dst_vnum=99), source=1)
        e2 = Exit(ExitDef(direction=3, dst_vnum=99), source=2)
        r1.exits = [e1]
        r2.exits = [e2]

        comps = decompose_components({1: r1, 2: r2, 99: dummy}, [e1, e2])
        assert len(comps) == 2
        assert comps[0] == {1}
        assert comps[1] == {2}

    def test_decompose_components_deterministic_ordering(self):
        """Components are sorted deterministically by min VNUM."""
        r100 = Room(RoomDef(vnum=100, name="R100", description=""))
        r10 = Room(RoomDef(vnum=10, name="R10", description=""))
        r50 = Room(RoomDef(vnum=50, name="R50", description=""))
        comps = decompose_components({100: r100, 10: r10, 50: r50}, [])
        assert len(comps) == 3
        assert comps[0] == {10}
        assert comps[1] == {50}
        assert comps[2] == {100}


@pytest.mark.slow
@pytest.mark.integration
class TestComponentDecompositionSolving:
    """Integration tests for multi-component solving and shelf packing."""

    def test_single_component_pass_through(self):
        """Single connected component (k=1) preserves direct solve path."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10)
        assert len(solved_rdb) == 2
        assert solved_rdb[1].x is not None
        assert solved_rdb[2].x is not None
        assert solved_rdb[1].y is not None
        assert solved_rdb[2].y is not None

        # Normalized to 0 minimum
        min_x = min(r.x for r in solved_rdb.values())
        min_y = min(r.y for r in solved_rdb.values())
        min_z = min(r.z for r in solved_rdb.values())
        assert min_x == 0
        assert min_y == 0
        assert min_z == 0

    def test_two_components_partitioned_and_packed(self):
        """Two disconnected components are solved and packed along X with padding."""
        # Component 1: Rooms 10, 11 (East-West)
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))
        # Component 2: Rooms 20, 21 (East-West)
        r20 = Room(RoomDef(vnum=20, name="R20", description="", exits=(ExitDef(direction=1, dst_vnum=21),)))
        r21 = Room(RoomDef(vnum=21, name="R21", description="", exits=(ExitDef(direction=3, dst_vnum=20),)))

        rdb = {10: r10, 11: r11, 20: r20, 21: r21}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=3)
        assert len(solved_rdb) == 4

        c1_rooms = [solved_rdb[10], solved_rdb[11]]
        c2_rooms = [solved_rdb[20], solved_rdb[21]]

        c1_max_x = max(r.x for r in c1_rooms)
        c2_min_x = min(r.x for r in c2_rooms)

        # Gap between Component 1 and Component 2 must be >= padding (3)
        assert c2_min_x - c1_max_x >= 3

        # Bounding boxes do not overlap in 3D
        c1_bbox = compute_bounding_box(c1_rooms)
        c2_bbox = compute_bounding_box(c2_rooms)
        assert c1_bbox[1] < c2_bbox[0]

    def test_three_components_with_configurable_padding(self):
        """Three components packed in sequence along the X-axis."""
        # C1: 1 -> 2
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        # C2: 10 -> 11
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))
        # C3: 20 -> 21
        r20 = Room(RoomDef(vnum=20, name="R20", description="", exits=(ExitDef(direction=1, dst_vnum=21),)))
        r21 = Room(RoomDef(vnum=21, name="R21", description="", exits=(ExitDef(direction=3, dst_vnum=20),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11, 20: r20, 21: r21}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=4)
        c1 = [solved_rdb[1], solved_rdb[2]]
        c2 = [solved_rdb[10], solved_rdb[11]]
        c3 = [solved_rdb[20], solved_rdb[21]]

        b1 = compute_bounding_box(c1)
        b2 = compute_bounding_box(c2)
        b3 = compute_bounding_box(c3)

        assert b1[1] + 4 <= b2[0]
        assert b2[1] + 4 <= b3[0]

    def test_padding_clamped_to_minimum_two(self, caplog):
        """component_padding < 2 is clamped to 2 with a warning."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11}

        with caplog.at_level(logging.WARNING):
            solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=0)

        assert "clamping to 2" in caplog.text
        c1 = [solved_rdb[1], solved_rdb[2]]
        c2 = [solved_rdb[10], solved_rdb[11]]
        assert min(r.x for r in c2) - max(r.x for r in c1) >= 2

    def test_multi_elevation_components(self):
        """Components with distinct elevation layers (Z-axis) pack disjointly along X."""
        # Component 1: Vertical Up/Down between 1 and 2
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=4, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=5, dst_vnum=1),)))
        # Component 2: Vertical Up/Down between 10 and 11
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=4, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=5, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=3)

        c1 = [solved_rdb[1], solved_rdb[2]]
        c2 = [solved_rdb[10], solved_rdb[11]]

        b1 = compute_bounding_box(c1)
        b2 = compute_bounding_box(c2)

        # Disjoint along X ensures zero 3D overlap across elevations
        assert b1[1] < b2[0]
        # Both components have vertical span
        assert b1[5] - b1[4] >= 1
        assert b2[5] - b2[4] >= 1

    def test_isolated_room_component(self):
        """Isolated room with no exits is partitioned into its own component and shelf packed."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        r99 = Room(RoomDef(vnum=99, name="Isolated", description="", exits=()))

        rdb = {1: r1, 2: r2, 99: r99}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=2)
        assert len(solved_rdb) == 3
        for r in solved_rdb.values():
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None

        c1_max_x = max(solved_rdb[1].x, solved_rdb[2].x)
        assert solved_rdb[99].x >= c1_max_x + 2

    def test_dummy_room_anchored_within_component_bounds(self):
        """Dummy room created for out-of-area exit is included in component bounding box."""
        # Room 1 exits east to out-of-area room 999 (becomes dummy)
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=999),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        # Component 2: 10 -> 11
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=3)
        assert 999 in solved_rdb
        assert solved_rdb[999].dummy is True

        c1_rooms = [solved_rdb[1], solved_rdb[2], solved_rdb[999]]
        c2_rooms = [solved_rdb[10], solved_rdb[11]]

        c1_max_x = max(r.x for r in c1_rooms)
        c2_min_x = min(r.x for r in c2_rooms)

        # Dummy room 999 must not overlap Component 2
        assert c2_min_x - c1_max_x >= 3


class TestDecisionVariableReductionMetric:
    """Acceptance Metric: maximum decision variables per MILP instance reduces to max_i |V(C_i)|."""

    def test_decision_variable_reduction_on_disconnected_graphs(self, monkeypatch):
        """For k disconnected subproblems, MILP model size reflects max_i |V(C_i)| instead of |V|."""
        # C1 has 4 rooms: 1-4
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1), ExitDef(direction=0, dst_vnum=3))))
        r3 = Room(RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=2, dst_vnum=2), ExitDef(direction=1, dst_vnum=4))))
        r4 = Room(RoomDef(vnum=4, name="R4", description="", exits=(ExitDef(direction=3, dst_vnum=3),)))

        # C2 has 2 rooms: 10-11
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 3: r3, 4: r4, 10: r10, 11: r11}

        # Track model room sizes across solve() calls
        model_room_counts = []
        import sys
        graph_mod = sys.modules['romutil.graph']
        orig_solve = graph_mod.solve

        def spy_solve(sub_rdb, sub_exits, **kwargs):
            model, results = orig_solve(sub_rdb, sub_exits, **kwargs)
            model_room_counts.append(len(model.Rooms))
            return model, results

        monkeypatch.setattr(graph_mod, "solve", spy_solve)

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10)

        # Monolithic solve would have 6 rooms (18 variables)
        total_rooms = len(rdb)
        assert total_rooms == 6

        # Decomposed solve had 2 independent solve() calls
        assert len(model_room_counts) == 2
        assert model_room_counts[0] == 4  # C1 has 4 rooms
        assert model_room_counts[1] == 2  # C2 has 2 rooms
        assert max(model_room_counts) == 4  # Max per instance is 4, not 6!


class TestSolverRecoveryAndFallbackInComponents:
    """Unit tests for recovery and fallback behavior across multiple components."""

    def test_component_feasible_solution_recovery_on_timeout(self, monkeypatch, caplog):
        """Component 1 solves optimally, Component 2 times out with feasible coords."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11}

        call_idx = [0]
        import sys
        graph_mod = sys.modules['romutil.graph']

        def mock_solve(sub_rdb, sub_exits, timeout=None):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                # C1: Optimal
                m = MockModelCoords({1: (0, 0, 0), 2: (1, 0, 0)})
                return m, MockSolverResults(pyomo.opt.TerminationCondition.optimal)
            else:
                # C2: Timeout with feasible coords
                m = MockModelCoords({10: (0, 0, 0), 11: (2, 0, 0)})
                return m, MockSolverResults(pyomo.opt.TerminationCondition.maxTimeLimit)

        monkeypatch.setattr(graph_mod, "solve", mock_solve)

        with caplog.at_level(logging.INFO):
            solved_rdb, exits = solve_layout(rdb, "TestArea", solver_timeout=15)

        assert "Component 2 reached time limit; using best feasible layout" in caplog.text
        # Both components must have valid non-zero coordinates
        assert solved_rdb[1].x == 0
        assert solved_rdb[2].x == 1
        assert solved_rdb[10].x == 3  # Packed after C1 (width 1 + padding 2 = 3)
        assert solved_rdb[11].x == 5

    def test_component_solver_failure_fallback(self, monkeypatch, caplog):
        """Component failure falls back to (0, 0, 0) locally and packs next component cleanly."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11}

        call_idx = [0]
        import sys
        graph_mod = sys.modules['romutil.graph']

        def mock_solve(sub_rdb, sub_exits, timeout=None):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                # C1: Infeasible failure
                return None, MockSolverResults(pyomo.opt.TerminationCondition.infeasible)
            else:
                # C2: Optimal
                m = MockModelCoords({10: (0, 0, 0), 11: (3, 0, 0)})
                return m, MockSolverResults(pyomo.opt.TerminationCondition.optimal)

        monkeypatch.setattr(graph_mod, "solve", mock_solve)

        with caplog.at_level(logging.ERROR):
            solved_rdb, exits = solve_layout(rdb, "FailArea")

        assert "Component 1 solver failed!" in caplog.text
        # C1 rooms fall back to 0
        assert solved_rdb[1].x == 0
        assert solved_rdb[2].x == 0
        # C2 packed after C1 (offset = 0 + 2 = 2)
        assert solved_rdb[10].x == 2
        assert solved_rdb[11].x == 5


class TestGraphRenderingIntegration:
    """Integration test for graph() map rendering with decomposed components."""

    def test_graph_multi_component_svg_render(self, tmp_path):
        """graph() renders an SVG containing all decomposed components."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1),)))
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 10: r10, 11: r11}
        out_svg = tmp_path / "multi_comp.svg"

        graph(rdb, str(out_svg), "MultiCompArea", solver_timeout=10, component_padding=3)
        assert out_svg.exists()
        content = out_svg.read_text(encoding="utf-8")
        assert "<svg" in content

    def test_hallway_collapse_and_restore_in_components(self):
        """Straight corridor collapsed during decomposition is restored cleanly in component bounds."""
        # C1: Corridor 1 -> 2 -> 3 (East)
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1), ExitDef(direction=1, dst_vnum=3))))
        r3 = Room(RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=3, dst_vnum=2),)))
        # C2: 10 -> 11 (East)
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 3: r3, 10: r10, 11: r11}

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10, component_padding=3)

        # Room 2 must be restored in solved_rdb
        assert 2 in solved_rdb
        assert solved_rdb[2].x is not None
        assert solved_rdb[1].x < solved_rdb[2].x < solved_rdb[3].x

        # Component 2 must be packed beyond Component 1 (including restored room 2 and room 3)
        c1_max_x = max(solved_rdb[1].x, solved_rdb[2].x, solved_rdb[3].x)
        c2_min_x = min(solved_rdb[10].x, solved_rdb[11].x)
        assert c2_min_x - c1_max_x >= 3

    def test_hallway_collapse_fallback_on_solver_failure(self, monkeypatch):
        """When component solver fails, collapsed hallways in that component are restored with 0 coords."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=3, dst_vnum=1), ExitDef(direction=1, dst_vnum=3))))
        r3 = Room(RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=3, dst_vnum=2),)))
        r10 = Room(RoomDef(vnum=10, name="R10", description="", exits=(ExitDef(direction=1, dst_vnum=11),)))
        r11 = Room(RoomDef(vnum=11, name="R11", description="", exits=(ExitDef(direction=3, dst_vnum=10),)))

        rdb = {1: r1, 2: r2, 3: r3, 10: r10, 11: r11}

        import sys
        graph_mod = sys.modules["romutil.graph"]
        call_idx = [0]

        def mock_solve(sub_rdb, sub_exits, timeout=None):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                # C1 fails
                return None, MockSolverResults(pyomo.opt.TerminationCondition.infeasible)
            else:
                m = MockModelCoords({10: (0, 0, 0), 11: (1, 0, 0)})
                return m, MockSolverResults(pyomo.opt.TerminationCondition.optimal)

        monkeypatch.setattr(graph_mod, "solve", mock_solve)

        solved_rdb, exits = solve_layout(rdb, solver_timeout=10)
        assert 2 in solved_rdb
        assert solved_rdb[2].x == 1
        assert solved_rdb[2].y == 0
        assert solved_rdb[2].z == 0

    def test_graph_empty_rdb_early_exit(self):
        """graph() exits cleanly without rendering if rdb and exits are empty."""
        assert graph({}, "empty.svg", None) is None
