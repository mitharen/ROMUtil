from __future__ import annotations

import sys
import time
import pytest
from pyomo.environ import ConcreteModel, Set, Var, Binary, Integers

from romutil.models import Direction, Exit, ExitDef, Room, RoomDef
from romutil.solver import find_overlap_candidates, _get_coords


class TestSpatialIndexPositive:
    """Positive tests verifying sweep-line detection of true spatial collisions."""

    def test_perpendicular_crossing_detected(self):
        """Horizontal X-segment crosses vertical Y-segment at the origin."""
        # Exit 0: Room 1 (-5, 0, 0) -> Room 2 (5, 0, 0) [along X axis]
        # Exit 1: Room 3 (0, -5, 0) -> Room 4 (0, 5, 0) [along Y axis]
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (-5, 0, 0),
            2: (5, 0, 0),
            3: (0, -5, 0),
            4: (0, 5, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == [(0, 1)]

    def test_collinear_overlapping_segments(self):
        """Two parallel segments along X that overlap along their span."""
        # Exit 0: Room 1 (0, 2, 1) -> Room 2 (10, 2, 1)
        # Exit 1: Room 3 (5, 2, 1) -> Room 4 (15, 2, 1)
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (0, 2, 1),
            2: (10, 2, 1),
            3: (5, 2, 1),
            4: (15, 2, 1),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == [(0, 1)]

    def test_touching_endpoint_without_shared_room(self):
        """Two distinct non-incident exits that touch at identical coordinates."""
        # Exit 0: Room 1 (0, 0, 0) -> Room 2 (5, 0, 0)
        # Exit 1: Room 3 (5, 0, 0) -> Room 4 (10, 0, 0) (distinct rooms, coincident coords)
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (0, 0, 0),
            2: (5, 0, 0),
            3: (5, 0, 0),
            4: (10, 0, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == [(0, 1)]

    def test_room_objects_as_coordinates(self):
        """Verify coords can be provided as a dict of Room objects with x, y, z attributes."""
        r1 = Room(RoomDef(vnum=1, name='R1', description='', exits=()))
        r2 = Room(RoomDef(vnum=2, name='R2', description='', exits=()))
        r3 = Room(RoomDef(vnum=3, name='R3', description='', exits=()))
        r4 = Room(RoomDef(vnum=4, name='R4', description='', exits=()))
        r1.x, r1.y, r1.z = -3, 0, 0
        r2.x, r2.y, r2.z = 3, 0, 0
        r3.x, r3.y, r3.z = 0, -3, 0
        r4.x, r4.y, r4.z = 0, 3, 0

        rdb = {1: r1, 2: r2, 3: r3, 4: r4}
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        overlaps = find_overlap_candidates(exits, rdb)
        assert overlaps == [(0, 1)]

    def test_pyomo_model_as_coordinates(self):
        """Verify coords can be provided directly as a Pyomo ConcreteModel."""
        m = ConcreteModel()
        m.Rooms = Set(initialize=[1, 2, 3, 4])
        m.x = Var(m.Rooms, within=Integers)
        m.y = Var(m.Rooms, within=Integers)
        m.z = Var(m.Rooms, within=Integers)
        m.x[1].value, m.y[1].value, m.z[1].value = -2, 0, 0
        m.x[2].value, m.y[2].value, m.z[2].value = 2, 0, 0
        m.x[3].value, m.y[3].value, m.z[3].value = 0, -2, 0
        m.x[4].value, m.y[4].value, m.z[4].value = 0, 2, 0

        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        overlaps = find_overlap_candidates(exits, m)
        assert overlaps == [(0, 1)]


class TestSpatialIndexDisjoint:
    """Tests verifying disjoint exit segments are correctly pruned."""

    def test_disjoint_along_x_axis(self):
        """Segments strictly separated along X axis should never trigger AABB test."""
        # Exit 0: [0, 5]
        # Exit 1: [10, 15]
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (0, 0, 0),
            2: (5, 0, 0),
            3: (10, 0, 0),
            4: (15, 0, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []

    def test_overlap_in_x_but_disjoint_in_y(self):
        """Segments overlap in X interval [0, 10], but are separated along Y (0 vs 100)."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (0, 0, 0),
            2: (10, 0, 0),
            3: (2, 100, 0),
            4: (8, 100, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []

        # With aabb_3d=False, X-overlap should be detected
        overlaps_x_only = find_overlap_candidates(exits, coords, aabb_3d=False)
        assert overlaps_x_only == [(0, 1)]

    def test_overlap_in_x_and_y_but_disjoint_in_z(self):
        """Segments overlap in XY plane, but are on different elevation floors (Z=0 vs Z=3)."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (-5, 0, 0),
            2: (5, 0, 0),
            3: (0, -5, 3),
            4: (0, 5, 3),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []


class TestSpatialIndexVertical:
    """Tests verifying vertical exit handling (x_min == x_max)."""

    def test_vertical_exits_identical_xy_overlapping_z(self):
        """Two vertical shafts at exact same (X, Y) with overlapping vertical spans."""
        # Exit 0: (5, 5, 0) -> (5, 5, 4)
        # Exit 1: (5, 5, 2) -> (5, 5, 6)
        exits = [
            Exit(ExitDef(direction=Direction.up.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.up.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (5, 5, 0),
            2: (5, 5, 4),
            3: (5, 5, 2),
            4: (5, 5, 6),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == [(0, 1)]

    def test_vertical_exits_identical_xy_disjoint_z(self):
        """Two vertical shafts at exact same (X, Y) but disjoint Z elevation ranges."""
        # Exit 0: (5, 5, 0) -> (5, 5, 2)
        # Exit 1: (5, 5, 3) -> (5, 5, 5)
        exits = [
            Exit(ExitDef(direction=Direction.up.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.up.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (5, 5, 0),
            2: (5, 5, 2),
            3: (5, 5, 3),
            4: (5, 5, 5),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []

    def test_vertical_exit_crossing_horizontal_plane(self):
        """Vertical exit in Z crossing a horizontal exit in X at the same 3D point."""
        # Exit 0: (0, 0, -2) -> (0, 0, 2) [vertical]
        # Exit 1: (-2, 0, 0) -> (2, 0, 0) [horizontal]
        exits = [
            Exit(ExitDef(direction=Direction.up.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (0, 0, -2),
            2: (0, 0, 2),
            3: (-2, 0, 0),
            4: (2, 0, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == [(0, 1)]


class TestSpatialIndexIncidenceAndExclusions:
    """Tests verifying incidence, cut relaxation, and one-way exclusions."""

    def test_incident_exits_sharing_source_skipped(self):
        """Exits radiating from the same room are incident and should not be flagged."""
        # Exit 0: Room 1 -> Room 2
        # Exit 1: Room 1 -> Room 3
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=3), source=1),
        ]
        coords = {
            1: (0, 0, 0),
            2: (5, 0, 0),
            3: (0, 5, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []

    def test_incident_exits_sharing_destination_skipped(self):
        """Exits converging on the same room are incident and should not be flagged."""
        # Exit 0: Room 1 -> Room 3
        # Exit 1: Room 2 -> Room 3
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=3), source=1),
            Exit(ExitDef(direction=Direction.west.value, dst_vnum=3), source=2),
        ]
        coords = {
            1: (-5, 0, 0),
            2: (5, 0, 0),
            3: (0, 0, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []

    def test_one_way_exits_skipped(self):
        """One-way exits are excluded from collision constraints."""
        e0 = Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1)
        e1 = Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3)
        e1.one_way = True
        exits = [e0, e1]
        coords = {
            1: (-5, 0, 0),
            2: (5, 0, 0),
            3: (0, -5, 0),
            4: (0, 5, 0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == []

    def test_cut_exits_skipped(self):
        """Cuts relax geometric constraints and are skipped."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (-5, 0, 0),
            2: (5, 0, 0),
            3: (0, -5, 0),
            4: (0, 5, 0),
        }
        # Mark exit 0 as cut via boolean
        overlaps = find_overlap_candidates(exits, coords, cuts=[True, False])
        assert overlaps == []

        # Mark exit 0 as cut via Pyomo Var
        m = ConcreteModel()
        m.cuts = Var([0, 1], within=Binary)
        m.cuts[0].value = 1
        m.cuts[1].value = 0
        overlaps_pyomo = find_overlap_candidates(exits, coords, cuts=[m.cuts[0], m.cuts[1]])
        assert overlaps_pyomo == []

    def test_candidate_pairs_filter(self):
        """Only pairs in candidate_pairs should be returned."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (-5, 0, 0),
            2: (5, 0, 0),
            3: (0, -5, 0),
            4: (0, 5, 0),
        }
        # If (0, 1) is in candidate_pairs, it is returned
        assert find_overlap_candidates(exits, coords, candidate_pairs={(0, 1)}) == [(0, 1)]
        # If candidate_pairs is empty, nothing returned
        assert find_overlap_candidates(exits, coords, candidate_pairs=set()) == []


class TestSpatialIndexEdgeCasesAndNegative:
    """Negative and boundary tests for empty/missing data, None coords, and extreme values."""

    def test_empty_exits_list(self):
        assert find_overlap_candidates([], {1: (0, 0, 0)}) == []

    def test_none_or_empty_coords(self):
        exits = [Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1)]
        assert find_overlap_candidates(exits, None) == []
        assert find_overlap_candidates(exits, {}) == []

    def test_single_exit(self):
        exits = [Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1)]
        coords = {1: (0, 0, 0), 2: (1, 0, 0)}
        assert find_overlap_candidates(exits, coords) == []

    def test_unplaced_room_with_none_coordinates(self):
        """Rooms with None coordinates should be gracefully skipped without crashing."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        # Room 2 has None coordinates
        coords = {
            1: (0, 0, 0),
            2: None,
            3: (0, -5, 0),
            4: (0, 5, 0),
        }
        assert find_overlap_candidates(exits, coords) == []

    def test_room_with_partial_none_coordinates(self):
        """Coordinates with (x, None, z) should be treated as unplaced."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (0, None, 0),
            2: (5, 0, 0),
            3: (0, -5, 0),
            4: (0, 5, 0),
        }
        assert find_overlap_candidates(exits, coords) == []

    def test_negative_and_float_coordinates(self):
        """Ensure floating point and negative coordinates are sorted correctly."""
        exits = [
            Exit(ExitDef(direction=Direction.east.value, dst_vnum=2), source=1),
            Exit(ExitDef(direction=Direction.north.value, dst_vnum=4), source=3),
        ]
        coords = {
            1: (-10.5, -2.5, -1.0),
            2: (-0.5, -2.5, -1.0),
            3: (-5.5, -10.0, -1.0),
            4: (-5.5, 0.0, -1.0),
        }
        overlaps = find_overlap_candidates(exits, coords)
        assert overlaps == [(0, 1)]

    def test_helper_get_coords_various_types(self):
        """Unit test for _get_coords helper with unsupported or edge types."""
        assert _get_coords(None, 1) is None
        assert _get_coords({}, 1) is None
        assert _get_coords({1: 'not_a_coord'}, 1) is None
        assert _get_coords({1: (1, 2)}, 1) is None  # only 2 elements
        assert _get_coords({1: (1, 2, 3)}, 1) == (1.0, 2.0, 3.0)

        # Pyomo model with missing or None coordinates
        m = ConcreteModel()
        m.Rooms = Set(initialize=[1])
        m.x = Var(m.Rooms)
        m.y = Var(m.Rooms)
        m.z = Var(m.Rooms)
        assert _get_coords(m, 1) is None  # None values
        assert _get_coords(m, 999) is None  # KeyError

        # Sequence lookup raising IndexError / TypeError
        assert _get_coords([1, 2, 3], 999) is None

        # Room object with None coords
        r_none = Room(RoomDef(vnum=1, name='R', description='', exits=()))
        r_none.x = None
        assert _get_coords({1: r_none}, 1) is None


class TestSpatialIndexPerformance:
    """Performance benchmarks verifying sub-20ms sweep-line scaling."""

    def test_scaling_150_exits_under_20ms(self):
        """Benchmark candidate generation on a dense 150-exit grid matching Midgaard scale."""
        import random
        rng = random.Random(42)

        # Generate 150 exits across 100 rooms in a realistic multi-floor layout
        num_rooms = 100
        num_exits = 150
        coords = {
            v: (float(v % 10 * 4), float((v // 10) % 10 * 4), float(v % 3))
            for v in range(num_rooms)
        }

        exits = []
        for i in range(num_exits):
            src = rng.randint(0, num_rooms - 1)
            dst = rng.randint(0, num_rooms - 1)
            while dst == src:
                dst = rng.randint(0, num_rooms - 1)
            exits.append(Exit(ExitDef(direction=0, dst_vnum=dst), source=src))

        # Warm-up call with coverage instrumentation active
        overlaps = find_overlap_candidates(exits, coords)
        assert isinstance(overlaps, list)

        # Measure pure execution time without line-by-line tracing overhead
        old_trace = sys.gettrace()
        try:
            sys.settrace(None)
            t0 = time.perf_counter()
            iterations = 50
            for _ in range(iterations):
                find_overlap_candidates(exits, coords)
            elapsed_per_iter = (time.perf_counter() - t0) / iterations
        finally:
            sys.settrace(old_trace)

        # Must execute in well under 20ms (>90% reduction criterion)
        assert elapsed_per_iter < 0.020, f'Sweep-line took {elapsed_per_iter*1000:.2f}ms, expected < 20ms'

    def test_midgaard_candidate_generation_under_20ms(self):
        """Verify candidate overlap detection on midgaard.are is well under 20ms."""
        from tests.conftest import SAMPLE_AREAS_DIR
        from romutil.parser import parse_file
        from pathlib import Path

        if not SAMPLE_AREAS_DIR:
            pytest.skip('QuickMUD area directory not found')
        midgaard_path = Path(SAMPLE_AREAS_DIR) / 'midgaard.are'
        if not midgaard_path.is_file():
            pytest.skip('midgaard.are not found')

        area = parse_file(str(midgaard_path))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        exits = list(set([e for r in rdb.values() for e in r.exits if e.src != e.dst]))
        coords = {vnum: (float(vnum % 20 * 2), float((vnum // 20) % 20 * 2), float(vnum % 3)) for vnum in rdb}

        # Warm-up call with coverage instrumentation active
        overlaps = find_overlap_candidates(exits, coords)
        assert isinstance(overlaps, list)

        old_trace = sys.gettrace()
        try:
            sys.settrace(None)
            t0 = time.perf_counter()
            iterations = 20
            for _ in range(iterations):
                find_overlap_candidates(exits, coords)
            elapsed = (time.perf_counter() - t0) / iterations
        finally:
            sys.settrace(old_trace)

        assert elapsed < 0.020, f'Candidate generation on midgaard took {elapsed*1000:.2f}ms, expected < 20ms'
