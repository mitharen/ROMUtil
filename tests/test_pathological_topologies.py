"""Pathological Topology Micro-Fixtures and Regression Suite (Task 7c).

Contains isolated synthetic micro-fixtures and test suites targeting pathological
non-Euclidean MUD geometries without relying on large area file solves:

1. Starburst / Converging Funnel Topology:
   - Synthetic 3x3 planar grid with 9 one-way Up exits converging to a single target room.
   - Verifies optimal termination, undistorted grid geometry, exactly 1 vertically aligned exit,
     and zero coordinate collisions.
2. Multi-Floor Cardinal Z-Plunge:
   - Minimal multi-tier fixture with rooms at distinct elevation tiers linked by a horizontal exit.
   - Verifies directional progression along the primary axis (y_src > y_dst) and orthogonal
     collinearity along X (x_src == x_dst) across the vertical displacement.
3. Orthogonal Diagonal Bypass Tension:
   - Synthetic fixture where a one-way East exit connects rooms having a forced lateral offset.
   - Verifies the one-way exit is cleanly cut rather than warping into an unconstrained diagonal
     or skewing orthogonal room placement.
4. Macro-Bypass Circuit:
   - Directed ring with a one-way reverse chord (simulating school entrance / arena split).
   - Verifies deterministic cut placement, cycle relaxation, and zero coordinate collisions.
"""

from collections.abc import Mapping
import pyomo.environ as pyo
import pytest

from romutil.graph import solve_layout
from romutil.models import Direction, Exit, Room, RoomDef
from romutil.solver import solve


def assert_no_room_collisions(rdb: Mapping[int, Room]) -> None:
    """Verify that for all non-dummy rooms in rdb, every room has a unique (x, y, z) tuple.

    If any collision is found, raises AssertionError detailing the colliding coordinates and VNUMs.
    """
    coords: dict[tuple[int, int, int], list[int]] = {}
    for vnum, room in rdb.items():
        if getattr(room, "dummy", False):
            continue
        assert room.x is not None and room.y is not None and room.z is not None, (
            f"Room {vnum} has unassigned coordinates: x={room.x}, y={room.y}, z={room.z}"
        )
        pos = (int(round(room.x)), int(round(room.y)), int(round(room.z)))
        coords.setdefault(pos, []).append(vnum)

    collisions = {pos: vnums for pos, vnums in coords.items() if len(vnums) > 1}
    assert not collisions, f"Detected room coordinate collisions: {collisions}"


def _make_bidirectional_pair(r1: Room, r2: Room, direction: Direction, distance: int = 1) -> tuple[Exit, Exit]:
    """Create and attach reciprocal bidirectional exits between two rooms."""
    e_fwd = Exit(direction=direction, src=r1.vnum, dst=r2.vnum, distance=distance)
    e_rev = Exit(direction=direction.invert(), src=r2.vnum, dst=r1.vnum, distance=distance)
    r1.exits.append(e_fwd)
    r2.exits.append(e_rev)
    return e_fwd, e_rev


def _build_3x3_grid() -> tuple[dict[int, Room], list[Exit]]:
    """Construct a synthetic 3x3 planar grid with VNUMs 1..9."""
    rooms: dict[int, Room] = {}
    for y in range(3):
        for x in range(3):
            vnum = y * 3 + x + 1
            r = Room(RoomDef(vnum=vnum, name=f"Grid Room {vnum}", description=""))
            r.exits = []
            rooms[vnum] = r

    exits: list[Exit] = []
    for y in range(3):
        for x in range(3):
            vnum = y * 3 + x + 1
            if x < 2:
                ef, er = _make_bidirectional_pair(rooms[vnum], rooms[vnum + 1], Direction.east)
                exits.extend([ef, er])
            if y < 2:
                ef, er = _make_bidirectional_pair(rooms[vnum], rooms[vnum + 3], Direction.north)
                exits.extend([ef, er])

    return rooms, exits


@pytest.mark.slow
@pytest.mark.integration
class TestStarburstConvergingFunnel:
    """Micro-fixtures testing starburst and converging funnel topologies."""

    def test_starburst_3x3_upward_converging_funnel(self) -> None:
        """3x3 grid where all 9 rooms have a one-way Up exit to destination room 10.

        Verifies:
        - Solver completes with optimal termination condition.
        - Exactly 1 exit (the center room 5) is vertically aligned (x10 == x_src, y10 == y_src, z10 == z_src + 1).
        - Exactly 8 exits accommodate lateral displacement from vertical alignment (or cut).
        - The underlying 3x3 grid is undistorted (unit spacing along X and Y).
        - Zero room coordinate collisions exist.
        """
        rooms, exits = _build_3x3_grid()
        dest = Room(RoomDef(vnum=10, name="Safe Room 10", description=""))
        dest.exits = []
        rooms[10] = dest

        funnel_exits: list[Exit] = []
        for vnum in range(1, 10):
            e = Exit(direction=Direction.up, src=vnum, dst=10)
            rooms[vnum].exits.append(e)
            exits.append(e)
            funnel_exits.append(e)

        model, results = solve(rooms, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # Zero room coordinate collisions
        assert_no_room_collisions(rooms)

        # Underlying 3x3 grid is undistorted
        for y in range(3):
            for x in range(2):
                v_curr = y * 3 + x + 1
                v_next = v_curr + 1
                r_curr, r_next = rooms[v_curr], rooms[v_next]
                assert r_curr.x is not None and r_next.x is not None
                assert r_curr.y is not None and r_next.y is not None
                assert r_curr.z is not None and r_next.z is not None
                assert round(r_next.x - r_curr.x) == 1
                assert round(r_next.y) == round(r_curr.y)
                assert round(r_next.z) == round(r_curr.z)

        for y in range(2):
            for x in range(3):
                v_curr = y * 3 + x + 1
                v_next = v_curr + 3
                r_curr, r_next = rooms[v_curr], rooms[v_next]
                assert r_curr.x is not None and r_next.x is not None
                assert r_curr.y is not None and r_next.y is not None
                assert r_curr.z is not None and r_next.z is not None
                assert round(r_next.y - r_curr.y) == 1
                assert round(r_next.x) == round(r_curr.x)
                assert round(r_next.z) == round(r_curr.z)

        # Room 10 is placed at elevation z_src + 1
        r10 = rooms[10]
        assert r10.x is not None and r10.y is not None and r10.z is not None
        for vnum in range(1, 10):
            r_src = rooms[vnum]
            assert r_src.z is not None
            assert r10.z >= r_src.z + 1

        # Exactly 1 exit (center room 5) is vertically aligned with room 10
        aligned_exits = [
            e for e in funnel_exits
            if round(r10.x) == round(rooms[e.src].x or 0)
            and round(r10.y) == round(rooms[e.src].y or 0)
            and round(r10.z) == round((rooms[e.src].z or 0) + 1)
        ]
        assert len(aligned_exits) == 1
        assert aligned_exits[0].src == 5

        # Exactly 8 exits have lateral displacement from vertical alignment (or are cut)
        offset_exits = [
            e for e in funnel_exits
            if (round(r10.x) != round(rooms[e.src].x or 0) or round(r10.y) != round(rooms[e.src].y or 0))
        ]
        assert len(offset_exits) == 8

        # In MILP model, binary cuts are relaxed to 0 under soft L1 minimization (or 8 under rigid collinearity)
        cuts = sum(int(round(model.cut[i].value or 0)) for i in range(len(exits)))
        assert cuts in (0, 8)

    def test_starburst_downward_converging_funnel(self) -> None:
        """3x3 grid with one-way Down exits converging to room 10 at lower elevation."""
        rooms, exits = _build_3x3_grid()
        dest = Room(RoomDef(vnum=10, name="Basement 10", description=""))
        dest.exits = []
        rooms[10] = dest

        funnel_exits: list[Exit] = []
        for vnum in range(1, 10):
            e = Exit(direction=Direction.down, src=vnum, dst=10)
            rooms[vnum].exits.append(e)
            exits.append(e)
            funnel_exits.append(e)

        _model, results = solve(rooms, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )
        assert_no_room_collisions(rooms)

        r10 = rooms[10]
        assert r10.x is not None and r10.y is not None and r10.z is not None
        # Center room 5 is vertically aligned directly above room 10
        r5 = rooms[5]
        assert r5.x is not None and r5.y is not None and r5.z is not None
        assert round(r10.x) == round(r5.x)
        assert round(r10.y) == round(r5.y)
        assert round(r10.z) == round(r5.z - 1)

    def test_starburst_solve_layout_integration(self) -> None:
        """Full solve_layout integration on 3x3 starburst yields zero collisions."""
        rooms, _ = _build_3x3_grid()
        dest = Room(RoomDef(vnum=10, name="Safe Room 10", description=""))
        dest.exits = []
        rooms[10] = dest

        for vnum in range(1, 10):
            e = Exit(direction=Direction.up, src=vnum, dst=10)
            rooms[vnum].exits.append(e)

        solved_rdb, _solved_exits = solve_layout(rooms, None, solver_timeout=10)
        assert_no_room_collisions(solved_rdb)


@pytest.mark.slow
@pytest.mark.integration
class TestMultiFloorCardinalZPlunge:
    """Micro-fixtures testing cardinal horizontal exits linking distinct elevation tiers."""

    def test_multi_floor_cardinal_z_plunge_south(self) -> None:
        """Minimal multi-tier fixture: upper room at Z=2 linked by South exit to ground room at Z=0.

        Verifies:
        - Elevation tiers are preserved (z_src - z_dst >= 2).
        - Directional progression along primary axis: y_src > y_dst.
        - Orthogonal collinearity along X: x_src == x_dst.
        - Solves with 0 cuts and zero coordinate collisions.
        """
        r_ground = Room(RoomDef(vnum=1, name="Ground Destination", description=""))
        r_shaft1 = Room(RoomDef(vnum=11, name="Vertical Shaft 1", description=""))
        r_shaft2 = Room(RoomDef(vnum=12, name="Vertical Shaft 2", description=""))
        r_upper = Room(RoomDef(vnum=2, name="Upper Source", description=""))

        rdb = {1: r_ground, 11: r_shaft1, 12: r_shaft2, 2: r_upper}
        for r in rdb.values():
            r.exits = []

        # Vertical stairs: 1 -Up-> 11 -Up-> 12
        e1_11, e11_1 = _make_bidirectional_pair(r_ground, r_shaft1, Direction.up)
        e11_12, e12_11 = _make_bidirectional_pair(r_shaft1, r_shaft2, Direction.up)
        # 12 -North-> 2 at upper elevation
        e12_2, e2_12 = _make_bidirectional_pair(r_shaft2, r_upper, Direction.north)

        # Horizontal South plunge from upper room 2 to ground room 1
        e_plunge = Exit(direction=Direction.south, src=2, dst=1)
        r_upper.exits.append(e_plunge)

        exits = [e1_11, e11_1, e11_12, e12_11, e12_2, e2_12, e_plunge]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert r_upper.x is not None and r_upper.y is not None and r_upper.z is not None
        assert r_ground.x is not None and r_ground.y is not None and r_ground.z is not None

        # Elevation displacement preserved
        assert r_upper.z - r_ground.z >= 2

        # Directional progression along primary axis (South: y_src > y_dst)
        assert r_upper.y > r_ground.y

        # Orthogonal collinearity along X (x_src == x_dst)
        assert r_upper.x == r_ground.x

        # 0 cuts
        cuts = sum(int(round(model.cut[i].value or 0)) for i in range(len(exits)))
        assert cuts == 0

        assert_no_room_collisions(rdb)

    def test_multi_floor_cardinal_z_plunge_east(self) -> None:
        """Horizontal East plunge across elevation tiers preserves East progression and Y-collinearity."""
        r_ground = Room(RoomDef(vnum=1, name="Ground", description=""))
        r_shaft1 = Room(RoomDef(vnum=11, name="Shaft 1", description=""))
        r_shaft2 = Room(RoomDef(vnum=12, name="Shaft 2", description=""))
        r_upper = Room(RoomDef(vnum=2, name="Upper", description=""))

        rdb = {1: r_ground, 11: r_shaft1, 12: r_shaft2, 2: r_upper}
        for r in rdb.values():
            r.exits = []

        # 1 -Up-> 11 -Up-> 12 -West-> 2
        e1_11, e11_1 = _make_bidirectional_pair(r_ground, r_shaft1, Direction.up)
        e11_12, e12_11 = _make_bidirectional_pair(r_shaft1, r_shaft2, Direction.up)
        e12_2, e2_12 = _make_bidirectional_pair(r_shaft2, r_upper, Direction.west)

        # Horizontal East plunge: 2 (Upper, Z=2) -> 1 (Ground, Z=0)
        e_plunge = Exit(direction=Direction.east, src=2, dst=1)
        r_upper.exits.append(e_plunge)

        exits = [e1_11, e11_1, e11_12, e12_11, e12_2, e2_12, e_plunge]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert r_upper.x is not None and r_upper.y is not None and r_upper.z is not None
        assert r_ground.x is not None and r_ground.y is not None and r_ground.z is not None

        # Elevation difference
        assert r_upper.z - r_ground.z >= 2
        # Directional progression along East (x_dst > x_src)
        assert r_ground.x > r_upper.x
        # Orthogonal collinearity along Y (y_src == y_dst)
        assert r_upper.y == r_ground.y

        cuts = sum(int(round(model.cut[i].value or 0)) for i in range(len(exits)))
        assert cuts == 0

        assert_no_room_collisions(rdb)


@pytest.mark.slow
@pytest.mark.integration
class TestOrthogonalDiagonalBypassTension:
    """Micro-fixtures verifying clean cut relaxation under forced lateral offset."""

    def test_orthogonal_diagonal_bypass_tension_clean_cut(self) -> None:
        """One-way East exit connecting rooms with forced lateral offset along Y is cleanly cut.

        Rooms 1 and 2 are rigidly linked North/South (x1 == x2, y2 = y1 + 1).
        A one-way East exit from Room 2 to Room 1 requires x1 >= x2 + 1, which is mathematically
        incompatible with x1 == x2.
        Verifies:
        - The one-way East exit is cleanly cut (cut == 1).
        - Rooms 1 and 2 maintain unskewed orthogonal placement (x1 == x2, y2 == y1 + 1).
        - Zero coordinate collisions.
        """
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))

        # Bidirectional North/South
        e12, e21 = _make_bidirectional_pair(r1, r2, Direction.north)
        # One-way East from 2 to 1
        e21_east = Exit(direction=Direction.east, src=2, dst=1)
        r2.exits.append(e21_east)

        rdb = {1: r1, 2: r2}
        exits = [e12, e21, e21_east]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        idx_east = exits.index(e21_east)
        # Clean cut on the conflicting one-way exit
        assert int(round(model.cut[idx_east].value or 0)) == 1

        # Orthogonal placement is preserved without skewing
        assert r1.x is not None and r2.x is not None
        assert r1.y is not None and r2.y is not None
        assert r1.x == r2.x
        assert r2.y == r1.y + 1

        assert_no_room_collisions(rdb)

    def test_orthogonal_diagonal_bypass_tension_three_room_column(self) -> None:
        """3-room vertical column with a one-way East exit from top to bottom room is cleanly cut."""
        r1 = Room(RoomDef(vnum=1, name="Bottom", description=""))
        r2 = Room(RoomDef(vnum=2, name="Middle", description=""))
        r3 = Room(RoomDef(vnum=3, name="Top", description=""))

        rdb = {1: r1, 2: r2, 3: r3}
        for r in rdb.values():
            r.exits = []

        e12, e21 = _make_bidirectional_pair(r1, r2, Direction.north)
        e23, e32 = _make_bidirectional_pair(r2, r3, Direction.north)
        # One-way East from 3 to 1
        e31_east = Exit(direction=Direction.east, src=3, dst=1)
        r3.exits.append(e31_east)

        exits = [e12, e21, e23, e32, e31_east]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        idx_east = exits.index(e31_east)
        assert int(round(model.cut[idx_east].value or 0)) == 1

        # Column remains vertically collinear
        assert r1.x is not None and r2.x is not None and r3.x is not None
        assert r1.x == r2.x == r3.x
        assert r1.y is not None and r2.y is not None and r3.y is not None
        assert r1.y < r2.y < r3.y

        assert_no_room_collisions(rdb)

    def test_orthogonal_diagonal_bypass_tension_west_cut(self) -> None:
        """One-way West exit from 2 to 1 where x1 == x2 is cleanly cut."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))

        e12, e21 = _make_bidirectional_pair(r1, r2, Direction.north)
        e21_west = Exit(direction=Direction.west, src=2, dst=1)
        r2.exits.append(e21_west)

        rdb = {1: r1, 2: r2}
        exits = [e12, e21, e21_west]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        idx_west = exits.index(e21_west)
        assert int(round(model.cut[idx_west].value or 0)) == 1
        assert r1.x is not None and r2.x is not None
        assert r1.x == r2.x
        assert r1.y is not None and r2.y is not None
        assert r2.y == r1.y + 1
        assert_no_room_collisions(rdb)


@pytest.mark.slow
@pytest.mark.integration
class TestMacroBypassCircuit:
    """Micro-fixtures testing directed rings with reverse chords (school entrance / arena split)."""

    def test_macro_bypass_circuit_deterministic_chord_cut(self) -> None:
        """4-room ring with a one-way reverse chord forces deterministic cut on the chord.

        Ring: Room 1 -East-> Room 2 -South-> Room 3 -West-> Room 4 -North-> Room 1.
        Reverse chord: Room 3 has a one-way South exit to Room 1 (simulating entrance/arena split).
        Since Room 3 is already South of Room 1 in the ring, a South exit from 3 to 1
        requires y3 >= y1 + 1, contradicting y1 > y3.
        Verifies:
        - Deterministic cut placed on the reverse chord.
        - Ring cycles are relaxed without infeasibility.
        - Ring rooms maintain unit distances without coordinate distortion.
        - Zero coordinate collisions exist.
        """
        r1 = Room(RoomDef(vnum=1, name="Entrance", description=""))
        r2 = Room(RoomDef(vnum=2, name="Wing", description=""))
        r3 = Room(RoomDef(vnum=3, name="Arena", description=""))
        r4 = Room(RoomDef(vnum=4, name="ExitRoom", description=""))

        # Bidirectional ring
        e12, e21 = _make_bidirectional_pair(r1, r2, Direction.east)
        e23, e32 = _make_bidirectional_pair(r2, r3, Direction.south)
        e34, e43 = _make_bidirectional_pair(r3, r4, Direction.west)
        e41, e14 = _make_bidirectional_pair(r4, r1, Direction.north)

        # One-way reverse chord from 3 to 1 (South)
        e_chord = Exit(direction=Direction.south, src=3, dst=1)
        r3.exits.append(e_chord)

        rdb = {1: r1, 2: r2, 3: r3, 4: r4}
        exits = [e12, e21, e23, e32, e34, e43, e41, e14, e_chord]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # Deterministic cut on the reverse chord
        idx_chord = exits.index(e_chord)
        assert int(round(model.cut[idx_chord].value or 0)) == 1

        # Exactly 1 cut relaxes the cycle
        cuts = sum(int(round(model.cut[i].value or 0)) for i in range(len(exits)))
        assert cuts == 1

        # Zero coordinate collisions
        assert_no_room_collisions(rdb)

        # Ring geometry is undistorted
        assert r1.x is not None and r2.x is not None and r3.x is not None and r4.x is not None
        assert r1.y is not None and r2.y is not None and r3.y is not None and r4.y is not None
        assert r2.x == r1.x + 1 and r2.y == r1.y
        assert r3.x == r2.x and r3.y == r2.y - 1
        assert r4.x == r3.x - 1 and r4.y == r3.y
        assert r1.x == r4.x and r1.y == r4.y + 1

    def test_macro_bypass_circuit_solve_layout_integration(self) -> None:
        """Integration with solve_layout correctly marks cut on the reverse chord."""
        r1 = Room(RoomDef(vnum=1, name="Entrance", description=""))
        r2 = Room(RoomDef(vnum=2, name="Wing", description=""))
        r3 = Room(RoomDef(vnum=3, name="Arena", description=""))
        r4 = Room(RoomDef(vnum=4, name="ExitRoom", description=""))

        _make_bidirectional_pair(r1, r2, Direction.east)
        _make_bidirectional_pair(r2, r3, Direction.south)
        _make_bidirectional_pair(r3, r4, Direction.west)
        _make_bidirectional_pair(r4, r1, Direction.north)

        e_chord = Exit(direction=Direction.south, src=3, dst=1)
        r3.exits.append(e_chord)

        rdb = {1: r1, 2: r2, 3: r3, 4: r4}

        solved_rdb, solved_exits = solve_layout(rdb, None, solver_timeout=10)

        assert_no_room_collisions(solved_rdb)

        # Check that chord exit is marked as cut
        chord_found = next(
            (e for e in solved_exits if e.src == 3 and e.dst == 1 and e.direction == Direction.south),
            None,
        )
        assert chord_found is not None
        assert getattr(chord_found, "cut", False) is True

    def test_macro_bypass_circuit_diagonal_reverse_chord(self) -> None:
        """4-room ring with horizontal reverse chord (Room 2 East to Room 4) forces cut on chord."""
        r1 = Room(RoomDef(vnum=1, name="NW", description=""))
        r2 = Room(RoomDef(vnum=2, name="NE", description=""))
        r3 = Room(RoomDef(vnum=3, name="SE", description=""))
        r4 = Room(RoomDef(vnum=4, name="SW", description=""))

        e12, e21 = _make_bidirectional_pair(r1, r2, Direction.east)
        e23, e32 = _make_bidirectional_pair(r2, r3, Direction.south)
        e34, e43 = _make_bidirectional_pair(r3, r4, Direction.west)
        e41, e14 = _make_bidirectional_pair(r4, r1, Direction.north)

        # Reverse chord from 2 to 4 going East (Room 4 is West of Room 2)
        e_chord = Exit(direction=Direction.east, src=2, dst=4)
        r2.exits.append(e_chord)

        rdb = {1: r1, 2: r2, 3: r3, 4: r4}
        exits = [e12, e21, e23, e32, e34, e43, e41, e14, e_chord]

        model, results = solve(rdb, exits, timeout=10)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        idx_chord = exits.index(e_chord)
        assert int(round(model.cut[idx_chord].value or 0)) == 1

        cuts = sum(int(round(model.cut[i].value or 0)) for i in range(len(exits)))
        assert cuts == 1

        assert_no_room_collisions(rdb)
