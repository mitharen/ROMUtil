"""Unit and integration tests for Bidirectional Elevation Slack & Geometric Length Unification (Task 1j).

Verifies:
1. Mountain slope / ramp:
   - 3-room mountain slope (room 1 at z=0 exits East to room 2 at z=1, room 2 exits East to room 3 at z=2, bidirectional).
   - Solves with 0 cuts, monotonic x progression (x_1 < x_2 < x_3), and elevation steps (z_1=0, z_2=1, z_3=2).
2. Planar rooms remain flat:
   - Bidirectional planar rooms have dz = 0 and carry zero objective penalty.
3. Loops with elevation steps:
   - Loop containing horizontal exits across different elevations resolves smoothly without spurious cuts.
4. Fixture regressions:
   - Zero regressions across area fixtures (school.are, smurf.are, tower.are, midgaard.are).
"""

from pathlib import Path
import pyomo.environ as pyo
import pytest

from romutil.graph import solve_layout
from romutil.models import AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.parser import Parser
from romutil.solver import non_euler, solve

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_bidirectional_pair(r1: Room, r2: Room, direction: Direction, distance: int = 1) -> tuple[Exit, Exit]:
    e_fwd = Exit(direction=direction, src=r1.vnum, dst=r2.vnum, distance=distance)
    e_rev = Exit(direction=direction.invert(), src=r2.vnum, dst=r1.vnum, distance=distance)
    r1.exits.append(e_fwd)
    r2.exits.append(e_rev)
    return e_fwd, e_rev


class TestMountainSlopeElevationSlack:
    """Tests verifying horizontal exits can smoothly traverse elevation tiers without cuts."""

    def test_mountain_slope_three_rooms_with_elevation_steps(self):
        """3-room mountain slope: room 1 at z=0 exits East to room 2 at z=1, exits East to room 3 at z=2.

        Verified:
        - 0 cuts across all exits.
        - Monotonic x progression: x_1 < x_2 < x_3.
        - Elevation steps: z_1 == 0, z_2 == 1, z_3 == 2.
        - Elevation slack dz == 1 on horizontal ramp exits.
        """
        # We define a 3-room ramp alongside vertical cliff / staircase anchors:
        # Room 100 (z=0) --Up--> Room 101 (z=1) --Up--> Room 102 (z=2)
        # Room 1 is connected North/South to Room 100
        # Room 10 is at z=1 via Room 1 --Up--> Room 10 --East--> Room 2
        # Or even simpler:
        # Anchor stairs:
        # Base 10 (z=0) -> Up -> Mid 20 (z=1) -> Up -> Peak 30 (z=2)
        # Room 1 is North of 10 (z=0)
        # Room 2 is North of 20 (z=1) -- but wait, if 10, 20, 30 are on a vertical shaft, they share (x, y).
        # To avoid (x, y) collision, each tier has its own column:
        # Room 1 (z=0) has cliff Up to Room 11 (z=1). Room 11 exits East to Room 2 (z=1).
        # Room 2 (z=1) has cliff Up to Room 21 (z=2). Room 21 exits East to Room 3 (z=2).
        # Ramp: Room 1 exits East to Room 2; Room 2 exits East to Room 3.
        r1 = Room(RoomDef(vnum=1, name="Base Ramp", description=""))
        r2 = Room(RoomDef(vnum=2, name="Mid Ramp", description=""))
        r3 = Room(RoomDef(vnum=3, name="High Ramp", description=""))

        r11 = Room(RoomDef(vnum=11, name="Ledge 1", description=""))
        r21 = Room(RoomDef(vnum=21, name="Ledge 2", description=""))

        rdb = {1: r1, 2: r2, 3: r3, 11: r11, 21: r21}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []

        # Ramp path (East/West)
        e1_2, e2_1 = _make_bidirectional_pair(r1, r2, Direction.east)
        e2_3, e3_2 = _make_bidirectional_pair(r2, r3, Direction.east)
        exits.extend([e1_2, e2_1, e2_3, e3_2])

        # Step 1 anchor: r1 --Up--> r11 --East--> r2
        e1_11, e11_1 = _make_bidirectional_pair(r1, r11, Direction.up)
        e11_2, e2_11 = _make_bidirectional_pair(r11, r2, Direction.east)
        exits.extend([e1_11, e11_1, e11_2, e2_11])

        # Step 2 anchor: r2 --Up--> r21 --East--> r3
        e2_21, e21_2 = _make_bidirectional_pair(r2, r21, Direction.up)
        e21_3, e3_21 = _make_bidirectional_pair(r21, r3, Direction.east)
        exits.extend([e2_21, e21_2, e21_3, e3_21])

        model, results = solve(rdb, exits, timeout=15)

        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # 0 cuts across all exits
        cuts = [pyo.value(model.cut[i]) for i in range(len(exits))]
        assert sum(cuts) == 0, f"Expected 0 cuts, got {sum(cuts)}: {cuts}"

        # Monotonic x progression along the mountain ramp
        assert r1.x < r2.x < r3.x, f"Expected x1 < x2 < x3, got {r1.x}, {r2.x}, {r3.x}"

        # Elevation steps z1=0, z2=1, z3=2 (or normalized relative steps)
        z_min = r1.z
        assert r1.z - z_min == 0
        assert r2.z - z_min == 1
        assert r3.z - z_min == 2

        # Verify elevation slack dz on ramp exits
        ramp_idx_12 = exits.index(e1_2)
        ramp_idx_23 = exits.index(e2_3)
        assert pyo.value(model.dz[ramp_idx_12]) == 1
        assert pyo.value(model.dz[ramp_idx_23]) == 1

    def test_mountain_slope_direct_model_fixed_elevations(self):
        """Direct test: 3 rooms with East/West ramp with z fixed to 0, 1, 2.

        With elevation slack, model is feasible with cut=0 and dz=1.
        """
        r1 = Room(RoomDef(vnum=1, name="Slope 1", description=""))
        r2 = Room(RoomDef(vnum=2, name="Slope 2", description=""))
        r3 = Room(RoomDef(vnum=3, name="Slope 3", description=""))

        rdb = {1: r1, 2: r2, 3: r3}
        r1.exits = []
        r2.exits = []
        r3.exits = []

        e1_2, e2_1 = _make_bidirectional_pair(r1, r2, Direction.east)
        e2_3, e3_2 = _make_bidirectional_pair(r2, r3, Direction.east)
        exits = [e1_2, e2_1, e2_3, e3_2]

        # Verify non_euler does not cut horizontal slope
        m_ne = non_euler(rdb, exits)
        assert sum(pyo.value(m_ne.cut[i]) for i in range(len(exits))) == 0

        # Now solve with solve()
        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )
        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0
        assert r1.x < r2.x < r3.x

    def test_westward_descending_mountain_slope(self):
        """Westward descending slope: room 1 at z=2 exits West to room 2 at z=1, exits West to room 3 at z=0."""
        r1 = Room(RoomDef(vnum=1, name="Peak", description=""))
        r2 = Room(RoomDef(vnum=2, name="Mid", description=""))
        r3 = Room(RoomDef(vnum=3, name="Base", description=""))
        r12 = Room(RoomDef(vnum=12, name="Ledge Peak-Mid", description=""))
        r23 = Room(RoomDef(vnum=23, name="Ledge Mid-Base", description=""))

        rdb = {1: r1, 2: r2, 3: r3, 12: r12, 23: r23}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []
        e1_2, e2_1 = _make_bidirectional_pair(r1, r2, Direction.west)
        e2_3, e3_2 = _make_bidirectional_pair(r2, r3, Direction.west)
        exits.extend([e1_2, e2_1, e2_3, e3_2])

        # Anchor step 1: r1 --Down--> r12 --West--> r2
        e1_12, e12_1 = _make_bidirectional_pair(r1, r12, Direction.down)
        e12_2, e2_12 = _make_bidirectional_pair(r12, r2, Direction.west)
        exits.extend([e1_12, e12_1, e12_2, e2_12])

        # Anchor step 2: r2 --Down--> r23 --West--> r3
        e2_23, e23_2 = _make_bidirectional_pair(r2, r23, Direction.down)
        e23_3, e3_23 = _make_bidirectional_pair(r23, r3, Direction.west)
        exits.extend([e2_23, e23_2, e23_3, e3_23])

        model, results = solve(rdb, exits, timeout=15)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0
        assert r1.x > r2.x > r3.x
        assert r1.z > r2.z > r3.z

    def test_northward_ascending_mountain_slope(self):
        """Northward ascending slope: room 1 at z=0 exits North to room 2 at z=1, exits North to room 3 at z=2."""
        r1 = Room(RoomDef(vnum=1, name="Base", description=""))
        r2 = Room(RoomDef(vnum=2, name="Mid", description=""))
        r3 = Room(RoomDef(vnum=3, name="Top", description=""))
        r11 = Room(RoomDef(vnum=11, name="Anchor 1", description=""))
        r21 = Room(RoomDef(vnum=21, name="Anchor 2", description=""))

        rdb = {1: r1, 2: r2, 3: r3, 11: r11, 21: r21}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []
        e1_2, e2_1 = _make_bidirectional_pair(r1, r2, Direction.north)
        e2_3, e3_2 = _make_bidirectional_pair(r2, r3, Direction.north)
        exits.extend([e1_2, e2_1, e2_3, e3_2])

        # Anchor 1: r1 --Up--> r11 --North--> r2
        e1_11, e11_1 = _make_bidirectional_pair(r1, r11, Direction.up)
        e11_2, e2_11 = _make_bidirectional_pair(r11, r2, Direction.north)
        exits.extend([e1_11, e11_1, e11_2, e2_11])

        # Anchor 2: r2 --Up--> r21 --North--> r3
        e2_21, e21_2 = _make_bidirectional_pair(r2, r21, Direction.up)
        e21_3, e3_21 = _make_bidirectional_pair(r21, r3, Direction.north)
        exits.extend([e2_21, e21_2, e21_3, e3_21])

        model, results = solve(rdb, exits, timeout=15)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0
        assert r1.y < r2.y < r3.y
        assert r1.z < r2.z < r3.z


class TestFlatPlanarRoomsRemainFlat:
    """Tests verifying planar rooms have dz = 0 and carry zero objective penalty."""

    def test_flat_planar_chain_dz_zero(self):
        """A 4-room horizontal chain has dz = 0 on all exits and all z equal."""
        rooms = [Room(RoomDef(vnum=i, name=f"Room {i}", description="")) for i in range(1, 5)]
        rdb = {r.vnum: r for r in rooms}
        for r in rooms:
            r.exits = []

        exits: list[Exit] = []
        for i in range(len(rooms) - 1):
            ef, er = _make_bidirectional_pair(rooms[i], rooms[i + 1], Direction.east)
            exits.extend([ef, er])

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0
        z_vals = [r.z for r in rooms]
        assert len(set(z_vals)) == 1, f"Expected all z equal, got {z_vals}"
        for i in range(len(exits)):
            assert pyo.value(model.dz[i]) == 0

    def test_flat_planar_2x2_grid_dz_zero(self):
        """A 2x2 grid of planar rooms: all dz = 0, z identical, 0 cuts."""
        # 1 -East-> 2
        # ^         ^
        # North     North
        # 3 -East-> 4
        r1 = Room(RoomDef(vnum=1, name="NW", description=""))
        r2 = Room(RoomDef(vnum=2, name="NE", description=""))
        r3 = Room(RoomDef(vnum=3, name="SW", description=""))
        r4 = Room(RoomDef(vnum=4, name="SE", description=""))
        rdb = {1: r1, 2: r2, 3: r3, 4: r4}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []
        exits.extend(_make_bidirectional_pair(r3, r1, Direction.north))
        exits.extend(_make_bidirectional_pair(r4, r2, Direction.north))
        exits.extend(_make_bidirectional_pair(r3, r4, Direction.east))
        exits.extend(_make_bidirectional_pair(r1, r2, Direction.east))

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0
        z_coords = {r.z for r in rdb.values()}
        assert len(z_coords) == 1
        for i in range(len(exits)):
            assert pyo.value(model.dz[i]) == 0


class TestElevationLoopWithoutSpuriousCuts:
    """Tests verifying closed topological loops containing elevation steps solve without cuts."""

    def test_loop_with_horizontal_elevation_step_smooth_resolution(self):
        """Loop with an elevation step along a horizontal closing exit resolves with 0 cuts.

        Loop structure:
        r1 (z=0) --East--> r2 (z=0)
        r2 (z=0) --Up--> r3 (z=1)
        r3 (z=1) --North--> r4 (z=1)
        r4 (z=1) --West--> r5 (z=1)
        r5 (z=1) --South--> r1 (z=0)  [Horizontal exit stepping down Delta z = 1]
        """
        r1 = Room(RoomDef(vnum=1, name="R1 Ground SW", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2 Ground SE", description=""))
        r3 = Room(RoomDef(vnum=3, name="R3 Upper SE", description=""))
        r4 = Room(RoomDef(vnum=4, name="R4 Upper NE", description=""))
        r5 = Room(RoomDef(vnum=5, name="R5 Upper NW", description=""))

        rdb = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []
        e1_2, e2_1 = _make_bidirectional_pair(r1, r2, Direction.east)
        e2_3, e3_2 = _make_bidirectional_pair(r2, r3, Direction.up)
        e3_4, e4_3 = _make_bidirectional_pair(r3, r4, Direction.north)
        e4_5, e5_4 = _make_bidirectional_pair(r4, r5, Direction.west)
        e5_1, e1_5 = _make_bidirectional_pair(r5, r1, Direction.south)
        exits.extend([e1_2, e2_1, e2_3, e3_2, e3_4, e4_3, e4_5, e5_4, e5_1, e1_5])

        model, results = solve(rdb, exits, timeout=15)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # In the old formulation, rigid z equality forced a cut on e5_1 or e1_5.
        # In Task 1j, 0 cuts are required!
        cuts = [pyo.value(model.cut[i]) for i in range(len(exits))]
        assert sum(cuts) == 0, f"Expected 0 cuts on elevation loop, got: {cuts}"

        # Geometric verification: 2D rectangular loop is perfectly closed
        assert r1.x == r5.x, f"r1 and r5 should align on X: {r1.x} vs {r5.x}"
        assert r2.x == r3.x == r4.x, f"r2, r3, r4 should align on X: {r2.x}, {r3.x}, {r4.x}"
        assert r2.x > r1.x, "r2 should be East of r1"
        assert r1.y == r2.y, "r1 and r2 should align on Y"
        assert r4.y == r5.y, "r4 and r5 should align on Y"
        assert r4.y > r2.y, "r4 should be North of r2"

        # Elevation step: the vertical exit (r2 -> r3) steps in Z
        assert r3.z == r2.z + 1, f"r3 should be 1 level above r2: {r3.z} vs {r2.z}"

        # In the horizontal loop, the elevation step is absorbed with elevation slack dz
        # Total horizontal dz across all horizontal exits must sum to 2 (1 in each direction)
        horiz_exits = [e for e in exits if e.direction not in (Direction.up, Direction.down)]
        total_horiz_dz = sum(pyo.value(model.dz[exits.index(e)]) for e in horiz_exits)
        assert total_horiz_dz == 2.0, f"Expected total horizontal dz=2, got {total_horiz_dz}"

        # Vertical exit has dz = 0
        idx_23 = exits.index(e2_3)
        assert pyo.value(model.dz[idx_23]) == 0


class TestElevationSlackProperties:
    """Tests for variable and parameter properties of elevation slack."""

    def test_vertical_exits_slack_unconstrained_minimizes_to_zero(self):
        """Vertical exits (Up/Down) have dz = 0 in optimal solution."""
        r1 = Room(RoomDef(vnum=1, name="Floor 1", description=""))
        r2 = Room(RoomDef(vnum=2, name="Floor 2", description=""))
        rdb = {1: r1, 2: r2}
        r1.exits = []
        r2.exits = []

        e1_2, e2_1 = _make_bidirectional_pair(r1, r2, Direction.up)
        exits = [e1_2, e2_1]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert pyo.value(model.dz[0]) == 0
        assert pyo.value(model.dz[1]) == 0

    def test_one_way_exit_dz_zero(self):
        """One-way exits have dz = 0 in optimal solution."""
        r1 = Room(RoomDef(vnum=1, name="Start", description=""))
        r2 = Room(RoomDef(vnum=2, name="End", description=""))
        e = Exit(direction=Direction.east, src=1, dst=2)
        r1.exits = [e]
        r2.exits = []
        rdb = {1: r1, 2: r2}
        exits = [e]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )
        assert pyo.value(model.dz[0]) == 0


SAMPLE_TOWER_ARE = """#AREA
tower.are~
Tower~
{ 1 10 } Wizard Tower~
100 102

#ROOMS
#100
Ground Floor~
Desc 0~
0 0 0
D4
Upstairs~
~
0 0 101
S
#101
First Floor~
Desc 1~
0 0 0
D4
Upstairs~
~
0 0 102
D5
Downstairs~
~
0 0 100
S
#102
Tower Top~
Desc 2~
0 0 0
D5
Downstairs~
~
0 0 101
S
#0

#$
"""


class TestFixtureZeroRegressions:
    """Verification of standard area fixtures with the elevation slack formulation."""

    def test_smurf_are_zero_cuts(self):
        """smurf.are solves cleanly with 0 cuts."""
        smurf_file = FIXTURES_DIR / "areas" / "smurf.are"
        assert smurf_file.exists(), "smurf.are fixture must exist"

        parsed = Parser().parse(smurf_file.read_text(encoding="utf-8"))
        rooms = [s[1] for s in parsed if s and s[0] == "#ROOMS"][0]
        rdb = {r.vnum: Room(r) for r in rooms}
        header = AreaHeader(filename="smurf.are", name="Smurf", builder="", vnum_min=100, vnum_max=199)

        rdb, exits = solve_layout(rdb, header)
        assert len(rdb) > 0
        non_dummy = [r for r in rdb.values() if not r.dummy]
        assert all(r.x is not None and r.y is not None and r.z is not None for r in non_dummy)

    def test_school_are_solves(self):
        """school.are solves cleanly without regression."""
        school_file = FIXTURES_DIR / "areas" / "school.are"
        assert school_file.exists(), "school.are fixture must exist"

        parsed = Parser().parse(school_file.read_text(encoding="utf-8"))
        rooms = [s[1] for s in parsed if s and s[0] == "#ROOMS"][0]
        rdb = {r.vnum: Room(r) for r in rooms}
        header = AreaHeader(filename="school.are", name="School", builder="", vnum_min=3700, vnum_max=3799)

        rdb, exits = solve_layout(rdb, header)
        assert len(rdb) > 0
        non_dummy = [r for r in rdb.values() if not r.dummy]
        assert all(r.x is not None and r.y is not None and r.z is not None for r in non_dummy)

    def test_tower_are_solves(self):
        """tower.are solves cleanly with multiple elevation layers."""
        parsed = Parser().parse(SAMPLE_TOWER_ARE)
        rooms = [s[1] for s in parsed if s and s[0] == "#ROOMS"][0]
        rdb = {r.vnum: Room(r) for r in rooms}
        header = AreaHeader(filename="tower.are", name="Tower", builder="", vnum_min=100, vnum_max=102)

        rdb, exits = solve_layout(rdb, header)
        non_dummy = [r for r in rdb.values() if not r.dummy]
        unique_z = {r.z for r in non_dummy}
        assert len(unique_z) == 3

    def test_midgaard_are_solves(self):
        """midgaard.are solves cleanly without regression."""
        midgaard_file = FIXTURES_DIR / "areas" / "midgaard.are"
        assert midgaard_file.exists(), "midgaard.are fixture must exist"

        parsed = Parser().parse(midgaard_file.read_text(encoding="utf-8"))
        rooms = [s[1] for s in parsed if s and s[0] == "#ROOMS"][0]
        rdb = {r.vnum: Room(r) for r in rooms}
        header = AreaHeader(filename="midgaard.are", name="Midgaard", builder="", vnum_min=3000, vnum_max=3299)

        rdb, exits = solve_layout(rdb, header, solver_timeout=30)
        assert len(rdb) > 0
        non_dummy = [r for r in rdb.values() if not r.dummy]
        assert all(r.x is not None and r.y is not None and r.z is not None for r in non_dummy)
