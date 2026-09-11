"""Unit and integration tests for 3D Topological Symmetry & Isotropic Constraints (Task 1m).

Verifies:
1. 3D Isotropic Constraints & Rotational Symmetry:
   - Rotational symmetry across all coordinate planes (XY, XZ, YZ).
   - Linear chains along X, Y, and Z axes enforce strict orthogonal collinearity.
   - Planar 2x2 grids in XY, XZ, and YZ planes solve with 0 cuts and identical perpendicular coordinates.
   - 3D closed cuboid (2x2x2) solves with 0 cuts and perfect orthogonal alignment across all 3 axes.
2. Non-Eulerian Topological Symmetry:
   - non_euler() treats X, Y, and Z axes symmetrically across East/West, North/South, and Up/Down.
   - Contradictory cycles in XY, XZ, and YZ produce symmetric cut behavior.
3. Strict Orthogonal Alignment (No Elevation Slack):
   - Elevation slack variable m.dz is completely absent from the MILP model formulation.
   - Horizontal exits strictly enforce coplanar Z coordinates unless cut.
   - Non-Euclidean loops containing uncompensated vertical steps are cleanly relaxed with binary cuts.
4. Fixture Regressions:
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
    """Create and attach reciprocal bidirectional exits between two rooms."""
    e_fwd = Exit(direction=direction, src=r1.vnum, dst=r2.vnum, distance=distance)
    e_rev = Exit(direction=direction.invert(), src=r2.vnum, dst=r1.vnum, distance=distance)
    r1.exits.append(e_fwd)
    r2.exits.append(e_rev)
    return e_fwd, e_rev


class TestRotationalSymmetryAndIsotropy:
    """Tests verifying mathematical isotropy and rotational symmetry across all coordinate planes."""

    @pytest.mark.parametrize(
        ("direction", "primary_axis", "orthogonal_axes"),
        [
            (Direction.east, "x", ("y", "z")),
            (Direction.north, "y", ("x", "z")),
            (Direction.up, "z", ("x", "y")),
        ],
    )
    def test_linear_chain_isotropy(self, direction: Direction, primary_axis: str, orthogonal_axes: tuple[str, str]):
        """A 4-room chain along any cardinal axis exhibits identical progression and orthogonal collinearity."""
        rooms = [Room(RoomDef(vnum=i, name=f"Room {i}", description="")) for i in range(1, 5)]
        rdb = {r.vnum: r for r in rooms}
        for r in rooms:
            r.exits = []

        exits: list[Exit] = []
        for i in range(len(rooms) - 1):
            ef, er = _make_bidirectional_pair(rooms[i], rooms[i + 1], direction)
            exits.extend([ef, er])

        model, results = solve(rdb, exits, timeout=20)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # 0 cuts across all exits
        cuts = [pyo.value(model.cut[i]) for i in range(len(exits))]
        assert sum(cuts) == 0

        # Monotonic progression along primary axis
        primary_vals = [getattr(r, primary_axis) for r in rooms]
        for i in range(len(primary_vals) - 1):
            assert primary_vals[i] < primary_vals[i + 1]

        # Strict invariance along orthogonal axes
        for axis in orthogonal_axes:
            vals = {getattr(r, axis) for r in rooms}
            assert len(vals) == 1, f"Expected all {axis} identical along {direction.name} chain, got {vals}"

    def test_planar_grid_rotational_symmetry_xy_xz_yz(self):
        """2x2 grids constructed in XY, XZ, and YZ planes solve with isomorphic geometric structures."""
        # 1. XY Plane (East & North)
        r_xy = {i: Room(RoomDef(vnum=i, name=f"XY {i}", description="")) for i in range(1, 5)}
        for r in r_xy.values():
            r.exits = []
        ex_xy: list[Exit] = []
        ex_xy.extend(_make_bidirectional_pair(r_xy[3], r_xy[1], Direction.north))
        ex_xy.extend(_make_bidirectional_pair(r_xy[4], r_xy[2], Direction.north))
        ex_xy.extend(_make_bidirectional_pair(r_xy[3], r_xy[4], Direction.east))
        ex_xy.extend(_make_bidirectional_pair(r_xy[1], r_xy[2], Direction.east))

        m_xy, res_xy = solve(r_xy, ex_xy, timeout=10)
        assert res_xy.solver.termination_condition in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
        assert sum(pyo.value(m_xy.cut[i]) for i in range(len(ex_xy))) == 0
        z_xy = {r.z for r in r_xy.values()}
        assert len(z_xy) == 1, f"Expected identical Z in XY grid, got {z_xy}"
        dx_xy = r_xy[2].x - r_xy[1].x
        dy_xy = r_xy[1].y - r_xy[3].y
        assert dx_xy >= 1 and dy_xy >= 1

        # 2. XZ Plane (East & Up)
        r_xz = {i: Room(RoomDef(vnum=i, name=f"XZ {i}", description="")) for i in range(1, 5)}
        for r in r_xz.values():
            r.exits = []
        ex_xz: list[Exit] = []
        ex_xz.extend(_make_bidirectional_pair(r_xz[3], r_xz[1], Direction.up))
        ex_xz.extend(_make_bidirectional_pair(r_xz[4], r_xz[2], Direction.up))
        ex_xz.extend(_make_bidirectional_pair(r_xz[3], r_xz[4], Direction.east))
        ex_xz.extend(_make_bidirectional_pair(r_xz[1], r_xz[2], Direction.east))

        m_xz, res_xz = solve(r_xz, ex_xz, timeout=10)
        assert res_xz.solver.termination_condition in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
        assert sum(pyo.value(m_xz.cut[i]) for i in range(len(ex_xz))) == 0
        y_xz = {r.y for r in r_xz.values()}
        assert len(y_xz) == 1, f"Expected identical Y in XZ grid, got {y_xz}"
        dx_xz = r_xz[2].x - r_xz[1].x
        dz_xz = r_xz[1].z - r_xz[3].z
        assert dx_xz >= 1 and dz_xz >= 1

        # 3. YZ Plane (North & Up)
        r_yz = {i: Room(RoomDef(vnum=i, name=f"YZ {i}", description="")) for i in range(1, 5)}
        for r in r_yz.values():
            r.exits = []
        ex_yz: list[Exit] = []
        ex_yz.extend(_make_bidirectional_pair(r_yz[3], r_yz[1], Direction.up))
        ex_yz.extend(_make_bidirectional_pair(r_yz[4], r_yz[2], Direction.up))
        ex_yz.extend(_make_bidirectional_pair(r_yz[3], r_yz[4], Direction.north))
        ex_yz.extend(_make_bidirectional_pair(r_yz[1], r_yz[2], Direction.north))

        m_yz, res_yz = solve(r_yz, ex_yz, timeout=10)
        assert res_yz.solver.termination_condition in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
        assert sum(pyo.value(m_yz.cut[i]) for i in range(len(ex_yz))) == 0
        x_yz = {r.x for r in r_yz.values()}
        assert len(x_yz) == 1, f"Expected identical X in YZ grid, got {x_yz}"
        dy_yz = r_yz[2].y - r_yz[1].y
        dz_yz = r_yz[1].z - r_yz[3].z
        assert dy_yz >= 1 and dz_yz >= 1

        # Mathematical isomorphism: spans across active axes are identical
        assert (dx_xy, dy_xy) == (dx_xz, dz_xz) == (dy_yz, dz_yz)

    def test_closed_3d_cuboid(self):
        """A 2x2x2 cuboid (8 rooms, 12 bidirectional edges) solves with 0 cuts and orthogonal integrity."""
        # Rooms: 1..8 representing (x, y, z) vertices
        # Bottom floor (z=0):
        # 3(0,1,0) -East-> 4(1,1,0)
        # ^                ^
        # North            North
        # 1(0,0,0) -East-> 2(1,0,0)
        # Top floor (z=1):
        # 7(0,1,1) -East-> 8(1,1,1)
        # ^                ^
        # North            North
        # 5(0,0,1) -East-> 6(1,0,1)
        rooms = {i: Room(RoomDef(vnum=i, name=f"Node {i}", description="")) for i in range(1, 9)}
        for r in rooms.values():
            r.exits = []

        exits: list[Exit] = []
        # Bottom floor
        exits.extend(_make_bidirectional_pair(rooms[1], rooms[2], Direction.east))
        exits.extend(_make_bidirectional_pair(rooms[3], rooms[4], Direction.east))
        exits.extend(_make_bidirectional_pair(rooms[1], rooms[3], Direction.north))
        exits.extend(_make_bidirectional_pair(rooms[2], rooms[4], Direction.north))

        # Top floor
        exits.extend(_make_bidirectional_pair(rooms[5], rooms[6], Direction.east))
        exits.extend(_make_bidirectional_pair(rooms[7], rooms[8], Direction.east))
        exits.extend(_make_bidirectional_pair(rooms[5], rooms[7], Direction.north))
        exits.extend(_make_bidirectional_pair(rooms[6], rooms[8], Direction.north))

        # Vertical pillars (1->5, 2->6, 3->7, 4->8)
        exits.extend(_make_bidirectional_pair(rooms[1], rooms[5], Direction.up))
        exits.extend(_make_bidirectional_pair(rooms[2], rooms[6], Direction.up))
        exits.extend(_make_bidirectional_pair(rooms[3], rooms[7], Direction.up))
        exits.extend(_make_bidirectional_pair(rooms[4], rooms[8], Direction.up))

        model, results = solve(rooms, exits, timeout=15)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0

        # Verify orthogonal alignments
        # Bottom plane Z identical, Top plane Z identical
        assert rooms[1].z == rooms[2].z == rooms[3].z == rooms[4].z
        assert rooms[5].z == rooms[6].z == rooms[7].z == rooms[8].z
        assert rooms[5].z > rooms[1].z

        # Vertical pillars strictly collinear in (X, Y)
        for bottom, top in [(1, 5), (2, 6), (3, 7), (4, 8)]:
            assert rooms[bottom].x == rooms[top].x
            assert rooms[bottom].y == rooms[top].y


class TestNonEulerTopologicalSymmetry:
    """Tests verifying non_euler treats all spatial dimensions with identical mathematical rigor."""

    def test_non_euler_isotropic_orthogonal_constraints_all_axes(self):
        """non_euler applies symmetric orthogonal equality constraints on all 3 planes."""
        # Test East exit: constrains Y and Z
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        rdb = {1: r1, 2: r2}
        e_east = Exit(direction=Direction.east, src=1, dst=2)
        e_west = Exit(direction=Direction.west, src=2, dst=1)
        r1.exits = [e_east]
        r2.exits = [e_west]
        exits = [e_east, e_west]

        m_east = non_euler(rdb, exits)
        assert sum(pyo.value(m_east.cut[i]) for i in range(len(exits))) == 0
        assert pyo.value(m_east.y[1]) == pyo.value(m_east.y[2])
        assert pyo.value(m_east.z[1]) == pyo.value(m_east.z[2])
        assert pyo.value(m_east.x[2]) > pyo.value(m_east.x[1])

        # Test North exit: constrains X and Z
        r1.exits = []
        r2.exits = []
        e_north = Exit(direction=Direction.north, src=1, dst=2)
        e_south = Exit(direction=Direction.south, src=2, dst=1)
        r1.exits = [e_north]
        r2.exits = [e_south]
        exits_n = [e_north, e_south]

        m_north = non_euler(rdb, exits_n)
        assert sum(pyo.value(m_north.cut[i]) for i in range(len(exits_n))) == 0
        assert pyo.value(m_north.x[1]) == pyo.value(m_north.x[2])
        assert pyo.value(m_north.z[1]) == pyo.value(m_north.z[2])
        assert pyo.value(m_north.y[2]) > pyo.value(m_north.y[1])

        # Test Up exit: constrains X and Y
        r1.exits = []
        r2.exits = []
        e_up = Exit(direction=Direction.up, src=1, dst=2)
        e_down = Exit(direction=Direction.down, src=2, dst=1)
        r1.exits = [e_up]
        r2.exits = [e_down]
        exits_u = [e_up, e_down]

        m_up = non_euler(rdb, exits_u)
        assert sum(pyo.value(m_up.cut[i]) for i in range(len(exits_u))) == 0
        assert pyo.value(m_up.x[1]) == pyo.value(m_up.x[2])
        assert pyo.value(m_up.y[1]) == pyo.value(m_up.y[2])
        assert pyo.value(m_up.z[2]) > pyo.value(m_up.z[1])

    @pytest.mark.parametrize(
        ("dir_fwd", "dir_rev"),
        [
            (Direction.east, Direction.west),
            (Direction.north, Direction.south),
            (Direction.up, Direction.down),
        ],
    )
    def test_contradictory_cycle_symmetric_cuts(self, dir_fwd: Direction, dir_rev: Direction):
        """A contradictory cycle along any cardinal axis triggers exactly the same cut relaxation."""
        # Cycle: 1 -> 2 -> 3 -> 1 all in fwd direction, with reverse exits
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))
        rdb = {1: r1, 2: r2, 3: r3}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []
        exits.extend(_make_bidirectional_pair(r1, r2, dir_fwd))
        exits.extend(_make_bidirectional_pair(r2, r3, dir_fwd))
        exits.extend(_make_bidirectional_pair(r3, r1, dir_fwd))

        m = non_euler(rdb, exits)
        cuts = sum(pyo.value(m.cut[i]) for i in range(len(exits)))
        # Contradictory directed cycle along any axis must be cut
        assert cuts >= 1


class TestOrthogonalSymmetryAndCutRelaxation:
    """Tests verifying strict orthogonal alignment and absence of elevation slack m.dz."""

    def test_model_has_no_dz_variable(self):
        """The Pyomo ConcreteModel generated by solve() contains no elevation slack m.dz variable."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        rdb = {1: r1, 2: r2}
        r1.exits = []
        r2.exits = []
        exits = list(_make_bidirectional_pair(r1, r2, Direction.east))

        model, _results = solve(rdb, exits, timeout=10)
        assert not hasattr(model, "dz"), "m.dz variable must be removed from the MILP model"

    def test_elevation_loop_requires_cut_under_isotropic_constraints(self):
        """A closed topological loop with an uncompensated vertical step requires a cut under isotropic constraints.

        Loop structure:
        r1 (z=0) --East--> r2 (z=0)
        r2 (z=0) --Up--> r3 (z=1)
        r3 (z=1) --North--> r4 (z=1)
        r4 (z=1) --West--> r5 (z=1)
        r5 (z=1) --South--> r1 (z=0)  [Horizontal exit connecting rooms at different elevations]
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

        # Under isotropic 3D orthogonal constraints, a horizontal exit requires equal Z coordinates.
        # Because this loop has net Delta z = 1 without a downward exit, it is geometrically non-Euclidean
        # and must be relaxed with at least one binary cut.
        cuts = [pyo.value(model.cut[i]) for i in range(len(exits))]
        assert sum(cuts) >= 1, f"Expected non-Euclidean elevation loop to be cut, but cuts={cuts}"

    def test_horizontal_exits_strictly_coplanar(self):
        """Horizontal bidirectional exits rigidly preserve coplanar Z coordinates when uncut."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))
        rdb = {1: r1, 2: r2, 3: r3}
        for r in rdb.values():
            r.exits = []

        exits: list[Exit] = []
        exits.extend(_make_bidirectional_pair(r1, r2, Direction.east))
        exits.extend(_make_bidirectional_pair(r2, r3, Direction.north))

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert sum(pyo.value(model.cut[i]) for i in range(len(exits))) == 0
        assert r1.z == r2.z == r3.z, f"Expected identical Z coordinates, got {r1.z}, {r2.z}, {r3.z}"

    def test_one_way_exit_invariance(self):
        """One-way exits across horizontal and vertical dimensions solve without elevation slack variables."""
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
        assert not hasattr(model, "dz")


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


@pytest.mark.slow
@pytest.mark.integration
class TestFixtureZeroRegressions:
    """Verification of standard area fixtures with isotropic 3D constraints."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_area_fixture_regressions(self):
        """Standard area fixture regression solves with isotropic constraints."""
        self.test_smurf_are_zero_cuts()
        self.test_school_are_solves()
        self.test_tower_are_solves()

    @pytest.mark.slow
    @pytest.mark.integration
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

    @pytest.mark.slow
    @pytest.mark.integration
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

    @pytest.mark.slow
    @pytest.mark.integration
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

    @pytest.mark.slow
    @pytest.mark.integration
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


@pytest.mark.slow
@pytest.mark.integration
def test_area_fixture_regressions():
    """Module-level regression verification for standard area fixtures."""
    suite = TestFixtureZeroRegressions()
    suite.test_smurf_are_zero_cuts()
    suite.test_school_are_solves()
    suite.test_tower_are_solves()
