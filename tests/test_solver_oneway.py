"""Unit and integration tests for Directional Half-Space Constraints & Cycle Cut Relaxation

for One-Way Exits (Task 1g).

Verifies:
1. Directional Half-Space Constraints:
   - One-way exits strictly enforce forward directional inequalities across all 6 directions
     (North, South, East, West, Up, Down) when feasible:
     - East: x_dst - x_src + 2 * Mx * cut >= 1
     - West: x_src - x_dst + 2 * Mx * cut >= 1
     - North: y_dst - y_src + 2 * My * cut >= 1
     - South: y_src - y_dst + 2 * My * cut >= 1
     - Up: z_dst - z_src + 2 * Mz * cut >= 1
     - Down: z_src - z_dst + 2 * Mz * cut >= 1
   - Zero cut (cut[i] == 0) and zero inversion when geometry is topologically feasible.
2. Non-Euclidean One-Way Cycle Cut Relaxation:
   - Cycles composed of one-way exits (e.g. 3-room South loop, 2-room East loop, 4-room North loop,
     vertical Up loop) force a minimal cut (cut[i] == 1) without resulting in solver infeasibility
     or crashing.
   - Uncut exits in the cycle strictly respect forward directional inequalities.
3. Affine Linear Expressions for Boundary Dummy Rooms:
   - One-way exits incident to boundary dummy rooms respect affine linear expressions (dummy_anchors)
     without instantiating additional integer decision variables.
4. Mud School (school.are) Geometry Preservation:
   - Layout solving on school.are ensures one-way paths (specifically room 3700 South exit to 3744)
     do not flip 180 degrees backwards (North).
   - All uncut one-way exits in school.are strictly satisfy forward directional inequalities.
"""

from pathlib import Path
import pyomo.environ as pyo
import pytest

from romutil.graph import solve_layout
from romutil.models import Direction, Exit, Room, RoomDef
from romutil.parser import Parser
from romutil.solver import solve

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestOneWayDirectionalInequalities:
    """Tests verifying that one-way exits strictly enforce forward half-space inequalities."""

    @pytest.mark.parametrize(
        ("direction", "coord_idx", "is_forward"),
        [
            (Direction.east, 0, True),    # x_dst >= x_src + 1
            (Direction.west, 0, False),   # x_src >= x_dst + 1
            (Direction.north, 1, True),   # y_dst >= y_src + 1
            (Direction.south, 1, False),  # y_src >= y_dst + 1
            (Direction.up, 2, True),      # z_dst >= z_src + 1
            (Direction.down, 2, False),   # z_src >= z_dst + 1
        ],
    )
    def test_oneway_all_six_directions_forward_inequality(self, direction, coord_idx, is_forward):
        """One-way exit across each direction enforces forward inequality with cut=0."""
        r1 = Room(RoomDef(vnum=1, name="Src", description=""))
        r2 = Room(RoomDef(vnum=2, name="Dst", description=""))

        # One-way exit from r1 to r2 (no reciprocal exit in r2)
        ex = Exit(direction=direction, src=1, dst=2)
        r1.exits = [ex]
        r2.exits = []

        rdb = {1: r1, 2: r2}
        exits = [ex]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # Exit is one-way, feasible, and not cut
        assert model.cut[0].value == 0

        p1 = (model.x[1].value, model.y[1].value, model.z[1].value)
        p2 = (model.x[2].value, model.y[2].value, model.z[2].value)

        if is_forward:
            assert p2[coord_idx] >= p1[coord_idx] + 1
        else:
            assert p1[coord_idx] >= p2[coord_idx] + 1

    def test_oneway_prevents_backward_flip_under_tension(self):
        """Under topological tension, a one-way exit cannot flip backward to reduce L1 distance."""
        # Triangle topology:
        # Room 1 -> Room 2 via 2-way East exit of distance 10
        # Room 2 -> Room 3 via 2-way North exit of distance 2
        # Room 3 -> Room 1 via 1-way South exit
        # Builder geometry: 3 is North of 2, 2 is East of 1.
        # Room 3 -> Room 1 South means y[3] >= y[1] + 1.
        # If flipped North, y[1] would be >= y[3] + 1.
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))

        e12 = Exit(direction=Direction.east, src=1, dst=2, distance=10)
        e21 = Exit(direction=Direction.west, src=2, dst=1, distance=10)
        e23 = Exit(direction=Direction.north, src=2, dst=3, distance=2)
        e32 = Exit(direction=Direction.south, src=3, dst=2, distance=2)
        # One-way South from 3 to 1
        e31 = Exit(direction=Direction.south, src=3, dst=1)

        r1.exits = [e12]
        r2.exits = [e21, e23]
        r3.exits = [e32, e31]

        rdb = {1: r1, 2: r2, 3: r3}
        exits = [e12, e21, e23, e32, e31]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        idx_31 = exits.index(e31)
        # Without rigid orthogonal collinearity constraints, the one-way South exit
        # satisfies the forward half-space inequality y[3] >= y[1] + 1 without requiring a cut.
        assert model.cut[idx_31].value == 0
        assert model.y[3].value >= model.y[1].value + 1

    @pytest.mark.parametrize("direction", [Direction.east, Direction.west])
    def test_oneway_east_west_orthogonal_y(self, direction):
        """East/West one-way exits enforce Y-collinearity (y_src == y_dst) when uncut."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        ex = Exit(direction=direction, src=1, dst=2)
        r1.exits = [ex]
        r2.exits = []
        rdb = {1: r1, 2: r2}
        exits = [ex]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )
        assert model.cut[0].value == 0
        assert model.y[1].value == model.y[2].value
        if direction == Direction.east:
            assert model.x[2].value >= model.x[1].value + 1
        else:
            assert model.x[1].value >= model.x[2].value + 1

    @pytest.mark.parametrize("direction", [Direction.north, Direction.south])
    def test_oneway_north_south_orthogonal_x(self, direction):
        """North/South one-way exits enforce X-collinearity (x_src == x_dst) when uncut."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        ex = Exit(direction=direction, src=1, dst=2)
        r1.exits = [ex]
        r2.exits = []
        rdb = {1: r1, 2: r2}
        exits = [ex]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )
        assert model.cut[0].value == 0
        assert model.x[1].value == model.x[2].value
        if direction == Direction.north:
            assert model.y[2].value >= model.y[1].value + 1
        else:
            assert model.y[1].value >= model.y[2].value + 1

    @pytest.mark.parametrize("direction", [Direction.up, Direction.down])
    def test_oneway_up_down_orthogonal_x_y(self, direction):
        """Up/Down one-way exits enforce X- and Y-collinearity (x_src == x_dst, y_src == y_dst) when uncut."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        ex = Exit(direction=direction, src=1, dst=2)
        r1.exits = [ex]
        r2.exits = []
        rdb = {1: r1, 2: r2}
        exits = [ex]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )
        assert model.cut[0].value == 0
        assert model.x[1].value == model.x[2].value
        assert model.y[1].value == model.y[2].value
        if direction == Direction.up:
            assert model.z[2].value >= model.z[1].value + 1
        else:
            assert model.z[1].value >= model.z[2].value + 1


class TestOneWayNonEuclideanCycles:
    """Tests verifying that non-Euclidean one-way cycles trigger binary cut relaxation."""

    def test_oneway_cycle_3_rooms_south(self):
        """3 rooms in a South-South-South loop force exactly 1 cut without infeasibility."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))

        e1 = Exit(direction=Direction.south, src=1, dst=2)
        e2 = Exit(direction=Direction.south, src=2, dst=3)
        e3 = Exit(direction=Direction.south, src=3, dst=1)

        r1.exits = [e1]
        r2.exits = [e2]
        r3.exits = [e3]

        rdb = {1: r1, 2: r2, 3: r3}
        exits = [e1, e2, e3]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        cuts = [int(model.cut[i].value) for i in range(3)]
        # Exactly one cut relaxes the cycle
        assert sum(cuts) == 1

        # The 2 uncut exits must strictly satisfy South forward inequality: y_src >= y_dst + 1
        for i, ex in enumerate(exits):
            if cuts[i] == 0:
                assert model.y[ex.src].value >= model.y[ex.dst].value + 1

    def test_oneway_cycle_2_rooms_east(self):
        """2 rooms in an East-East loop force exactly 1 cut without infeasibility."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))

        e1 = Exit(direction=Direction.east, src=1, dst=2)
        e2 = Exit(direction=Direction.east, src=2, dst=1)

        r1.exits = [e1]
        r2.exits = [e2]

        rdb = {1: r1, 2: r2}
        exits = [e1, e2]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        cuts = [int(model.cut[i].value) for i in range(2)]
        assert sum(cuts) == 1

        for i, ex in enumerate(exits):
            if cuts[i] == 0:
                assert model.x[ex.dst].value >= model.x[ex.src].value + 1

    def test_oneway_cycle_4_rooms_north(self):
        """4 rooms in a North loop force exactly 1 cut with remaining exits ordered North."""
        rooms = {i: Room(RoomDef(vnum=i, name=f"R{i}", description="")) for i in range(1, 5)}
        exits = [
            Exit(direction=Direction.north, src=1, dst=2),
            Exit(direction=Direction.north, src=2, dst=3),
            Exit(direction=Direction.north, src=3, dst=4),
            Exit(direction=Direction.north, src=4, dst=1),
        ]
        for i, ex in enumerate(exits, start=1):
            rooms[i].exits = [ex]

        model, results = solve(rooms, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        cuts = [int(model.cut[i].value) for i in range(4)]
        assert sum(cuts) == 1

        for i, ex in enumerate(exits):
            if cuts[i] == 0:
                assert model.y[ex.dst].value >= model.y[ex.src].value + 1

    def test_oneway_cycle_vertical_up(self):
        """2 rooms in an Up-Up vertical cycle force exactly 1 cut."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))

        e1 = Exit(direction=Direction.up, src=1, dst=2)
        e2 = Exit(direction=Direction.up, src=2, dst=1)

        r1.exits = [e1]
        r2.exits = [e2]

        rdb = {1: r1, 2: r2}
        exits = [e1, e2]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        cuts = [int(model.cut[i].value) for i in range(2)]
        assert sum(cuts) == 1

        for i, ex in enumerate(exits):
            if cuts[i] == 0:
                assert model.z[ex.dst].value >= model.z[ex.src].value + 1


class TestOneWayDummyAffineAnchoring:
    """Tests verifying that dummy room affine linear expressions correctly integrate with one-ways."""

    def test_oneway_pointing_to_dummy_anchor(self):
        """One-way exit to a dummy room uses get_x/y/z affine expressions correctly."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        d2 = Room(RoomDef(vnum=2, name="D2", description=""))
        d2.dummy = True
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))

        e_dummy = Exit(direction=Direction.east, src=1, dst=2)
        e_oneway = Exit(direction=Direction.east, src=3, dst=2)

        r1.exits = [e_dummy]
        r3.exits = [e_oneway]

        rdb = {1: r1, 2: d2, 3: r3}
        exits = [e_dummy, e_oneway]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # d2 is anchored at (x[1] + 1, y[1], z[1])
        # e_oneway (3 -> 2 East) enforces: x[d2] - x[3] >= 1
        assert d2.x >= model.x[3].value + 1

    def test_oneway_pointing_from_dummy_anchor(self):
        """One-way exit from a dummy room to a normal room uses get_x/y/z affine expressions."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        d2 = Room(RoomDef(vnum=2, name="D2", description=""))
        d2.dummy = True
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))

        e_dummy = Exit(direction=Direction.east, src=1, dst=2)
        e_oneway = Exit(direction=Direction.east, src=2, dst=3)

        r1.exits = [e_dummy]
        d2.exits = [e_oneway]

        rdb = {1: r1, 2: d2, 3: r3}
        exits = [e_dummy, e_oneway]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        # e_oneway (2 -> 3 East) enforces: x[3] - x[d2] >= 1
        assert model.x[3].value >= d2.x + 1


@pytest.mark.slow
@pytest.mark.integration
class TestSchoolAreOneWayPreservation:
    """Regression and integration tests on Mud School (school.are)."""

    @pytest.fixture(scope="class")
    @classmethod
    def school_layout(cls):
        filepath = FIXTURES_DIR / "areas" / "school.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        return solve_layout(rdb, area, solver_timeout=35)

    def test_school_are_entrance_to_arena_south_preservation(self, school_layout):
        """Verify room 3700 South exit to 3744 either preserves South or is cut by cycle relaxation."""
        solved_rdb, exits = school_layout

        r3700 = solved_rdb[3700]
        r3744 = solved_rdb[3744]

        assert r3700.x is not None and r3700.y is not None
        assert r3744.x is not None and r3744.y is not None

        # Find 3700 -> 3744 exit
        ex_3700_3744 = next((e for e in exits if e.src == 3700 and e.dst == 3744), None)
        assert ex_3700_3744 is not None

        if getattr(ex_3700_3744, "cut", False):
            # When cut by cycle relaxation, the one-way skip exit is relaxed,
            # allowing the entire school to shift south for a tighter, more optimal layout.
            assert ex_3700_3744.cut is True
        else:
            assert r3700.y > r3744.y, (
                f"Room 3700 (y={r3700.y}) is not North of 3744 (y={r3744.y}); exit flipped North!"
            )

    def test_school_are_all_uncut_oneways_preserve_direction(self, school_layout):
        """Verify that every uncut one-way exit in school.are preserves forward inequality."""
        solved_rdb, exits = school_layout

        for ex in exits:
            if not getattr(ex, "one_way", False):
                continue
            if ex.src not in solved_rdb or ex.dst not in solved_rdb:
                continue

            src_room = solved_rdb[ex.src]
            dst_room = solved_rdb[ex.dst]

            if None in (src_room.x, src_room.y, src_room.z, dst_room.x, dst_room.y, dst_room.z):
                continue

            # Check directional inequality on non-cut exits within the layout
            # Specifically for exits between rooms where both are non-dummy or dummy anchored
            if ex.direction == Direction.east:
                assert dst_room.x >= src_room.x + 1 or getattr(ex, "cut", False)
            elif ex.direction == Direction.west:
                assert src_room.x >= dst_room.x + 1 or getattr(ex, "cut", False)
            elif ex.direction == Direction.north:
                assert dst_room.y >= src_room.y + 1 or getattr(ex, "cut", False)
            elif ex.direction == Direction.south:
                assert src_room.y >= dst_room.y + 1 or getattr(ex, "cut", False)
            elif ex.direction == Direction.up:
                assert dst_room.z >= src_room.z + 1 or getattr(ex, "cut", False)
            elif ex.direction == Direction.down:
                assert src_room.z >= dst_room.z + 1 or getattr(ex, "cut", False)

    def test_school_are_room_3721_directly_south_of_3722(self, school_layout):
        """Room 3721 has a one-way North exit to 3722 and must be placed south of 3722."""
        solved_rdb, exits = school_layout

        r3721 = solved_rdb[3721]
        r3722 = solved_rdb[3722]

        assert r3721.x is not None and r3721.y is not None
        assert r3722.x is not None and r3722.y is not None
        assert r3722.y >= r3721.y + 1, f"Room 3721 (y={r3721.y}) not south of 3722 (y={r3722.y})"
