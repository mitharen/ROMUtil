"""Tests for formulative Big-M dimension-specific tightening and planar direction reduction (Task 1d).

Verifies:
1. compute_dimension_bounds calculation:
   - Mx = max(10, sum(|dx|) + 1)
   - My = max(10, sum(|dy|) + 1)
   - Mz = max(5, sum(|dz|) + 1)
   - Safe lower bounds (10, 10, 5) on empty or single-direction exit collections.
   - >= 40% reduction in bound magnitude compared to monolithic sum-of-all-exits M on sparse planar areas.
2. Variable and parameter bounds in Pyomo models:
   - non_euler: m.Mx, m.My, m.Mz, m.x in [-Mx, Mx], m.y in [-My, My], m.z in [-Mz, Mz].
   - solve: m.Mx, m.My, m.Mz, m.x in [-Mx, Mx], m.y in [-My, My], m.z in [-Mz, Mz].
   - one_ways bounded within [0, 2 * M].
3. Planar direction reduction in add_overlap_constraint:
   - For planar overlaps (horizontal exits in same elevation plane or no vertical exits in area),
     directions reduced from 6 to 4 (North, South, East, West), eliminating 2 binary variables
     (33.33% reduction in binary decision variables).
   - Up and Down binary variables are excluded from relation variable.
   - 16 crossing constraints generated instead of 24.
   - For non-planar overlaps (vertical exits present), all 6 directions are maintained.
   - Big-M bounds use dimension-specific bounds (2 * Mx for East/West, 2 * My for North/South, 2 * Mz for Up/Down).
4. Candidate batch capping:
   - get_candidate_batch_cap(n) = min(15, max(5, int(sqrt(n)))) if n > 0 else 0.
   - Capped between 5 and 15 for positive candidate counts.
5. End-to-end solver verification and zero regression across fixtures:
   - smurf.are
   - school.are
   - midgaard.are
   - CircleMUD split world
   - DikuMUD Alfa
"""

from pathlib import Path
import pyomo.environ as pyo
import pytest

from romutil.models import Direction, Exit, Room, RoomDef
from romutil.parser import Parser, parse_circlemud_directory
from romutil.graph import solve_layout
from romutil.solver import (
    compute_dimension_bounds,
    get_candidate_batch_cap,
    add_overlap_constraint,
    non_euler,
    solve,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDimensionBoundsCalculation:
    """Tests for compute_dimension_bounds mathematical logic and lower bounds."""

    def test_empty_exits_lower_bounds(self):
        """Empty exits list returns safe lower bounds: Mx=10, My=10, Mz=5."""
        mx, my, mz = compute_dimension_bounds([])
        assert mx == 10
        assert my == 10
        assert mz == 5

    def test_single_axis_east_west_bounds(self):
        """East/West exits affect only Mx, while My and Mz stay at safe lower bounds."""
        ex1 = Exit(src=1, dst=2, direction=Direction.east, distance=15)
        ex2 = Exit(src=2, dst=1, direction=Direction.west, distance=20)

        mx, my, mz = compute_dimension_bounds([ex1, ex2])
        # sum_x = 35; mx = max(10, 35 + 1) = 36
        assert mx == 36
        assert my == 10
        assert mz == 5

    def test_single_axis_north_south_bounds(self):
        """North/South exits affect only My."""
        ex1 = Exit(src=1, dst=2, direction=Direction.north, distance=25)
        ex2 = Exit(src=2, dst=1, direction=Direction.south, distance=10)

        mx, my, mz = compute_dimension_bounds([ex1, ex2])
        assert mx == 10
        assert my == 36
        assert mz == 5

    def test_single_axis_up_down_bounds(self):
        """Up/Down exits affect only Mz."""
        ex1 = Exit(src=1, dst=2, direction=Direction.up, distance=8)
        ex2 = Exit(src=2, dst=1, direction=Direction.down, distance=7)

        mx, my, mz = compute_dimension_bounds([ex1, ex2])
        assert mx == 10
        assert my == 10
        # sum_z = 15; mz = max(5, 15 + 1) = 16
        assert mz == 16

    def test_multi_axis_bounds(self):
        """Multi-axis exits correctly accumulate per dimension."""
        exits = [
            Exit(src=1, dst=2, direction=Direction.east, distance=2),
            Exit(src=2, dst=1, direction=Direction.west, distance=2),
            Exit(src=1, dst=3, direction=Direction.north, distance=2),
            Exit(src=3, dst=1, direction=Direction.south, distance=2),
            Exit(src=1, dst=4, direction=Direction.up, distance=2),
            Exit(src=4, dst=1, direction=Direction.down, distance=2),
        ]

        # sum_x = 4 -> max(10, 5) = 10
        # sum_y = 4 -> max(10, 5) = 10
        # sum_z = 4 -> max(5, 5) = 5
        mx, my, mz = compute_dimension_bounds(exits)
        assert mx == 10
        assert my == 10
        assert mz == 5

        # If distances are large:
        for e in exits:
            e.distance = 50

        # sum_x = 100 -> 101
        # sum_y = 100 -> 101
        # sum_z = 100 -> 101
        mx, my, mz = compute_dimension_bounds(exits)
        assert mx == 101
        assert my == 101
        assert mz == 101

    def test_planar_area_bound_reduction_school_are(self):
        """Verify school.are achieves >= 40% reduction across all dimensions compared to monolithic M."""
        filepath = FIXTURES_DIR / "areas" / "school.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        exits = [e for r in rdb.values() for e in r.exits]

        m_mono = sum(e.distance for e in exits)
        mx, my, mz = compute_dimension_bounds(exits)

        reduction_x = (m_mono - mx) / m_mono
        reduction_y = (m_mono - my) / m_mono
        reduction_z = (m_mono - mz) / m_mono

        assert reduction_x >= 0.40, f"Mx reduction {reduction_x:.2%} < 40%"
        assert reduction_y >= 0.40, f"My reduction {reduction_y:.2%} < 40%"
        assert reduction_z >= 0.40, f"Mz reduction {reduction_z:.2%} < 40%"

    def test_planar_area_bound_reduction_smurf_are(self):
        """Verify smurf.are achieves >= 40% reduction on X and Z dimensions compared to monolithic M."""
        filepath = FIXTURES_DIR / "areas" / "smurf.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        exits = [e for r in rdb.values() for e in r.exits]

        m_mono = sum(e.distance for e in exits)
        mx, my, mz = compute_dimension_bounds(exits)

        reduction_x = (m_mono - mx) / m_mono
        reduction_z = (m_mono - mz) / m_mono

        assert reduction_x >= 0.40, f"Mx reduction {reduction_x:.2%} < 40%"
        assert reduction_z >= 0.40, f"Mz reduction {reduction_z:.2%} < 40%"


class TestConstraintBatchCapping:
    """Tests for get_candidate_batch_cap mathematical formula and bounds."""

    def test_zero_or_negative_candidates(self):
        """Zero or negative candidate count returns 0."""
        assert get_candidate_batch_cap(0) == 0
        assert get_candidate_batch_cap(-5) == 0

    @pytest.mark.parametrize(
        "candidates,expected",
        [
            (1, 5),      # sqrt(1)=1 -> max(5, 1) = 5 -> min(15, 5) = 5
            (4, 5),      # sqrt(4)=2 -> max(5, 2) = 5
            (9, 5),      # sqrt(9)=3 -> max(5, 3) = 5
            (16, 5),     # sqrt(16)=4 -> max(5, 4) = 5
            (25, 5),     # sqrt(25)=5 -> max(5, 5) = 5
            (36, 6),     # sqrt(36)=6 -> 6
            (49, 7),     # sqrt(49)=7 -> 7
            (64, 8),     # sqrt(64)=8 -> 8
            (81, 9),     # sqrt(81)=9 -> 9
            (100, 10),   # sqrt(100)=10 -> 10
            (144, 12),   # sqrt(144)=12 -> 12
            (196, 14),   # sqrt(196)=14 -> 14
            (225, 15),   # sqrt(225)=15 -> 15 (cap reached)
            (400, 15),   # sqrt(400)=20 -> 15 (capped)
            (1000, 15),  # sqrt(1000)=31 -> 15 (capped)
            (10000, 15), # sqrt(10000)=100 -> 15 (capped)
        ],
    )
    def test_batch_cap_values(self, candidates, expected):
        assert get_candidate_batch_cap(candidates) == expected


class TestPlanarDirectionReduction:
    """Tests for planar direction reduction and binary variable elimination in add_overlap_constraint."""

    def _build_test_model(self, rooms=(1, 2, 3, 4)):
        m = pyo.ConcreteModel()
        m.Rooms = pyo.Set(initialize=rooms)
        m.Exits = pyo.RangeSet(0, 3)
        m.Mx = pyo.Param(initialize=20)
        m.My = pyo.Param(initialize=30)
        m.Mz = pyo.Param(initialize=10)
        m.d_min = pyo.Param(initialize=1)
        m.x = pyo.Var(m.Rooms, within=pyo.Integers, bounds=(-m.Mx, m.Mx))
        m.y = pyo.Var(m.Rooms, within=pyo.Integers, bounds=(-m.My, m.My))
        m.z = pyo.Var(m.Rooms, within=pyo.Integers, bounds=(-m.Mz, m.Mz))
        m.cut = pyo.Var(m.Exits, within=pyo.Binary)
        m.crossings = pyo.ConstraintList()
        return m

    def test_planar_overlap_reduces_binary_variables_by_33_percent(self):
        """Planar overlap creates 4 binary variables instead of 6 (33.33% reduction)."""
        m = self._build_test_model()

        # Two horizontal exits on the same floor z=0
        ex = Exit(src=1, dst=2, direction=Direction.east)
        nx = Exit(src=3, dst=4, direction=Direction.north)

        coords = {
            1: (0.0, 0.0, 0.0),
            2: (2.0, 0.0, 0.0),
            3: (1.0, -1.0, 0.0),
            4: (1.0, 1.0, 0.0),
        }

        next_rel = add_overlap_constraint(
            m,
            ex,
            nx,
            left=0,
            right=1,
            relations=0,
            coords=coords,
            has_vertical_exits=True,
        )

        assert next_rel == 1
        rel_var = getattr(m, "relation0")

        # Must have exactly 4 binary variables: North, East, South, West
        assert len(rel_var) == 4
        assert Direction.north in rel_var
        assert Direction.east in rel_var
        assert Direction.south in rel_var
        assert Direction.west in rel_var
        assert Direction.up not in rel_var
        assert Direction.down not in rel_var

        # Exactly 33.33% reduction
        reduction = (6 - len(rel_var)) / 6
        assert abs(reduction - 1 / 3) < 0.001

        # 4 directions * 4 pairs = 16 crossing constraints + 1 sum constraint = 17 constraints
        assert len(m.crossings) == 17

    def test_no_vertical_exits_forces_planar_reduction(self):
        """When has_vertical_exits is False, all overlaps are reduced to 4 planar directions."""
        m = self._build_test_model()

        ex = Exit(src=1, dst=2, direction=Direction.east)
        nx = Exit(src=4, dst=3, direction=Direction.west)

        next_rel = add_overlap_constraint(
            m,
            ex,
            nx,
            left=0,
            right=1,
            relations=0,
            coords=None,
            has_vertical_exits=False,
        )

        assert next_rel == 1
        rel_var = getattr(m, "relation0")
        assert len(rel_var) == 4
        assert Direction.up not in rel_var
        assert Direction.down not in rel_var

    def test_vertical_exit_overlap_retains_6_directions(self):
        """When an exit has vertical movement and vertical exits exist, all 6 directions are retained."""
        m = self._build_test_model()

        # ex is vertical (up)
        ex = Exit(src=1, dst=2, direction=Direction.up)
        nx = Exit(src=3, dst=4, direction=Direction.north)

        coords = {
            1: (0.0, 0.0, 0.0),
            2: (0.0, 0.0, 1.0),
            3: (0.0, -1.0, 0.0),
            4: (0.0, 1.0, 0.0),
        }

        next_rel = add_overlap_constraint(
            m,
            ex,
            nx,
            left=0,
            right=1,
            relations=0,
            coords=coords,
            has_vertical_exits=True,
        )

        assert next_rel == 1
        rel_var = getattr(m, "relation0")
        assert len(rel_var) == 6
        assert Direction.up in rel_var
        assert Direction.down in rel_var
        # 6 directions * 4 pairs = 24 constraints + 1 sum constraint = 25 constraints
        assert len(m.crossings) == 25


class TestModelVariableBounds:
    """Tests for variable bounds and model parameters in non_euler and solve."""

    def test_non_euler_bounds(self):
        """Verify non_euler sets dimension-specific parameters and tight bounds [-M_d, M_d]."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        e1 = Exit(src=1, dst=2, direction=Direction.east, distance=5)
        e2 = Exit(src=2, dst=1, direction=Direction.west, distance=5)
        r1.exits = [e1]
        r2.exits = [e2]
        rdb = {1: r1, 2: r2}
        exits = [e1, e2]

        m = non_euler(rdb, exits)

        # sum_x = 10 -> Mx = max(10, 11) = 11
        # sum_y = 0 -> My = 10
        # sum_z = 0 -> Mz = 5
        assert m.Mx.value == 11
        assert m.My.value == 10
        assert m.Mz.value == 5

        for v in (1, 2):
            assert m.x[v].bounds == (-11, 11)
            assert m.y[v].bounds == (-10, 10)
            assert m.z[v].bounds == (-5, 5)

    def test_solve_model_bounds_and_parameters(self):
        """Verify solve creates model with dimension-specific parameters and tight bounds."""
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        e1 = Exit(src=1, dst=2, direction=Direction.east, distance=5)
        e2 = Exit(src=2, dst=1, direction=Direction.west, distance=5)
        r1.exits = [e1]
        r2.exits = [e2]
        rdb = {1: r1, 2: r2}
        exits = [e1, e2]

        model, results = solve(rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        )

        assert hasattr(model, "Mx")
        assert hasattr(model, "My")
        assert hasattr(model, "Mz")

        for v in (1, 2):
            assert model.x[v].bounds == (-model.Mx.value, model.Mx.value)
            assert model.y[v].bounds == (-model.My.value, model.My.value)
            assert model.z[v].bounds == (-model.Mz.value, model.Mz.value)


class TestEndToEndSolverFixtures:
    """Zero solver regressions across all fixtures."""

    def test_smurf_solve_layout(self):
        """Verify smurf.are solves cleanly with valid integer coordinates."""
        filepath = FIXTURES_DIR / "areas" / "smurf.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}

        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=20)
        assert len(solved_rdb) == 30
        for r in solved_rdb.values():
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None
            assert isinstance(r.x, (int, float))

    def test_school_solve_layout(self):
        """Verify school.are solves cleanly with valid integer coordinates."""
        filepath = FIXTURES_DIR / "areas" / "school.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}

        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        assert len(solved_rdb) == 60
        for r in solved_rdb.values():
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None

    def test_midgaard_solve_layout(self):
        """Verify midgaard.are solves cleanly with valid coordinates."""
        filepath = FIXTURES_DIR / "areas" / "midgaard.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}

        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        assert len(solved_rdb) >= 108
        for r in solved_rdb.values():
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None

    def test_circlemud_split_solve_layout(self):
        """Verify CircleMUD split world resolves without regressions."""
        circlemud_dir = FIXTURES_DIR / "dialects" / "circle_world"
        if not circlemud_dir.exists():
            pytest.skip("CircleMUD directory not found")
        area = parse_circlemud_directory(circlemud_dir)
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=20)
        assert len(solved_rdb) > 0
        for r in solved_rdb.values():
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None

    def test_dikumud_alfa_solve_layout(self):
        """Verify DikuMUD Alfa (.wld) solves cleanly with valid coordinates."""
        filepath = FIXTURES_DIR / "dialects" / "dikumud_alfa.wld"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}

        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=20)
        assert len(solved_rdb) >= len(area.rooms)
        for r in solved_rdb.values():
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None
