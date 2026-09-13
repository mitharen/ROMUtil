"""Tests for CBC Solver Parameter Tuning and Multithreading (Task 1f).

Validates:
1. CBC Solver Options Configuration:
   - Threads: configured to min(4, os.cpu_count() or 1).
   - Optimality Gap: ratioGap configured to 0.05 (5% relative MIP gap).
   - Presolve, cuts, and heuristics: presolve='on', cuts='on', heuristics='on'.
   - Timeout handling: seconds (and sec) passed when timeout is specified, omitted when None.
   - Custom options and overrides via kwargs.
2. CPU Count Fallback:
   - os.cpu_count() -> None or 1 sets threads=1.
   - os.cpu_count() -> 2 sets threads=2.
   - os.cpu_count() >= 4 capped at threads=4.
3. Integration with non_euler() and solve():
   - non_euler() and solve() instantiate and apply tuned CBC options.
4. Fixture Solves & Integer Coordinate Validity:
   - Verifies layout solving succeeds across standard area fixtures:
     smurf.are, school.are, midgaard.are, CircleMUD split world (circle_world),
     and DikuMUD Alfa (dikumud_alfa.wld).
"""

import copy
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pyomo.environ as pyo
import pytest

from romutil.graph import solve_layout
from romutil.models import AreaData, AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.parser import Parser, parse_circlemud_directory
from romutil.solver import compute_dynamic_solver_timeout, get_cbc_solver, non_euler, solve


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCBCSolverConfiguration:
    """Unit tests for get_cbc_solver configuration and option settings."""

    def test_default_options(self):
        """Verify tuned default options on CBC solver instance."""
        solver = get_cbc_solver()
        cpu_count = os.cpu_count() or 1
        expected_threads = min(4, cpu_count)

        assert solver.options["threads"] == expected_threads
        assert solver.options["ratioGap"] == 0.05
        assert solver.options["presolve"] == "on"
        assert solver.options["cuts"] == "on"
        assert solver.options["heuristics"] == "on"
        assert "seconds" not in solver.options
        assert "sec" not in solver.options

    def test_timeout_specified(self):
        """When timeout is provided, both 'seconds' and 'sec' are configured."""
        solver = get_cbc_solver(timeout=120)
        assert solver.options["seconds"] == 120
        assert solver.options["sec"] == 120
        assert solver.options["ratioGap"] == 0.05
        assert solver.options["presolve"] == "on"
        assert solver.options["cuts"] == "on"
        assert solver.options["heuristics"] == "on"

    def test_custom_options_override_and_extend(self):
        """Custom keyword arguments override defaults and add extra options."""
        solver = get_cbc_solver(
            timeout=45,
            ratioGap=0.01,
            threads=2,
            maxNodes=1000,
        )
        assert solver.options["seconds"] == 45
        assert solver.options["sec"] == 45
        assert solver.options["ratioGap"] == 0.01
        assert solver.options["threads"] == 2
        assert solver.options["maxNodes"] == 1000
        assert solver.options["presolve"] == "on"

    def test_cpu_count_fallback_none(self, monkeypatch):
        """When os.cpu_count() returns None, threads defaults to 1."""
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        solver = get_cbc_solver()
        assert solver.options["threads"] == 1

    def test_cpu_count_fallback_one(self, monkeypatch):
        """When os.cpu_count() returns 1, threads defaults to 1."""
        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        solver = get_cbc_solver()
        assert solver.options["threads"] == 1

    def test_cpu_count_two(self, monkeypatch):
        """When os.cpu_count() returns 2, threads defaults to 2."""
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        solver = get_cbc_solver()
        assert solver.options["threads"] == 2

    def test_cpu_count_capped_at_four(self, monkeypatch):
        """When os.cpu_count() returns large numbers (e.g. 16, 64), threads is capped at 4."""
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        solver = get_cbc_solver()
        assert solver.options["threads"] == 4

        monkeypatch.setattr(os, "cpu_count", lambda: 64)
        solver_large = get_cbc_solver()
        assert solver_large.options["threads"] == 4

    def test_non_euler_invokes_tuned_solver(self, monkeypatch):
        """non_euler() instantiates solver with timeout=20 and tuned options."""
        captured_options = {}
        import romutil.solver as solver_mod
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_get_cbc(timeout=None, **kwargs):
            s = orig_get_cbc(timeout=timeout, **kwargs)
            captured_options.update(dict(s.options))
            return s

        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        r1 = Room(RoomDef(vnum=1, name="R1", description="D", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="D", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        m = non_euler(rdb, exits)
        assert captured_options.get("threads") in (1, 2, 3, 4)
        assert captured_options.get("ratioGap") == 0.05
        assert captured_options.get("presolve") == "on"
        assert captured_options.get("cuts") == "on"
        assert captured_options.get("heuristics") == "on"
        assert captured_options.get("seconds") == 20
        assert captured_options.get("sec") == 20

    def test_solve_invokes_tuned_solver_with_timeout(self, monkeypatch):
        """solve() applies default timeout 300 or explicit timeout with tuned options."""
        captured_options = []
        import romutil.solver as solver_mod
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_get_cbc(timeout=None, **kwargs):
            s = orig_get_cbc(timeout=timeout, **kwargs)
            captured_options.append(dict(s.options))
            return s

        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        r1 = Room(RoomDef(vnum=1, name="R1", description="D", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="D", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        solve(rdb, exits, timeout=45)
        # Check that solver options in solve() included 45s timeout and tuned options
        solve_opts = captured_options[-1]
        assert solve_opts.get("seconds") == 45
        assert solve_opts.get("sec") == 45
        assert solve_opts.get("ratioGap") == 0.05
        assert solve_opts.get("presolve") == "on"
        assert solve_opts.get("cuts") == "on"
        assert solve_opts.get("heuristics") == "on"


@pytest.mark.slow
@pytest.mark.integration
class TestStandardFixturesLayoutSolving:
    """Verify layout solving succeeds with valid integer coordinates across all standard area fixtures."""

    def _verify_coordinates(self, solved_rdb):
        """Helper to assert all rooms have valid non-None integer coordinates."""
        assert len(solved_rdb) > 0
        for vnum, room in solved_rdb.items():
            assert room.x is not None, f"Room {vnum} x is None"
            assert room.y is not None, f"Room {vnum} y is None"
            assert room.z is not None, f"Room {vnum} z is None"
            assert isinstance(room.x, (int, float)), f"Room {vnum} x not numeric: {room.x}"
            assert isinstance(room.y, (int, float)), f"Room {vnum} y not numeric: {room.y}"
            assert isinstance(room.z, (int, float)), f"Room {vnum} z not numeric: {room.z}"
            assert round(room.x) == room.x, f"Room {vnum} x not integer: {room.x}"
            assert round(room.y) == room.y, f"Room {vnum} y not integer: {room.y}"
            assert round(room.z) == room.z, f"Room {vnum} z not integer: {room.z}"

    def test_smurf_solve(self):
        """Verify smurf.are solves cleanly with valid integer coordinates."""
        filepath = FIXTURES_DIR / "areas" / "smurf.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        self._verify_coordinates(solved_rdb)

    def test_school_solve(self):
        """Verify school.are solves cleanly with valid integer coordinates."""
        filepath = FIXTURES_DIR / "areas" / "school.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        self._verify_coordinates(solved_rdb)

    def test_midgaard_solve(self):
        """Verify midgaard.are solves cleanly with valid integer coordinates."""
        filepath = FIXTURES_DIR / "areas" / "midgaard.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        self._verify_coordinates(solved_rdb)

    def test_circlemud_split_solve(self):
        """Verify CircleMUD split world resolves cleanly with valid integer coordinates."""
        circlemud_dir = FIXTURES_DIR / "dialects" / "circle_world"
        if not circlemud_dir.exists():
            pytest.skip("CircleMUD directory fixture not found")
        area = parse_circlemud_directory(circlemud_dir)
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        self._verify_coordinates(solved_rdb)

    def test_dikumud_alfa_solve(self):
        """Verify DikuMUD Alfa world resolves cleanly with valid integer coordinates."""
        filepath = FIXTURES_DIR / "dialects" / "dikumud_alfa.wld"
        if not filepath.exists():
            pytest.skip("DikuMUD Alfa fixture not found")
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)
        self._verify_coordinates(solved_rdb)



class TestDynamicSolverTimeout:
    """Unit tests for dynamic room-scaled solver timeout computation and plumbing."""

    def test_compute_dynamic_solver_timeout_boundary_cases(self):
        """Verify dynamic timeout scaling across standard room count boundary cases."""
        # 0 rooms: floored to min_timeout (30s)
        assert compute_dynamic_solver_timeout(0) == 30
        # Negative room count floored to min_timeout
        assert compute_dynamic_solver_timeout(-10) == 30

        # Benchmark reference points from design specification:
        # 10 rooms: round(30 + 0.8 * 10) = 38s
        assert compute_dynamic_solver_timeout(10) == 38
        # 20 rooms: round(30 + 0.8 * 20) = 46s
        assert compute_dynamic_solver_timeout(20) == 46
        # 60 rooms: round(30 + 0.8 * 60) = 78s
        assert compute_dynamic_solver_timeout(60) == 78
        # 100 rooms: round(30 + 0.8 * 100) = 110s
        assert compute_dynamic_solver_timeout(100) == 110
        # 110 rooms: round(30 + 0.8 * 110) = 118s
        assert compute_dynamic_solver_timeout(110) == 118
        # 230 rooms: round(30 + 0.8 * 230) = 214s
        assert compute_dynamic_solver_timeout(230) == 214
        # 250 rooms: round(30 + 0.8 * 250) = 230s
        assert compute_dynamic_solver_timeout(250) == 230
        # 350 rooms: round(30 + 0.8 * 350) = 310s -> capped at max_timeout (300s)
        assert compute_dynamic_solver_timeout(350) == 300
        # 1000 rooms: round(30 + 0.8 * 1000) = 830s -> capped at max_timeout (300s)
        assert compute_dynamic_solver_timeout(1000) == 300

    def test_compute_dynamic_solver_timeout_custom_min_max(self):
        """Verify dynamic timeout with custom min_timeout and max_timeout limits."""
        # Custom bounds: min=50, max=200
        assert compute_dynamic_solver_timeout(0, min_timeout=50, max_timeout=200) == 50
        assert compute_dynamic_solver_timeout(10, min_timeout=50, max_timeout=200) == 50
        assert compute_dynamic_solver_timeout(100, min_timeout=50, max_timeout=200) == 110
        assert compute_dynamic_solver_timeout(250, min_timeout=50, max_timeout=200) == 200
        assert compute_dynamic_solver_timeout(1000, min_timeout=50, max_timeout=200) == 200

        # Custom bounds: min=10, max=40
        assert compute_dynamic_solver_timeout(0, min_timeout=10, max_timeout=40) == 30
        assert compute_dynamic_solver_timeout(20, min_timeout=10, max_timeout=40) == 40

        # Invalid bounds where min_timeout > max_timeout
        with pytest.raises(ValueError, match="cannot exceed max_timeout"):
            compute_dynamic_solver_timeout(50, min_timeout=100, max_timeout=50)

    def test_solve_dynamic_timeout_applied_when_none(self, monkeypatch):
        """solve() computes and applies dynamic room-scaled timeout when solver_timeout is None."""
        captured_options = []
        import romutil.solver as solver_mod
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_get_cbc(timeout=None, **kwargs):
            s = orig_get_cbc(timeout=timeout, **kwargs)
            captured_options.append(dict(s.options))
            return s

        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        r1 = Room(RoomDef(vnum=1, name="R1", description="D", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="D", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        # Call with solver_timeout=None
        model, results = solve(rdb, exits, solver_timeout=None)
        assert results is not None
        expected_sec = compute_dynamic_solver_timeout(2)  # 32s
        assert expected_sec == 32
        solve_opts = captured_options[-1]
        assert solve_opts.get("seconds") == 32
        assert solve_opts.get("sec") == 32

    def test_solve_dynamic_timeout_applied_when_zero_or_negative(self, monkeypatch):
        """solve() applies dynamic timeout when solver_timeout <= 0."""
        captured_options = []
        import romutil.solver as solver_mod
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_get_cbc(timeout=None, **kwargs):
            s = orig_get_cbc(timeout=timeout, **kwargs)
            captured_options.append(dict(s.options))
            return s

        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        r1 = Room(RoomDef(vnum=1, name="R1", description="D", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="D", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        # solver_timeout = 0
        solve(rdb, exits, solver_timeout=0)
        assert captured_options[-1].get("seconds") == 32

        # solver_timeout = -15
        solve(rdb, exits, solver_timeout=-15)
        assert captured_options[-1].get("seconds") == 32

    def test_solve_explicit_timeout_strictly_preserved(self, monkeypatch):
        """Explicit positive timeout is preserved by solve()."""
        captured_options = []
        import romutil.solver as solver_mod
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_get_cbc(timeout=None, **kwargs):
            s = orig_get_cbc(timeout=timeout, **kwargs)
            captured_options.append(dict(s.options))
            return s

        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        r1 = Room(RoomDef(vnum=1, name="R1", description="D", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="D", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        # Explicit solver_timeout=15
        solve(rdb, exits, solver_timeout=15)
        assert captured_options[-1].get("seconds") == 15

        # Explicit solver_timeout takes precedence over legacy timeout
        solve(rdb, exits, timeout=10, solver_timeout=25)
        assert captured_options[-1].get("seconds") == 25


class TestCliDynamicTimeoutIntegration:
    """Tests for CLI integration with dynamic solver timeout."""

    def test_main_computes_dynamic_timeout_and_logs(self, tmp_path, monkeypatch, caplog):
        """When solver_timeout is None, main() logs dynamic component scaling and passes None to solve_layout."""
        import sys
        from romutil.models import AreaData, AreaHeader
        from romutil.parser import Parser

        are_file = tmp_path / "test.are"
        are_file.write_text("dummy")

        mock_area = AreaData(
            header=AreaHeader("test.are", "TestArea", "Builder", 1, 3),
            rooms=[
                RoomDef(1, "R1", "D1", exits=(ExitDef(direction=1, dst_vnum=2),)),
                RoomDef(2, "R2", "D2", exits=(ExitDef(direction=3, dst_vnum=1), ExitDef(direction=1, dst_vnum=3))),
                RoomDef(3, "R3", "D3", exits=(ExitDef(direction=3, dst_vnum=2),)),
            ],
        )
        monkeypatch.setattr(Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)
            for r in rdb.values():
                r.x = 0
                r.y = 0
                r.z = 0
            return rdb, []

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "solve_layout", mock_solve_layout)

        with caplog.at_level(logging.INFO, logger="Mapper"):
            try:
                cli_mod.main([are_file], str(tmp_path / "out"), fmt="json", solver_timeout=None)
            except SystemExit:
                pass

        assert passed_timeouts == [None]
        assert "Dynamic solver timeout scaling enabled across 1 component(s)" in caplog.text

    def test_main_preserves_explicit_timeout_without_dynamic_log(self, tmp_path, monkeypatch, caplog):
        """When solver_timeout is explicitly provided, main() forwards it without dynamic timeout log."""
        import sys
        from romutil.models import AreaData, AreaHeader
        from romutil.parser import Parser

        are_file = tmp_path / "test.are"
        are_file.write_text("dummy")

        mock_area = AreaData(
            header=AreaHeader("test.are", "TestArea", "Builder", 1, 2),
            rooms=[
                RoomDef(1, "R1", "D1", exits=(ExitDef(direction=1, dst_vnum=2),)),
                RoomDef(2, "R2", "D2", exits=(ExitDef(direction=3, dst_vnum=1),)),
            ],
        )
        monkeypatch.setattr(Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)
            for r in rdb.values():
                r.x = 0
                r.y = 0
                r.z = 0
            return rdb, []

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "solve_layout", mock_solve_layout)

        with caplog.at_level(logging.INFO, logger="Mapper"):
            try:
                cli_mod.main([are_file], str(tmp_path / "out"), fmt="json", solver_timeout=80)
            except SystemExit:
                pass

        assert passed_timeouts == [80]
        assert "Dynamic solver timeout scaling enabled" not in caplog.text
        assert "Using user-specified fixed solver timeout of 80s across components" in caplog.text

    def test_cli_invoked_omitted_timeout_uses_dynamic(self, tmp_path, monkeypatch, caplog):
        """CLI invocation without --solver-timeout forwards solver_timeout=None for dynamic scaling."""
        import sys
        from romutil.models import AreaData, AreaHeader
        from romutil.parser import Parser

        are_file = tmp_path / "cli_test.are"
        are_file.write_text("dummy")

        # Create 10 connected rooms in a line (10->11->...->19)
        rooms = []
        for i in range(10, 20):
            exits = (ExitDef(direction=1, dst_vnum=i+1),) if i < 19 else (ExitDef(direction=3, dst_vnum=i-1),)
            rooms.append(RoomDef(i, f"R{i}", f"D{i}", exits=exits))
        mock_area = AreaData(
            header=AreaHeader("cli_test.are", "CliTest", "Builder", 10, 19),
            rooms=rooms,
        )
        monkeypatch.setattr(Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)
            for r in rdb.values():
                r.x = 0
                r.y = 0
                r.z = 0
            return rdb, []

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "solve_layout", mock_solve_layout)
        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "-f", "json"])

        with caplog.at_level(logging.INFO, logger="Mapper"):
            try:
                cli_mod.cli()
            except SystemExit:
                pass

        assert passed_timeouts == [None]
        assert "Dynamic solver timeout scaling enabled across 1 component(s)" in caplog.text

    def test_cli_invoked_explicit_timeout_strictly_preserved(self, tmp_path, monkeypatch, caplog):
        """CLI invocation with explicit --solver-timeout preserves that timeout."""
        import sys
        from romutil.models import AreaData, AreaHeader
        from romutil.parser import Parser

        are_file = tmp_path / "cli_test.are"
        are_file.write_text("dummy")

        # Create 5 connected rooms in a line (1->2->...->5)
        rooms = []
        for i in range(1, 6):
            exits = (ExitDef(direction=1, dst_vnum=i+1),) if i < 5 else (ExitDef(direction=3, dst_vnum=i-1),)
            rooms.append(RoomDef(i, f"R{i}", f"D{i}", exits=exits))
        mock_area = AreaData(
            header=AreaHeader("cli_test.are", "CliTest", "Builder", 1, 5),
            rooms=rooms,
        )
        monkeypatch.setattr(Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)
            for r in rdb.values():
                r.x = 0
                r.y = 0
                r.z = 0
            return rdb, []

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "solve_layout", mock_solve_layout)
        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "--solver-timeout", "95", "-f", "json"])

        with caplog.at_level(logging.INFO, logger="Mapper"):
            try:
                cli_mod.cli()
            except SystemExit:
                pass

        assert passed_timeouts == [95]
        assert "Dynamic solver timeout scaling enabled" not in caplog.text
        assert "Using user-specified fixed solver timeout of 95s across components" in caplog.text


class TestComponentLevelDynamicTimeoutScaling:
    """Unit and integration tests for component-level dynamic solver timeout scaling (Task 1o).

    Verifies:
    1. Component-level dynamic scaling: When solver_timeout is None, disconnected components
       receive dynamic timeouts scaled to their respective room counts (len(sub_rdb)) rather
       than the global area room count.
    2. Explicit override: When solver_timeout is explicitly specified (e.g. 45s), that fixed
       timeout is applied to each component.
    3. CLI integration: CLI parsing and multi-component dispatch respects explicit vs None solver_timeout.
    """

    def test_solve_layout_component_level_dynamic_scaling(self, monkeypatch, caplog):
        """When solver_timeout is None, disconnected components receive timeouts scaled to each component."""
        # Component 1: 2 rooms (101 <-> 102) -> expected timeout = compute_dynamic_solver_timeout(2) = 32s
        # Component 2: 10 rooms (201 <-> 202 <-> ... <-> 210) -> expected timeout = compute_dynamic_solver_timeout(10) = 38s
        # Total rooms in area = 12 rooms -> global timeout would be compute_dynamic_solver_timeout(12) = 40s
        rooms_c1 = [
            Room(RoomDef(101, "C1_1", "D", exits=(ExitDef(direction=1, dst_vnum=102),))),
            Room(RoomDef(102, "C1_2", "D", exits=(ExitDef(direction=3, dst_vnum=101),))),
        ]
        # Alternating East (1) and North (0) so straight hallway collapse does not trim intermediate rooms
        rooms_c2 = [
            Room(RoomDef(201, "C2_201", "D", exits=(ExitDef(direction=1, dst_vnum=202),))),
            Room(RoomDef(202, "C2_202", "D", exits=(ExitDef(direction=3, dst_vnum=201), ExitDef(direction=0, dst_vnum=203)))),
            Room(RoomDef(203, "C2_203", "D", exits=(ExitDef(direction=2, dst_vnum=202), ExitDef(direction=1, dst_vnum=204)))),
            Room(RoomDef(204, "C2_204", "D", exits=(ExitDef(direction=3, dst_vnum=203), ExitDef(direction=0, dst_vnum=205)))),
            Room(RoomDef(205, "C2_205", "D", exits=(ExitDef(direction=2, dst_vnum=204), ExitDef(direction=1, dst_vnum=206)))),
            Room(RoomDef(206, "C2_206", "D", exits=(ExitDef(direction=3, dst_vnum=205), ExitDef(direction=0, dst_vnum=207)))),
            Room(RoomDef(207, "C2_207", "D", exits=(ExitDef(direction=2, dst_vnum=206), ExitDef(direction=1, dst_vnum=208)))),
            Room(RoomDef(208, "C2_208", "D", exits=(ExitDef(direction=3, dst_vnum=207), ExitDef(direction=0, dst_vnum=209)))),
            Room(RoomDef(209, "C2_209", "D", exits=(ExitDef(direction=2, dst_vnum=208), ExitDef(direction=1, dst_vnum=210)))),
            Room(RoomDef(210, "C2_210", "D", exits=(ExitDef(direction=3, dst_vnum=209),))),
        ]

        rdb = {r.vnum: r for r in rooms_c1 + rooms_c2}
        assert len(rdb) == 12

        captured_solve_calls = []
        captured_cbc_timeouts = []

        import sys; graph_mod = sys.modules["romutil.graph"]
        import romutil.solver as solver_mod

        orig_solve = graph_mod.solve
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_solve(rdb_i, exits_i, timeout=None, **kwargs):
            captured_solve_calls.append({
                "room_count": len(rdb_i),
                "timeout": timeout,
                "vnums": set(rdb_i.keys()),
            })
            return orig_solve(rdb_i, exits_i, timeout=timeout, **kwargs)

        def spy_get_cbc(timeout=None, **kwargs):
            captured_cbc_timeouts.append(timeout)
            return orig_get_cbc(timeout=timeout, **kwargs)

        monkeypatch.setattr(graph_mod, "solve", spy_solve)
        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        with caplog.at_level(logging.INFO, logger="Mapper.graph"):
            solved_rdb, solved_exits = solve_layout(
                rdb,
                area=AreaHeader("multi.are", "MultiComp", "Builder", 101, 210),
                solver_timeout=None,
            )

        assert len(captured_solve_calls) == 2
        # Both calls to solve() were invoked with timeout=None so solve() computes dynamic component timeout
        call_c1 = [c for c in captured_solve_calls if 101 in c["vnums"]][0]
        call_c2 = [c for c in captured_solve_calls if 201 in c["vnums"]][0]

        assert call_c1["timeout"] is None
        assert call_c2["timeout"] is None

        # Verify get_cbc_solver was configured with 32s (2 rooms) and 38s (10 rooms), NOT global 40s (12 rooms)
        assert len(captured_cbc_timeouts) >= 2
        assert 32 in captured_cbc_timeouts
        assert 38 in captured_cbc_timeouts
        assert 40 not in captured_cbc_timeouts

        # Verify logging captured dynamic component timeouts
        assert "Component 1/2: solving for 1 exits across 2 rooms (dynamic timeout 32s)..." in caplog.text
        assert "Component 2/2: solving for 9 exits across 10 rooms (dynamic timeout 38s)..." in caplog.text

        # Verify all rooms solved have valid coordinates
        for v, r in solved_rdb.items():
            assert r.x is not None and r.y is not None and r.z is not None

    def test_solve_layout_explicit_timeout_override_all_components(self, monkeypatch, caplog):
        """When solver_timeout is explicitly provided, each component uses the fixed timeout."""
        rooms_c1 = [
            Room(RoomDef(101, "C1_1", "D", exits=(ExitDef(direction=1, dst_vnum=102),))),
            Room(RoomDef(102, "C1_2", "D", exits=(ExitDef(direction=3, dst_vnum=101),))),
        ]
        rooms_c2 = [
            Room(RoomDef(201, "C2_1", "D", exits=(ExitDef(direction=0, dst_vnum=202),))),
            Room(RoomDef(202, "C2_2", "D", exits=(ExitDef(direction=2, dst_vnum=201),))),
        ]
        rdb = {r.vnum: r for r in rooms_c1 + rooms_c2}

        captured_timeouts = []
        import sys; graph_mod = sys.modules["romutil.graph"]
        orig_solve = graph_mod.solve

        def spy_solve(rdb_i, exits_i, timeout=None, **kwargs):
            captured_timeouts.append(timeout)
            return orig_solve(rdb_i, exits_i, timeout=timeout, **kwargs)

        monkeypatch.setattr(graph_mod, "solve", spy_solve)

        with caplog.at_level(logging.INFO, logger="Mapper.graph"):
            solved_rdb, solved_exits = solve_layout(
                rdb,
                area=AreaHeader("multi.are", "MultiComp", "Builder", 101, 202),
                solver_timeout=45,
            )

        assert captured_timeouts == [45, 45]
        assert "fixed timeout 45s" in caplog.text
        assert "dynamic timeout" not in caplog.text

    def test_solve_layout_single_component_dynamic_vs_explicit(self, monkeypatch, caplog):
        """Single component area dynamically scales when None, uses fixed when provided."""
        rooms = [
            Room(RoomDef(1, "R1", "D", exits=(ExitDef(direction=1, dst_vnum=2),))),
            Room(RoomDef(2, "R2", "D", exits=(ExitDef(direction=3, dst_vnum=1), ExitDef(direction=0, dst_vnum=3)))),
            Room(RoomDef(3, "R3", "D", exits=(ExitDef(direction=2, dst_vnum=2),))),
        ]
        rdb = {r.vnum: r for r in rooms}

        captured_timeouts = []
        captured_cbc_timeouts = []

        import sys; graph_mod = sys.modules["romutil.graph"]
        import romutil.solver as solver_mod

        orig_solve = graph_mod.solve
        orig_get_cbc = solver_mod.get_cbc_solver

        def spy_solve(rdb_i, exits_i, timeout=None, **kwargs):
            captured_timeouts.append(timeout)
            return orig_solve(rdb_i, exits_i, timeout=timeout, **kwargs)

        def spy_get_cbc(timeout=None, **kwargs):
            captured_cbc_timeouts.append(timeout)
            return orig_get_cbc(timeout=timeout, **kwargs)

        monkeypatch.setattr(graph_mod, "solve", spy_solve)
        monkeypatch.setattr(solver_mod, "get_cbc_solver", spy_get_cbc)

        # 1. solver_timeout is None -> dynamic timeout = 32s (3 rooms)
        caplog.clear()
        captured_cbc_timeouts.clear()
        with caplog.at_level(logging.INFO, logger="Mapper.graph"):
            solve_layout(copy.deepcopy(rdb), solver_timeout=None)
        assert captured_timeouts[-1] is None
        assert captured_cbc_timeouts[-1] == 32
        assert "dynamic timeout 32s" in caplog.text

        # 2. solver_timeout is 65 -> fixed timeout = 65s
        caplog.clear()
        captured_cbc_timeouts.clear()
        with caplog.at_level(logging.INFO, logger="Mapper.graph"):
            solve_layout(copy.deepcopy(rdb), solver_timeout=65)
        assert captured_timeouts[-1] == 65
        assert captured_cbc_timeouts[-1] == 65
        assert "fixed timeout 65s" in caplog.text

    def test_cli_multi_component_dispatch_dynamic_vs_explicit(self, tmp_path, monkeypatch, caplog):
        """CLI correctly dispatches solver_timeout=None or explicit value across multi-component areas."""
        import sys
        from romutil.models import AreaData, AreaHeader
        from romutil.parser import Parser

        are_file = tmp_path / "multi_cli.are"
        are_file.write_text("dummy")

        # Two disconnected components in the area:
        # C1: 1 <-> 2
        # C2: 10 <-> 11
        mock_area = AreaData(
            header=AreaHeader("multi_cli.are", "MultiCLI", "Builder", 1, 11),
            rooms=[
                RoomDef(1, "R1", "D1", exits=(ExitDef(direction=1, dst_vnum=2),)),
                RoomDef(2, "R2", "D2", exits=(ExitDef(direction=3, dst_vnum=1),)),
                RoomDef(10, "R10", "D10", exits=(ExitDef(direction=1, dst_vnum=11),)),
                RoomDef(11, "R11", "D11", exits=(ExitDef(direction=3, dst_vnum=10),)),
            ],
        )
        monkeypatch.setattr(Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        passed_sub_rdbs = []
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)
            passed_sub_rdbs.append(sorted(rdb.keys()))
            for r in rdb.values():
                r.x = 0
                r.y = 0
                r.z = 0
            return rdb, []

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "solve_layout", mock_solve_layout)

        # Case 1: solver_timeout=None -> passes solver_timeout=None to each component
        caplog.clear()
        passed_timeouts.clear()
        passed_sub_rdbs.clear()
        with caplog.at_level(logging.INFO, logger="Mapper"):
            try:
                cli_mod.main([are_file], str(tmp_path / "out1"), fmt="json", solver_timeout=None)
            except SystemExit:
                pass

        assert len(passed_timeouts) == 2
        assert passed_timeouts == [None, None]
        assert [1, 2] in passed_sub_rdbs
        assert [10, 11] in passed_sub_rdbs
        assert "Dynamic solver timeout scaling enabled across 2 component(s)" in caplog.text

        # Case 2: explicit solver_timeout=75 -> passes 75 to each component
        caplog.clear()
        passed_timeouts.clear()
        passed_sub_rdbs.clear()
        with caplog.at_level(logging.INFO, logger="Mapper"):
            try:
                cli_mod.main([are_file], str(tmp_path / "out2"), fmt="json", solver_timeout=75)
            except SystemExit:
                pass

        assert len(passed_timeouts) == 2
        assert passed_timeouts == [75, 75]
        assert [1, 2] in passed_sub_rdbs
        assert [10, 11] in passed_sub_rdbs
        assert "Using user-specified fixed solver timeout of 75s across components" in caplog.text
