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

import os
from pathlib import Path
from unittest.mock import MagicMock

import pyomo.environ as pyo
import pytest

from romutil.graph import solve_layout
from romutil.models import Direction, Exit, ExitDef, Room, RoomDef
from romutil.parser import Parser, parse_circlemud_directory
from romutil.solver import get_cbc_solver, non_euler, solve


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
