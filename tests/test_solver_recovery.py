"""Comprehensive unit and integration tests for solver recovery and timeout parameterization.

Verifies:
1. Solver timeout parameterization (romutil/solver.py)
2. Feasible solution recovery on maxTimeLimit / timeout (romutil/graph.py)
3. Fallback to (0, 0, 0) on infeasible or unassigned conditions
4. CLI --solver-timeout argument parsing and plumbing (romutil/cli.py)
"""

import copy
import logging
from pathlib import Path
import sys
from unittest.mock import MagicMock
import pyomo.opt
import pytest

from romutil.models import Room, RoomDef, Exit, ExitDef, AreaHeader, AreaData
from romutil.solver import solve
from romutil.graph import solve_layout, graph, _has_feasible_coordinates
from romutil.cli import cli, main


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


class TestSolverTimeoutParameterization:
    """Unit tests for solve() timeout parameterization and loop termination."""

    def test_solve_default_timeout_used(self, monkeypatch):
        """When timeout is None, CBC solver option 'sec' defaults to 300."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        recorded_sec = []
        solver_mod = sys.modules["romutil.solver"]
        orig_factory = solver_mod.SolverFactory

        def mock_factory(name, **kwargs):
            s = orig_factory(name, **kwargs)
            orig_solve = s.solve
            def wrapped_solve(m, **kw):
                recorded_sec.append(s.options.get("sec"))
                return orig_solve(m, **kw)
            s.solve = wrapped_solve
            return s

        monkeypatch.setattr(solver_mod, "SolverFactory", mock_factory)
        model, result = solve(rdb, exits)
        assert 300 in recorded_sec

    def test_solve_custom_timeout_passed(self, monkeypatch):
        """When timeout is explicitly provided, CBC solver option 'sec' matches."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        recorded_sec = []
        solver_mod = sys.modules["romutil.solver"]
        orig_factory = solver_mod.SolverFactory

        def mock_factory(name, **kwargs):
            s = orig_factory(name, **kwargs)
            orig_solve = s.solve
            def wrapped_solve(m, **kw):
                recorded_sec.append(s.options.get("sec"))
                return orig_solve(m, **kw)
            s.solve = wrapped_solve
            return s

        monkeypatch.setattr(solver_mod, "SolverFactory", mock_factory)
        model, result = solve(rdb, exits, timeout=42)
        assert 42 in recorded_sec

    def test_solve_maxtimelimit_with_feasible_solution_terminates_gracefully(self, monkeypatch):
        """Solver loop terminates immediately and updates room coordinates on maxTimeLimit."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        solver_mod = sys.modules["romutil.solver"]
        monkeypatch.setattr(solver_mod, "non_euler", lambda rdb, exits: MagicMock(cut={0: MagicMock(value=0), 1: MagicMock(value=0)}))

        mock_solver = MagicMock()
        mock_result = MockSolverResults(pyomo.opt.TerminationCondition.maxTimeLimit)

        def mock_solve_call(m, **kwargs):
            m.x[1].value = 10
            m.x[2].value = 20
            m.y[1].value = 5
            m.y[2].value = 5
            m.z[1].value = 0
            m.z[2].value = 0
            return mock_result

        mock_solver.solve = mock_solve_call
        mock_solver.options = {}
        monkeypatch.setattr(solver_mod, "SolverFactory", lambda name, **kwargs: mock_solver)

        mock_plot = MagicMock()
        monkeypatch.setattr(solver_mod.Plotter, "plot", mock_plot)

        m, res = solve(rdb, exits, timeout=15)
        assert res.solver.termination_condition == pyomo.opt.TerminationCondition.maxTimeLimit
        assert rdb[1].x == 10
        assert rdb[2].x == 20
        mock_plot.assert_called()

    def test_solve_maxtimelimit_without_feasible_solution(self, monkeypatch):
        """Solver loop returns (m, result) without plotting when coordinates are None."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        solver_mod = sys.modules["romutil.solver"]
        monkeypatch.setattr(solver_mod, "non_euler", lambda rdb, exits: MagicMock(cut={0: MagicMock(value=0), 1: MagicMock(value=0)}))

        mock_solver = MagicMock()
        mock_result = MockSolverResults(pyomo.opt.TerminationCondition.maxTimeLimit)

        def mock_solve_call(m, **kwargs):
            m.x[1].value = None
            m.x[2].value = None
            return mock_result

        mock_solver.solve = mock_solve_call
        mock_solver.options = {}
        monkeypatch.setattr(solver_mod, "SolverFactory", lambda name, **kwargs: mock_solver)

        mock_plot = MagicMock()
        monkeypatch.setattr(solver_mod.Plotter, "plot", mock_plot)

        m, res = solve(rdb, exits, timeout=10)
        assert res.solver.termination_condition == pyomo.opt.TerminationCondition.maxTimeLimit
        mock_plot.assert_not_called()

    def test_solve_infeasible_terminates_immediately(self, monkeypatch):
        """Solver terminates immediately on infeasible result."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        solver_mod = sys.modules["romutil.solver"]
        monkeypatch.setattr(solver_mod, "non_euler", lambda rdb, exits: MagicMock(cut={0: MagicMock(value=0), 1: MagicMock(value=0)}))

        mock_solver = MagicMock()
        mock_result = MockSolverResults(pyomo.opt.TerminationCondition.infeasible)
        mock_solver.solve.return_value = mock_result
        mock_solver.options = {}
        monkeypatch.setattr(solver_mod, "SolverFactory", lambda name, **kwargs: mock_solver)

        m, res = solve(rdb, exits)
        assert res.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible


class TestFeasibleCoordinatesHelper:
    """Unit tests for _has_feasible_coordinates."""

    def test_helper_with_none_or_missing_attributes(self):
        assert _has_feasible_coordinates(None, {1: MagicMock()}) is False
        assert _has_feasible_coordinates(object(), {1: MagicMock()}) is False

    def test_helper_with_empty_rdb(self):
        model = MockModelCoords({1: (0, 0, 0)})
        assert _has_feasible_coordinates(model, {}) is False

    def test_helper_with_none_values(self):
        model = MockModelCoords({1: (0, None, 0)})
        assert _has_feasible_coordinates(model, {1: MagicMock()}) is False

    def test_helper_with_valid_coordinates(self):
        model = MockModelCoords({1: (1, 2, 3), 2: (4, 5, 6)})
        assert _has_feasible_coordinates(model, {1: MagicMock(), 2: MagicMock()}) is True

    def test_helper_with_missing_key(self):
        model = MockModelCoords({1: (1, 2, 3)})
        assert _has_feasible_coordinates(model, {1: MagicMock(), 2: MagicMock()}) is False


class TestSolverRecoveryInGraph:
    """Unit tests for solve_layout recovery under various termination conditions."""

    def test_solve_layout_optimal_normalizes_and_logs(self, monkeypatch, caplog):
        """Optimal solve normalizes coordinates and logs completion."""
        r1 = Room(RoomDef(vnum=10, name="R10", description="Desc", exits=(ExitDef(direction=1, dst_vnum=20),)))
        r2 = Room(RoomDef(vnum=20, name="R20", description="Desc", exits=(ExitDef(direction=3, dst_vnum=10),)))
        rdb = {10: r1, 20: r2}

        mock_model = MockModelCoords({10: (5, 5, 0), 20: (10, 5, 0)}, {0: 0, 1: 0})
        mock_results = MockSolverResults(pyomo.opt.TerminationCondition.optimal)

        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve", lambda rdb, exits, timeout=None: (mock_model, mock_results))

        with caplog.at_level(logging.INFO):
            solved_rdb, exits = solve_layout(rdb, AreaHeader("test.are", "TestArea", "B", 10, 20))

        assert "Solve completed" in caplog.text
        assert solved_rdb[10].x == 0
        assert solved_rdb[20].x == 5

    def test_solve_layout_maxtimelimit_feasible_recovery(self, monkeypatch, caplog):
        """maxTimeLimit with feasible coordinates retains coordinates instead of resetting to 0."""
        r1 = Room(RoomDef(vnum=101, name="R101", description="Desc", exits=(ExitDef(direction=1, dst_vnum=102),)))
        r2 = Room(RoomDef(vnum=102, name="R102", description="Desc", exits=(ExitDef(direction=3, dst_vnum=101),)))
        rdb = {101: r1, 102: r2}

        mock_model = MockModelCoords({101: (15, 20, 2), 102: (25, 20, 2)}, {0: 0, 1: 0})
        mock_results = MockSolverResults(pyomo.opt.TerminationCondition.maxTimeLimit)

        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve", lambda rdb, exits, timeout=None: (mock_model, mock_results))

        with caplog.at_level(logging.WARNING):
            solved_rdb, exits = solve_layout(rdb, AreaHeader("test.are", "TimeoutArea", "B", 101, 102))

        assert "Solver reached time limit; using best feasible layout" in caplog.text
        assert solved_rdb[101].x == 0
        assert solved_rdb[102].x == 10
        assert solved_rdb[101].y == 0
        assert solved_rdb[102].y == 0
        assert solved_rdb[101].z == 0
        assert solved_rdb[102].z == 0

    def test_solve_layout_feasible_status_recovery(self, monkeypatch, caplog):
        """feasible termination condition retains coordinates and logs warning."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}

        mock_model = MockModelCoords({1: (2, 3, 0), 2: (5, 3, 0)}, {0: 0, 1: 0})
        mock_results = MockSolverResults(pyomo.opt.TerminationCondition.feasible)

        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve", lambda rdb, exits, timeout=None: (mock_model, mock_results))

        with caplog.at_level(logging.WARNING):
            solved_rdb, exits = solve_layout(rdb, "FeasibleArea")

        assert "Solver reached time limit; using best feasible layout" in caplog.text
        assert solved_rdb[1].x == 0
        assert solved_rdb[2].x == 3

    def test_solve_layout_maxtimelimit_without_feasible_coords_falls_back(self, monkeypatch, caplog):
        """maxTimeLimit without feasible coordinates falls back to zeroing coordinates."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}

        mock_model = MockModelCoords({1: (None, None, None), 2: (None, None, None)})
        mock_results = MockSolverResults(pyomo.opt.TerminationCondition.maxTimeLimit)

        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve", lambda rdb, exits, timeout=None: (mock_model, mock_results))

        with caplog.at_level(logging.ERROR):
            solved_rdb, exits = solve_layout(rdb, "EmptyTimeout")

        assert "Solver failed!" in caplog.text
        assert solved_rdb[1].x == 0
        assert solved_rdb[1].y == 0
        assert solved_rdb[2].x == 0
        assert solved_rdb[2].y == 0

    def test_solve_layout_infeasible_status_falls_back(self, monkeypatch, caplog):
        """infeasible solver condition falls back to zeroing coordinates."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        rdb = {1: r1}

        mock_results = MockSolverResults(pyomo.opt.TerminationCondition.infeasible)
        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve", lambda rdb, exits, timeout=None: (None, mock_results))

        with caplog.at_level(logging.ERROR):
            solved_rdb, exits = solve_layout(rdb, "InfeasibleArea")

        assert "Solver failed!" in caplog.text
        assert solved_rdb[1].x == 0

    def test_solve_layout_timeout_passed_to_solver(self, monkeypatch):
        """solver_timeout is passed from solve_layout to solve()."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}

        passed_timeout = []
        def mock_solve(rdb, exits, timeout=None):
            passed_timeout.append(timeout)
            return MockModelCoords({1: (0, 0, 0), 2: (1, 0, 0)}), MockSolverResults(pyomo.opt.TerminationCondition.optimal)

        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve", mock_solve)
        solve_layout(rdb, solver_timeout=77)
        assert passed_timeout == [77]

    def test_graph_timeout_passed_to_solve_layout(self, monkeypatch):
        """solver_timeout is forwarded from graph() to solve_layout()."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=1, dst_vnum=2),)))
        rdb = {1: r1}

        passed_timeout = []
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeout.append(solver_timeout)
            return rdb, []

        graph_mod = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_mod, "solve_layout", mock_solve_layout)
        graph(rdb, "out.svg", None, solver_timeout=99)
        assert passed_timeout == [99]


class TestCliSolverTimeout:
    """Unit and integration tests for --solver-timeout CLI flag."""

    def test_cli_argument_parsing_with_timeout(self, monkeypatch):
        """--solver-timeout parses integer correctly."""
        monkeypatch.setattr("sys.argv", ["romutil", "dummy.are", "--solver-timeout", "120"])
        captured_args = []

        def mock_main(areas, outbase, split_levels=False, fmt="svg", solver_timeout=None, **kwargs):
            captured_args.append((areas, solver_timeout))

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "main", mock_main)
        cli()
        assert len(captured_args) == 1
        assert captured_args[0][1] == 120

    def test_cli_argument_parsing_default_none(self, monkeypatch):
        """Default without --solver-timeout is None."""
        monkeypatch.setattr("sys.argv", ["romutil", "dummy.are"])
        captured_args = []

        def mock_main(areas, outbase, split_levels=False, fmt="svg", solver_timeout=None, **kwargs):
            captured_args.append(solver_timeout)

        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod, "main", mock_main)
        cli()
        assert captured_args == [None]

    def test_main_svg_passes_solver_timeout_to_graph(self, tmp_path, monkeypatch):
        """main() forwards solver_timeout to graph() for SVG format."""
        are_file = tmp_path / "dummy.are"
        are_file.write_text("dummy")

        mock_area = AreaData(
            header=AreaHeader("dummy.are", "Dummy", "Builder", 1, 2),
            rooms=[
                RoomDef(1, "R1", "D1", exits=(ExitDef(direction=1, dst_vnum=2),)),
                RoomDef(2, "R2", "D2", exits=(ExitDef(direction=3, dst_vnum=1),)),
            ]
        )
        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod.Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        def mock_graph(rdb, name, area, split_levels=False, outbase=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)

        monkeypatch.setattr(cli_mod, "graph", mock_graph)
        with pytest.raises(SystemExit):
            main([are_file], str(tmp_path / "out"), fmt="svg", solver_timeout=45)
        assert 45 in passed_timeouts

    def test_main_json_passes_solver_timeout_to_solve_layout(self, tmp_path, monkeypatch):
        """main() forwards solver_timeout to solve_layout() for JSON format."""
        are_file = tmp_path / "dummy.are"
        are_file.write_text("dummy")

        mock_area = AreaData(
            header=AreaHeader("dummy.are", "Dummy", "Builder", 1, 2),
            rooms=[
                RoomDef(1, "R1", "D1", exits=(ExitDef(direction=1, dst_vnum=2),)),
                RoomDef(2, "R2", "D2", exits=(ExitDef(direction=3, dst_vnum=1),)),
            ]
        )
        cli_mod = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_mod.Parser, "parse", lambda self, text: mock_area)

        passed_timeouts = []
        orig_solve_layout = solve_layout
        def mock_solve_layout(rdb, area=None, solver_timeout=None):
            passed_timeouts.append(solver_timeout)
            return orig_solve_layout(rdb, area, solver_timeout=solver_timeout)

        monkeypatch.setattr(cli_mod, "solve_layout", mock_solve_layout)
        with pytest.raises(SystemExit):
            main([are_file], str(tmp_path / "out"), fmt="json", solver_timeout=88)
        assert 88 in passed_timeouts

    def test_real_cbc_solve_with_timeout(self, tmp_path):
        """Integration test with real CBC solver passing a timeout."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc 1", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc 2", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        solved_rdb, exits = solve_layout(rdb, AreaHeader("test.are", "RealSolve", "Builder", 1, 2), solver_timeout=5)
        assert solved_rdb[1].x is not None
        assert solved_rdb[2].x is not None
        assert solved_rdb[1].x != solved_rdb[2].x
