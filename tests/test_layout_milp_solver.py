"""Unit tests for Stage 4: Solve and Extraction with warm-starting support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyomo.environ as pyo
import pytest
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.ir import LayoutEdge, LayoutGraph, NodeSolution, SolveTreeNode
from romutil.models import Direction, Room
from romutil.layout.milp.formulation import formulate_cluster_model
from romutil.layout.milp.inventory import inventory_cluster
from romutil.layout.milp.solver import (
    SolvedClusterExtraction,
    _is_valid_solver_status,
    solve_and_extract,
)
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus


def _make_graph(rooms: list[int], edges: list[tuple[int, int, Direction, int]]) -> LayoutGraph:
    """Helper to construct a LayoutGraph for testing."""
    room_map = {v: Room(vnum=v, name=f"Room {v}", description="", exits=()) for v in rooms}
    edge_objs = tuple(
        LayoutEdge(src=s, dst=d, direction=int(dirn), weight=dist, cut=False)
        for s, d, dirn, dist in edges
    )
    return LayoutGraph(rooms=room_map, edges=edge_objs)


class TestSolveAndExtract:
    """Unit tests verifying solve_and_extract behavior and warm-start execution."""

    def test_solve_and_extract_cold_and_warmstart_real_cbc(self) -> None:
        """CBC solver successfully solves cold and re-solves via warmstart with populated incumbent."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Initial cold solve
        ext_cold = solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=False)
        assert ext_cold.solver_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert ext_cold.raw_parent_coords[1] == (0, 0, 0)
        assert ext_cold.raw_parent_coords[2] == (1, 0, 0)

        # Warm-start solve using incumbent values populated on formulation.model
        ext_warm = solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=True)
        assert ext_warm.solver_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert ext_warm.raw_parent_coords[1] == (0, 0, 0)
        assert ext_warm.raw_parent_coords[2] == (1, 0, 0)

    def test_solve_and_extract_warmstart_flag_invoked_on_supporting_solver(self) -> None:
        """When warmstart=True on a supporting solver, solver.solve is called with warmstart=True."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.warm_start_capable.return_value = True
        mock_solver.options = {}

        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=0):
                # warmstart=True
                solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=True)
                assert mock_solver.solve.call_count == 1
                assert mock_solver.solve.call_args.kwargs.get("warmstart") is True

                mock_solver.solve.reset_mock()

                # warmstart=False
                solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=False)
                assert mock_solver.solve.call_count == 1
                assert "warmstart" not in mock_solver.solve.call_args.kwargs

    def test_solve_and_extract_warmstart_unsupported_solver_falls_back_to_cold_solve(self) -> None:
        """When warmstart=True on a solver that does not support warmstart, warmstart kwarg is omitted."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.warm_start_capable.return_value = False
        mock_solver.options = {}

        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=0):
                solve_and_extract(formulation=formulation, solver_name="glpk", warmstart=True)
                assert mock_solver.solve.call_count == 1
                assert "warmstart" not in mock_solver.solve.call_args.kwargs

    def test_solve_and_extract_warmstart_exception_falls_back_to_cold_start(self) -> None:
        """When solver.solve(..., warmstart=True) raises an exception, it falls back to cold solve."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.warm_start_capable.return_value = True
        mock_solver.options = {}

        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal

        # First call with warmstart raises, second call without warmstart succeeds
        mock_solver.solve.side_effect = [
            RuntimeError("CBC mipstart parse failure"),
            mock_results,
        ]

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=0):
                ext = solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=True)
                assert ext.solver_status == SolverStatus.OPTIMAL
                assert mock_solver.solve.call_count == 2
                assert mock_solver.solve.call_args_list[0].kwargs.get("warmstart") is True
                assert "warmstart" not in mock_solver.solve.call_args_list[1].kwargs

    def test_solve_and_extract_warmstart_invalid_status_falls_back_to_cold_start(self) -> None:
        """When solver.solve(..., warmstart=True) returns non-optimal status, falls back to cold solve."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.warm_start_capable.return_value = True
        mock_solver.options = {}

        bad_results = MagicMock()
        bad_results.solver.status = PyomoSolverStatus.error
        bad_results.solver.termination_condition = TerminationCondition.infeasible

        good_results = MagicMock()
        good_results.solver.status = PyomoSolverStatus.ok
        good_results.solver.termination_condition = TerminationCondition.optimal

        mock_solver.solve.side_effect = [bad_results, good_results]

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=0):
                ext = solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=True)
                assert ext.solver_status == SolverStatus.OPTIMAL
                assert mock_solver.solve.call_count == 2

    def test_solve_and_extract_both_warm_and_cold_fail_raises_error(self) -> None:
        """When both warm-start and cold-start return non-optimal status, LayoutSolverError is raised."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.warm_start_capable.return_value = True
        mock_solver.options = {}

        bad_results = MagicMock()
        bad_results.solver.status = PyomoSolverStatus.error
        bad_results.solver.termination_condition = TerminationCondition.infeasible

        mock_solver.solve.side_effect = [bad_results, bad_results]

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError, match="returned non-optimal status"):
                solve_and_extract(formulation=formulation, solver_name="cbc", warmstart=True)

    def test_solve_and_extract_boundary_ports_and_missing_values(self) -> None:
        """Extracts boundary port coordinates and raises LayoutSolverError if coordinates are None."""
        node = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset({1}))
        graph = _make_graph([1, 99], [(1, 99, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph, boundary_ports=frozenset({99}))
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.options = {}

        good_results = MagicMock()
        good_results.solver.status = PyomoSolverStatus.ok
        good_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = good_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            # Normal extraction of boundary ports
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 1, 0, 0]):
                ext = solve_and_extract(formulation=formulation, solver_name="cbc")
                assert 99 in ext.raw_port_coords
                assert ext.raw_port_coords[99] == (1, 0, 0)

            # Missing boundary port coordinate value raises LayoutSolverError
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, None, 0, 0]):
                with pytest.raises(LayoutSolverError, match="failed to retrieve coordinate values for boundary port 99"):
                    solve_and_extract(formulation=formulation, solver_name="cbc")

            # Missing room coordinate value raises LayoutSolverError
            with patch("pyomo.environ.value", side_effect=[None, 0, 0]):
                with pytest.raises(LayoutSolverError, match="failed to retrieve coordinate values for room 1"):
                    solve_and_extract(formulation=formulation, solver_name="cbc")

    def test_solve_and_extract_child_offsets_and_errors(self) -> None:
        """Extracts child offsets and raises LayoutSolverError if child offset is None."""
        child = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset({1}), children=(child,))
        graph = _make_graph([1, 10], [])
        c_sol = NodeSolution(node_id="c1", local_coordinates={10: Coord3D(0, 0, 0)})
        inv = inventory_cluster(node=parent, child_solutions={"c1": c_sol}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            # Normal extraction
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, 5, 6, 7]):
                ext = solve_and_extract(formulation=formulation, solver_name="cbc")
                assert ext.raw_child_offsets["c1"] == (5, 6, 7)

            # Missing child offset raises LayoutSolverError
            with patch("pyomo.environ.value", side_effect=[0, 0, 0, None, 6, 7]):
                with pytest.raises(LayoutSolverError, match="failed to retrieve offset values for child 'c1'"):
                    solve_and_extract(formulation=formulation, solver_name="cbc")

    def test_solve_and_extract_timeout_and_solver_options(self) -> None:
        """Configures solver timeout, maxSolutions, and custom solver_options across cbc and glpk."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.options = {}
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.maxTimeLimit
        mock_solver.solve.return_value = mock_results

        opts = LayoutOptions(timeout_seconds=25)
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=0):
                # cbc
                ext_cbc = solve_and_extract(
                    formulation=formulation,
                    solver_name="cbc",
                    options=opts,
                    max_solutions=2,
                    solver_options={"heuristics": "on"},
                )
                assert ext_cbc.solver_status == SolverStatus.TIMEOUT
                assert mock_solver.options.get("seconds") == 25
                assert mock_solver.options.get("maxSolutions") == 2
                assert mock_solver.options.get("heuristics") == "on"

                # glpk
                mock_solver.options.clear()
                ext_glpk = solve_and_extract(
                    formulation=formulation,
                    solver_name="glpk",
                    options=opts,
                )
                assert ext_glpk.solver_status == SolverStatus.TIMEOUT
                assert mock_solver.options.get("tmlim") == 25

    def test_solve_and_extract_solver_initialization_failure(self) -> None:
        """SolverFactory initialization failure raises LayoutSolverError."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = False
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError, match="is not available"):
                solve_and_extract(formulation=formulation, solver_name="cbc")

        with patch("pyomo.environ.SolverFactory", side_effect=Exception("Factory crashed")):
            with pytest.raises(LayoutSolverError, match="Failed to initialize MILP solver"):
                solve_and_extract(formulation=formulation, solver_name="cbc")

    def test_solve_and_extract_multithreading_crash_retries_single_threaded(self) -> None:
        """When multi-threaded CBC solver invocation crashes with an exception, it falls back to threads=1."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1}))
        graph = _make_graph([1], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.options = {"threads": 4}

        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.ok
        mock_results.solver.termination_condition = TerminationCondition.optimal

        # First call with threads=4 raises ApplicationError, second call with threads=1 succeeds
        mock_solver.solve.side_effect = [
            RuntimeError("ClpSimplexDual assertion failure"),
            mock_results,
        ]

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=0):
                ext = solve_and_extract(
                    formulation=formulation,
                    solver_name="cbc",
                    solver_options={"threads": 4},
                )
                assert ext.solver_status == SolverStatus.OPTIMAL
                assert mock_solver.solve.call_count == 2
                assert mock_solver.options["threads"] == 1


class TestStatusValidation:
    """Unit tests for _is_valid_solver_status helper."""

    def test_is_valid_solver_status_variations(self) -> None:
        """Validates optimal, feasible, timeout, and handles None or malformed objects."""
        assert _is_valid_solver_status(None) is False
        assert _is_valid_solver_status("invalid") is False

        mock_res = MagicMock()
        mock_res.solver.status = PyomoSolverStatus.ok
        mock_res.solver.termination_condition = TerminationCondition.optimal
        assert _is_valid_solver_status(mock_res) is True

        mock_res.solver.termination_condition = TerminationCondition.feasible
        assert _is_valid_solver_status(mock_res) is True

        mock_res.solver.status = PyomoSolverStatus.aborted
        mock_res.solver.termination_condition = TerminationCondition.maxTimeLimit
        assert _is_valid_solver_status(mock_res) is True

        mock_res.solver.status = PyomoSolverStatus.error
        mock_res.solver.termination_condition = TerminationCondition.infeasible
        assert _is_valid_solver_status(mock_res) is False
