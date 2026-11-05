"""Stage 4: Solve and extraction for layout cluster MILP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.milp.formulation import ClusterFormulation
from romutil.types import LayoutOptions, LayoutSolverError, SolverStatus


@dataclass(frozen=True)
class SolvedClusterExtraction:
    """Solved unnormalized room coordinates and child offsets."""

    raw_parent_coords: Mapping[int, tuple[int, int, int]]
    raw_child_offsets: Mapping[str, tuple[int, int, int]]
    solver_status: SolverStatus
    raw_port_coords: Mapping[int, tuple[int, int, int]] = field(default_factory=dict)


def _is_valid_solver_status(results: Any) -> bool:
    """Check if Pyomo solver results represent an optimal, feasible, or bounded timeout state."""
    if results is None or not hasattr(results, "solver"):
        return False
    return bool(
        (
            results.solver.status == PyomoSolverStatus.ok
            and results.solver.termination_condition
            in (
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            )
        )
        or (
            results.solver.termination_condition
            in (
                TerminationCondition.maxTimeLimit,
                TerminationCondition.intermediateNonInteger,
                TerminationCondition.other,
            )
            and results.solver.status
            in (
                PyomoSolverStatus.ok,
                PyomoSolverStatus.aborted,
                PyomoSolverStatus.warning,
            )
        )
    )


def solve_and_extract(
    formulation: ClusterFormulation,
    solver_name: str = "cbc",
    options: LayoutOptions | None = None,
    max_solutions: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
    warmstart: bool = False,
) -> SolvedClusterExtraction:
    """Invoke the MILP solver and extract raw coordinates and child offsets.

    Handles solver availability, timeouts, infeasibility, and value extraction errors,
    raising LayoutSolverError on non-optimal solver exits or missing variables.

    Args:
        formulation: ClusterFormulation containing the configured Pyomo model.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').
        options: Optional layout configuration parameters (e.g. timeout_seconds).
        max_solutions: Optional limit on the number of integer solutions before early
            termination (e.g. 1 for intermediate cutting-plane passes).
        solver_options: Optional mapping of solver-specific configuration options passed
            directly to the solver.
        warmstart: Whether to use warm-start initialization with current variable values.
            Only solvers that support warm-starting (e.g. CBC) will utilize this option;
            solvers without native warm-start support fall back to standard solve.

    Returns:
        SolvedClusterExtraction containing raw coordinates and child offsets.

    Raises:
        LayoutSolverError: If solver is unavailable, infeasible, or coordinate extraction fails.
    """
    node = formulation.inventory.node
    rooms = formulation.inventory.rooms
    children = formulation.inventory.children
    child_ids = [c.node_id for c in children]

    try:
        solver = pyo.SolverFactory(solver_name)
        if not solver.available():
            raise LayoutSolverError(f"MILP solver '{solver_name}' is not available.")
    except Exception as exc:
        if isinstance(exc, LayoutSolverError):
            raise
        raise LayoutSolverError(f"Failed to initialize MILP solver '{solver_name}': {exc}") from exc

    if solver_name == "cbc":
        solver.options["timeMode"] = "elapsed"

    if options and options.timeout_seconds:
        if solver_name == "cbc":
            solver.options["seconds"] = options.timeout_seconds
        elif solver_name == "glpk":
            solver.options["tmlim"] = options.timeout_seconds

    if max_solutions is not None:
        if solver_name == "cbc":
            solver.options["maxSolutions"] = max_solutions

    if solver_options:
        for opt_key, opt_val in solver_options.items():
            solver.options[opt_key] = opt_val

    supports_warmstart = (
        solver_name == "cbc"
        or (
            hasattr(solver, "warm_start_capable")
            and callable(solver.warm_start_capable)
            and solver.warm_start_capable()
        )
    )

    results = None
    if warmstart and supports_warmstart:
        try:
            results = solver.solve(formulation.model, warmstart=True, tee=False)
            if not _is_valid_solver_status(results):
                results = solver.solve(formulation.model, tee=False)
        except Exception:
            results = solver.solve(formulation.model, tee=False)
    else:
        results = solver.solve(formulation.model, tee=False)

    if not _is_valid_solver_status(results):
        raise LayoutSolverError(
            f"MILP solver returned non-optimal status: {results.solver.status}, "
            f"termination condition: {results.solver.termination_condition} for cluster '{node.node_id}'."
        )

    raw_parent_coords: dict[int, tuple[int, int, int]] = {}
    for u in rooms:
        vx = pyo.value(formulation.model.x[u])
        vy = pyo.value(formulation.model.y[u])
        vz = pyo.value(formulation.model.z[u])
        if vx is None or vy is None or vz is None:
            raise LayoutSolverError(
                f"MILP solver failed to retrieve coordinate values for room {u} in cluster '{node.node_id}'."
            )
        raw_parent_coords[u] = (int(round(vx)), int(round(vy)), int(round(vz)))

    raw_port_coords: dict[int, tuple[int, int, int]] = {}
    if hasattr(formulation.model, "x"):
        for u in formulation.inventory.boundary_ports:
            if u in formulation.model.x:
                vx = pyo.value(formulation.model.x[u])
                vy = pyo.value(formulation.model.y[u])
                vz = pyo.value(formulation.model.z[u])
                if vx is None or vy is None or vz is None:
                    raise LayoutSolverError(
                        f"MILP solver failed to retrieve coordinate values for boundary port {u} in cluster '{node.node_id}'."
                    )
                raw_port_coords[u] = (int(round(vx)), int(round(vy)), int(round(vz)))

    raw_child_offsets: dict[str, tuple[int, int, int]] = {}
    for cid in child_ids:
        vcx = pyo.value(formulation.model.cx[cid])
        vcy = pyo.value(formulation.model.cy[cid])
        vcz = pyo.value(formulation.model.cz[cid])
        if vcx is None or vcy is None or vcz is None:
            raise LayoutSolverError(
                f"MILP solver failed to retrieve offset values for child '{cid}' in cluster '{node.node_id}'."
            )
        raw_child_offsets[cid] = (int(round(vcx)), int(round(vcy)), int(round(vcz)))

    solver_status = SolverStatus.FEASIBLE
    if results.solver.termination_condition == TerminationCondition.optimal:
        solver_status = SolverStatus.OPTIMAL
    elif (
        results.solver.termination_condition
        in (
            TerminationCondition.maxTimeLimit,
            TerminationCondition.intermediateNonInteger,
        )
        or (
            results.solver.status in (PyomoSolverStatus.aborted, PyomoSolverStatus.warning)
            and max_solutions is None
        )
        or (
            results.solver.termination_condition == TerminationCondition.other
            and max_solutions is None
        )
    ):
        solver_status = SolverStatus.TIMEOUT

    return SolvedClusterExtraction(
        raw_parent_coords=raw_parent_coords,
        raw_child_offsets=raw_child_offsets,
        solver_status=solver_status,
        raw_port_coords=raw_port_coords,
    )


__all__ = [
    "SolvedClusterExtraction",
    "solve_and_extract",
]
