"""Modular 5-stage MILP cluster layout solver."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import pyomo.environ as pyo

from romutil.layout.ir import LayoutGraph, NodeSolution, SolveTreeNode, TreeNodeType
from romutil.layout.milp.clearance import ClearanceViolation, detect_cluster_collisions
from romutil.layout.milp.formulation import ClusterFormulation, formulate_cluster_model
from romutil.layout.milp.inventory import (
    ActiveEdge,
    ClusterInventory,
    get_child_boundary_ports,
    inventory_cluster,
)
from romutil.layout.milp.normalization import (
    normalize_cluster_solution,
    normalize_empty_cluster,
    normalize_single_room_cluster,
    pack_world_root_components,
)
from romutil.layout.milp.solver import (
    LayoutInfeasibleError,
    SolvedClusterExtraction,
    solve_and_extract,
)
from romutil.types import LayoutOptions, LayoutSolverError, SolverStatus


class ClusterSolvePhase(str, Enum):
    """Execution phases of the cutting-plane clearance state machine."""

    PHYSICAL_CLEARANCE = "physical_clearance"
    CROSSING_CLEARANCE = "crossing_clearance"
    CONVERGENCE = "convergence"
    DONE = "done"


class UnresolvableCollisionError(LayoutSolverError):
    """Raised when cutting-plane clearance encounters unresolvable geometric collisions."""

    def __init__(
        self,
        message: str,
        *,
        node_id: str = "",
        collisions: Sequence[ClearanceViolation] = (),
        colliding_entities: Sequence[tuple[str, Any]] = (),
        pruned_reasons: Sequence[Mapping[str, str]] = (),
        inventory: ClusterInventory | None = None,
    ) -> None:
        super().__init__(message)
        self.node_id = node_id
        self.collisions = tuple(collisions)
        self.colliding_entities = tuple(colliding_entities)
        self.pruned_reasons = tuple(dict(reason) for reason in pruned_reasons)
        self.inventory = inventory


def _extract_colliding_entities(
    collisions: Sequence[ClearanceViolation],
    active_edges: Sequence[ActiveEdge],
) -> tuple[tuple[str, Any], ...]:
    """Extract deduplicated colliding macro entities (including corridor children)."""
    seen: set[tuple[str, Any]] = set()
    ordered: list[tuple[str, Any]] = []

    def _add(ent: tuple[str, Any]) -> None:
        if ent not in seen:
            seen.add(ent)
            ordered.append(ent)

    for viol in collisions:
        for item in (viol.item_a, viol.item_b):
            if isinstance(item, int):
                if viol.category in ("corridor_room", "corridor_child", "corridor_crossing"):
                    _add(("corridor", item))
                    if 0 <= item < len(active_edges):
                        for cid in sorted(active_edges[item].associated_children):
                            _add(("child", cid))
                else:
                    _add(("room", item))
            elif hasattr(item, "vnum"):
                _add(("room", item.vnum))
            elif hasattr(item, "child_id"):
                _add(("child", item.child_id))
            elif hasattr(item, "edge_index"):
                idx = item.edge_index
                _add(("corridor", idx))
                if 0 <= idx < len(active_edges):
                    for cid in sorted(active_edges[idx].associated_children):
                        _add(("child", cid))
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                if item[0] == "edge":
                    _add(("corridor", item[1]))
                    if isinstance(item[1], int) and 0 <= item[1] < len(active_edges):
                        for cid in sorted(active_edges[item[1]].associated_children):
                            _add(("child", cid))
                else:
                    _add(("child", item[0]))
    return tuple(ordered)


def _raise_collision_error(
    formulation: ClusterFormulation,
    node_id: str,
    message: str,
    fallback_collisions: Sequence[ClearanceViolation] = (),
) -> NoReturn:
    """Raise UnresolvableCollisionError if unsatisfiable or fallback collisions exist, else LayoutSolverError."""
    cb = getattr(formulation, "clearance_builder", None)
    raw_unsat = getattr(cb, "unsatisfiable_collisions", ())
    unsat = tuple(raw_unsat) if isinstance(raw_unsat, (list, tuple)) else ()
    collisions = unsat if unsat else tuple(fallback_collisions)
    if collisions:
        raw_reasons = getattr(cb, "unsatisfiable_collision_reasons", ()) if unsat else ()
        reasons = tuple(raw_reasons) if isinstance(raw_reasons, (list, tuple)) else ()
        inv = getattr(formulation, "inventory", None)
        active_edges = getattr(inv, "active_edges", ()) if isinstance(inv, ClusterInventory) else ()
        entities = _extract_colliding_entities(collisions, active_edges)
        raise UnresolvableCollisionError(
            message,
            node_id=node_id,
            collisions=collisions,
            colliding_entities=entities,
            pruned_reasons=reasons,
            inventory=inv if isinstance(inv, ClusterInventory) else None,
        )
    raise LayoutSolverError(message)


def compute_adaptive_cluster_timeout(
    node: SolveTreeNode,
    options: LayoutOptions | None = None,
    *,
    n_sub_boxes: int = 0,
) -> int:
    """Compute adaptive solver timeout limit scaled to cluster entity complexity.

    Scales linearly with the total count of constituent rooms and child complexity
    (taking into account decomposed child sub-box counts) beyond a 20-entity baseline
    to ensure large composite clusters receive sufficient solver budget for integer
    convergence.

    Args:
        node: SolveTreeNode representing the cluster to lay out.
        options: Optional layout configuration parameters containing base timeout.
        n_sub_boxes: Total count of decomposed child keep-out sub-boxes in the cluster.

    Returns:
        Adaptive timeout duration in seconds.
    """
    n_entities = len(node.rooms) + max(len(node.children), n_sub_boxes // 4)
    base_timeout = (
        options.timeout_seconds
        if options is not None and options.timeout_seconds
        else 10
    )
    return max(base_timeout, int(base_timeout * max(1.0, n_entities / 20.0)))


def _has_populated_warmstart_coordinates(formulation: ClusterFormulation) -> bool:
    """Verify that model coordinate and edge length variables have populated values."""
    model = getattr(formulation, "model", None)
    if model is None:
        return False
    var_names = ("x", "y", "z", "cx", "cy", "cz", "L")
    for name in var_names:
        if not hasattr(model, name):
            continue
        var_component = getattr(model, name)
        if not hasattr(var_component, "values") or not callable(var_component.values):
            continue
        try:
            for var_data in var_component.values():
                if getattr(var_data, "value", None) is None:
                    return False
        except (TypeError, AttributeError):
            continue
    return True


def _is_compacted_active_edge(ae: ActiveEdge) -> bool:
    """Return True if ae represents a compacted multi-room hallway."""
    flag = getattr(ae, "is_compacted", None)
    if flag is not None:
        return bool(flag)
    return ae.min_length > 1


def _pendant_gateway_edge_indices(active_edges: Sequence[ActiveEdge]) -> frozenset[int]:
    """Return active-edge indices of gateway bridges attached to a degree-1 pendant child."""
    child_edge_counts: dict[str, int] = {}
    for ae in active_edges:
        for cid in ae.associated_children:
            child_edge_counts[cid] = child_edge_counts.get(cid, 0) + 1
    return frozenset(
        i
        for i, ae in enumerate(active_edges)
        if ae.edge_id.startswith("gw_")
        and any(child_edge_counts.get(cid, 0) == 1 for cid in ae.associated_children)
    )


def _compacted_crossings(
    crossings: Sequence[ClearanceViolation],
    active_edges: Sequence[ActiveEdge],
) -> list[ClearanceViolation]:
    """Return crossing violations involving a compacted hallway or pendant gateway bridge."""
    pendant_gw_indices = _pendant_gateway_edge_indices(active_edges)
    return [
        viol
        for viol in crossings
        if any(
            isinstance(idx, int)
            and 0 <= idx < len(active_edges)
            and (_is_compacted_active_edge(active_edges[idx]) or idx in pendant_gw_indices)
            for idx in (viol.item_a, viol.item_b)
        )
    ]


def _has_compacted_crossing(
    crossings: Sequence[ClearanceViolation],
    active_edges: Sequence[ActiveEdge],
) -> bool:
    """Return True if any crossing violation involves a compacted hallway or pendant gateway bridge."""
    return bool(_compacted_crossings(crossings, active_edges))


def _extract_objective_value(formulation: Any) -> float | None:
    """Return the scalar objective value from a solved ClusterFormulation, if available."""
    if not isinstance(formulation, ClusterFormulation):
        return None
    model = getattr(formulation, "model", None)
    if model is None or not hasattr(model, "obj"):
        return None
    val = pyo.value(model.obj, exception=False)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _select_verified_incumbent(
    best_clean_extraction: SolvedClusterExtraction | None,
    best_physical_extraction: SolvedClusterExtraction | None,
) -> SolvedClusterExtraction | None:
    """Return the highest-quality verified incumbent, preferring clean over physical-only."""
    if best_clean_extraction is not None:
        return best_clean_extraction
    return best_physical_extraction


def _select_better_incumbent(
    current: SolvedClusterExtraction | None,
    current_obj: float | None,
    candidate: SolvedClusterExtraction,
    candidate_obj: float | None,
) -> tuple[SolvedClusterExtraction, float | None]:
    """Return the preferred incumbent and objective, breaking ties toward the candidate."""
    if (
        current is None
        or candidate_obj is None
        or current_obj is None
        or candidate_obj <= current_obj
    ):
        return candidate, candidate_obj
    return current, current_obj


def _detect_and_partition_violations(
    formulation: ClusterFormulation,
    extraction: SolvedClusterExtraction,
) -> tuple[list[ClearanceViolation], list[ClearanceViolation], list[ClearanceViolation]]:
    """Detect violations and partition them into physical, all crossings, and mandatory crossings."""
    violations = detect_cluster_collisions(
        formulation=formulation,
        extraction=extraction,
    )
    physical = [v for v in violations if v.category != "corridor_crossing"]
    crossings = [v for v in violations if v.category == "corridor_crossing"]
    compacted = _compacted_crossings(crossings, formulation.inventory.active_edges)
    return physical, crossings, compacted


def _build_cbc_solver_options(ratio_gap: float, threads: int) -> dict[str, Any]:
    """Build standard CBC solver options for a cutting-plane or convergence pass."""
    return {
        "heuristics": "on",
        "ratioGap": ratio_gap,
        "threads": threads,
        "timeMode": "elapsed",
    }


def _run_convergence_solve(
    formulation: ClusterFormulation,
    options: LayoutOptions | None,
    solver_name: str,
    ratio_gap: float,
    remaining_budget: float,
) -> SolvedClusterExtraction:
    """Execute a full-budget convergence MILP pass warm-started from current coordinates."""
    conv_timeout = max(1, int(round(remaining_budget)))
    conv_options = (
        replace(options, timeout_seconds=conv_timeout)
        if options is not None
        else LayoutOptions(timeout_seconds=conv_timeout)
    )
    conv_solver_options = (
        _build_cbc_solver_options(
            ratio_gap=ratio_gap,
            threads=min(4, os.cpu_count() or 1),
        )
        if solver_name == "cbc"
        else {}
    )
    conv_warmstart = (
        getattr(options, "warmstart", False)
        and _has_populated_warmstart_coordinates(formulation)
    )
    return solve_and_extract(
        formulation=formulation,
        solver_name=solver_name,
        options=conv_options,
        max_solutions=None,
        solver_options=conv_solver_options if conv_solver_options else None,
        warmstart=conv_warmstart,
    )


def solve_cluster_with_cutting_planes(
    formulation: ClusterFormulation,
    node: SolveTreeNode,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
    max_iterations: int = 50,
    total_cluster_budget: float | None = None,
    max_solutions: int | None = None,
) -> SolvedClusterExtraction:
    """Iteratively solve cluster MILP formulation with a cutting-plane clearance state machine.

    Coordinates an explicit 4-phase state machine:
      1. PHYSICAL_CLEARANCE: When physical room or macro-box collisions exist, inject disjunctive
         clearance cuts for physical collisions alongside mandatory crossings (compacted multi-room
         hallways and degree-1 pendant gateway bridges) at lower priority tier (tier 5 inside
         ``inject_clearance_cuts``), while deferring non-mandatory unit-edge crossings to
         CROSSING_CLEARANCE, using intermediate fast passes.
      2. CROSSING_CLEARANCE: When all physical collisions are cleared but planar corridor crossings
         remain, first resolve mandatory crossings. Before injecting optional unit-edge crossing
         cuts, if the latest physically clean layout came from a relaxed intermediate pass
         (``effective_inter_rgap > ratio_gap``, ``TIMEOUT``, or ``max_solutions`` limit), runs a
         pre-unit-crossing CONVERGENCE optimization at ``ratio_gap`` to tighten coordinates and
         clear transient crossings without risky crossing cuts; any newly revealed physical or
         mandatory crossing violations immediately re-enter clearance. Optional unit-edge crossing
         cuts are injected only when ``remaining_budget >= min(15.0, total_cluster_budget * 0.75)``
         so low-budget models are never mutated with potentially infeasible cuts. If resolving
         crossings re-introduces physical collisions, immediately falls back to PHYSICAL_CLEARANCE
         (self-healing).
      3. CONVERGENCE: When both physical collisions and corridor crossings are zero, runs a final
         convergence optimization pass warm-started from the clean incumbent using the full
         remaining cluster budget and no early solution limits. If violations reappear during
         convergence, falls back to the appropriate clearance phase.
      4. DONE: Reached when an extraction is verified clean or a verified clean/physical
         incumbent is preserved under budget exhaustion or non-planar unit-edge crossings.

    For medium clusters (``<= 50`` entities), intermediate CBC cutting-plane passes run
    single-threaded to reduce multi-threaded branch-and-bound overhead, while Pass 0,
    CONVERGENCE, and larger clusters use up to 4 threads. When
    ``options.intermediate_ratio_gap`` is unspecified on ``> 20``-entity clusters, the
    intermediate ``ratioGap`` adaptively relaxes after intermediate ``TIMEOUT`` extractions
    (``0.20 -> 0.50 -> 0.80``) while keeping ``ratio_gap`` on Pass 0 and CONVERGENCE. When an
    intermediate retry escalates ``pass_timeout`` without an explicit
    ``options.intermediate_pass_timeout``, ``intermediate_timeout`` escalates monotonically
    across subsequent passes (raising to at least ``15.0``s and up to ``30.0``s without
    shrinking larger initial timeouts). Verified clean and physical incumbents track scalar
    objective values via ``_extract_objective_value`` so strictly worse timed-out passes do not
    displace better incumbents.

    Aggregate cluster budget is tracked monotonically across all iterations. If the budget
    expires or a tail-budget intermediate failure occurs with a verified incumbent available,
    returns that incumbent; otherwise raises LayoutSolverError.

    Args:
        formulation: ClusterFormulation containing the Pyomo model and clearance builder.
        node: SolveTreeNode representing the cluster to lay out.
        options: Optional layout configuration parameters containing base timeout.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').
        max_iterations: Maximum cutting-plane refinement iterations (defaults to 50).
        total_cluster_budget: Optional explicit aggregate cluster budget in seconds.
            If None, computed dynamically via compute_adaptive_cluster_timeout.
        max_solutions: Optional limit on the number of solutions found on intermediate passes.

    Returns:
        SolvedClusterExtraction containing valid coordinates and child offsets with zero violations.

    Raises:
        LayoutSolverError: If geometric collisions or crossings cannot be resolved within budget or iterations.
    """
    if total_cluster_budget is None:
        total_cluster_budget = float(compute_adaptive_cluster_timeout(node, options))
    unit_crossing_budget_floor = min(15.0, total_cluster_budget * 0.75)
    t_start = time.monotonic()
    n_entities = len(node.rooms) + len(node.children)
    base_timeout = float(options.timeout_seconds) if options is not None and options.timeout_seconds else 15.0
    has_explicit_inter_timeout = bool(
        options is not None
        and getattr(options, "intermediate_pass_timeout", None)
        and options.intermediate_pass_timeout != 15
    )
    intermediate_timeout = (
        float(options.intermediate_pass_timeout)
        if has_explicit_inter_timeout and options is not None
        else (8.0 if n_entities > 20 else max(15.0, min(60.0, base_timeout * 0.67)))
    )
    ratio_gap = (
        float(options.ratio_gap)
        if options is not None and getattr(options, "ratio_gap", None)
        else 0.05
    )
    raw_inter_rgap = getattr(options, "intermediate_ratio_gap", None) if options is not None else None
    intermediate_ratio_gap = (
        float(raw_inter_rgap)
        if raw_inter_rgap is not None
        else (max(ratio_gap, 0.20) if n_entities > 20 else ratio_gap)
    )
    effective_max_solutions = (
        max_solutions
        if max_solutions is not None
        else (options.max_solutions if options is not None else None)
    )

    phase: ClusterSolvePhase = ClusterSolvePhase.PHYSICAL_CLEARANCE
    extraction: SolvedClusterExtraction | None = None
    best_physical_extraction: SolvedClusterExtraction | None = None
    best_physical_obj: float | None = None
    best_clean_extraction: SolvedClusterExtraction | None = None
    best_clean_obj: float | None = None
    unit_crossing_cuts_injected = False
    physical_converged = False
    intermediate_timeouts = 0

    has_slack = isinstance(formulation, ClusterFormulation) and formulation.has_transverse_slack()
    slack_stage = 0 if has_slack else 2
    if has_slack:
        formulation.lock_transverse_slack()

    def _escalate_transverse_slack(*, proven_infeasible: bool = False) -> bool:
        nonlocal slack_stage
        if not has_slack or slack_stage >= 2:
            return False
        if slack_stage == 0:
            if proven_infeasible:
                formulation.enforce_min_transverse_slack(1)
            if formulation.unlock_vertical_transverse_slack():
                slack_stage = 1
                return True
            formulation.unlock_transverse_slack()
            slack_stage = 2
            return True
        if slack_stage == 1:
            if proven_infeasible:
                formulation.enforce_min_transverse_slack(1, horizontal_only=True)
            formulation.unlock_transverse_slack()
            slack_stage = 2
            return True
        return False

    try:
        for iteration in range(max_iterations):
            elapsed = time.monotonic() - t_start
            remaining_budget = total_cluster_budget - elapsed
            if remaining_budget <= 0:
                verified_incumbent = _select_verified_incumbent(
                    best_clean_extraction, best_physical_extraction
                )
                if verified_incumbent is not None:
                    extraction = verified_incumbent
                    phase = ClusterSolvePhase.DONE
                    break
                raise LayoutSolverError(
                    f"MILP solver budget of {total_cluster_budget:.1f}s expired with unresolved collisions for cluster '{node.node_id}'."
                )

            if iteration > 0 and solver_name == "cbc":
                effective_inter_rgap = intermediate_ratio_gap
                if raw_inter_rgap is None and n_entities > 20:
                    if intermediate_timeouts == 1:
                        effective_inter_rgap = max(effective_inter_rgap, 0.50)
                    elif intermediate_timeouts >= 2:
                        effective_inter_rgap = max(effective_inter_rgap, 0.80)
            else:
                effective_inter_rgap = ratio_gap

            solver_opts: dict[str, Any] = {}
            if solver_name == "cbc":
                pass_threads = (
                    (1 if n_entities <= 50 else min(4, os.cpu_count() or 1))
                    if iteration > 0
                    else min(4, os.cpu_count() or 1)
                )
                solver_opts = _build_cbc_solver_options(
                    ratio_gap=effective_inter_rgap,
                    threads=pass_threads,
                )

            if iteration > 0:
                pass_timeout = max(1, int(round(min(remaining_budget, intermediate_timeout))))
                pass_max_sol = effective_max_solutions
                use_warmstart = (
                    getattr(options, "warmstart", False)
                    and _has_populated_warmstart_coordinates(formulation)
                )
            else:
                pass_timeout = max(1, int(round(remaining_budget)))
                pass_max_sol = None
                use_warmstart = False

            finish_with_incumbent = False
            pass_escalated = False
            while True:
                pass_options = (
                    replace(options, timeout_seconds=pass_timeout)
                    if options is not None
                    else LayoutOptions(timeout_seconds=pass_timeout)
                )
                try:
                    extraction = solve_and_extract(
                        formulation=formulation,
                        solver_name=solver_name,
                        options=pass_options,
                        max_solutions=pass_max_sol,
                        solver_options=solver_opts if solver_opts else None,
                        warmstart=use_warmstart,
                    )
                    break
                except LayoutInfeasibleError as exc:
                    verified_incumbent = _select_verified_incumbent(
                        best_clean_extraction, best_physical_extraction
                    )
                    if unit_crossing_cuts_injected and verified_incumbent is not None:
                        extraction = verified_incumbent
                        finish_with_incumbent = True
                        phase = ClusterSolvePhase.DONE
                        break
                    if _escalate_transverse_slack(proven_infeasible=True):
                        continue
                    cb = getattr(formulation, "clearance_builder", None)
                    raw_unsat = getattr(cb, "unsatisfiable_collisions", None)
                    if isinstance(raw_unsat, (list, tuple)) and raw_unsat:
                        _raise_collision_error(
                            formulation,
                            node.node_id,
                            str(exc),
                        )
                    raise
                except LayoutSolverError:
                    verified_incumbent = _select_verified_incumbent(
                        best_clean_extraction, best_physical_extraction
                    )
                    if unit_crossing_cuts_injected and verified_incumbent is not None:
                        rem = total_cluster_budget - (time.monotonic() - t_start)
                        if rem <= 2.0 or pass_timeout >= 20 or pass_timeout >= int(round(rem)):
                            extraction = verified_incumbent
                            finish_with_incumbent = True
                            phase = ClusterSolvePhase.DONE
                            break
                        pass_timeout = min(max(1, int(round(rem))), max(pass_timeout * 2, 20))
                        continue
                    if iteration == 0:
                        if _escalate_transverse_slack(proven_infeasible=False):
                            continue
                        raise
                    rem = total_cluster_budget - (time.monotonic() - t_start)
                    if rem <= 2.0:
                        if verified_incumbent is not None:
                            extraction = verified_incumbent
                            finish_with_incumbent = True
                            phase = ClusterSolvePhase.DONE
                            break
                        raise
                    next_timeout = min(max(1, int(round(rem))), max(pass_timeout * 2, 30))
                    if pass_timeout >= int(round(rem)) or next_timeout <= pass_timeout:
                        if _escalate_transverse_slack(proven_infeasible=False):
                            pass_timeout = min(pass_timeout, max(1, int(round(rem))))
                            continue
                        raise
                    pass_timeout = next_timeout
                    pass_escalated = True

            if finish_with_incumbent:
                break

            if (
                iteration > 0
                and pass_escalated
                and not has_explicit_inter_timeout
                and not unit_crossing_cuts_injected
            ):
                intermediate_timeout = max(
                    intermediate_timeout, min(30.0, max(intermediate_timeout * 1.5, 15.0))
                )

            if iteration > 0 and extraction.solver_status == SolverStatus.TIMEOUT:
                intermediate_timeouts += 1

            physical_collisions, corridor_crossings, compacted_crossings = (
                _detect_and_partition_violations(formulation, extraction)
            )
            has_compacted_crossing = bool(compacted_crossings)

            # Fast single-pass exit when iteration 0 is already optimal and violation-free.
            if (
                not physical_collisions
                and not corridor_crossings
                and iteration == 0
                and extraction.solver_status == SolverStatus.OPTIMAL
                and effective_max_solutions is None
            ):
                phase = ClusterSolvePhase.DONE
                break

            cur_obj = _extract_objective_value(formulation)

            if not physical_collisions and not corridor_crossings:
                best_clean_extraction, best_clean_obj = _select_better_incumbent(
                    best_clean_extraction,
                    best_clean_obj,
                    extraction,
                    cur_obj,
                )
            if (
                not physical_collisions
                and not has_compacted_crossing
                and (best_clean_extraction is None or not corridor_crossings)
            ):
                best_physical_extraction, best_physical_obj = _select_better_incumbent(
                    best_physical_extraction,
                    best_physical_obj,
                    extraction,
                    cur_obj,
                )

            if physical_collisions:
                phase = ClusterSolvePhase.PHYSICAL_CLEARANCE
                physical_converged = False
                new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                    [*physical_collisions, *compacted_crossings],
                    extraction=extraction,
                )
                if new_cuts == 0:
                    verified_incumbent = _select_verified_incumbent(
                        best_clean_extraction, best_physical_extraction
                    )
                    if unit_crossing_cuts_injected and verified_incumbent is not None:
                        extraction = verified_incumbent
                        phase = ClusterSolvePhase.DONE
                        break
                    if _escalate_transverse_slack(proven_infeasible=False):
                        continue
                    if iteration > 0 and intermediate_timeout < remaining_budget - 1.0:
                        intermediate_timeout = remaining_budget
                        continue
                    _raise_collision_error(
                        formulation,
                        node.node_id,
                        f"MILP solver encountered unresolvable geometric collisions for cluster '{node.node_id}'.",
                        fallback_collisions=physical_collisions,
                    )
                continue

            if corridor_crossings:
                phase = ClusterSolvePhase.CROSSING_CLEARANCE
                max_crossing_cuts = (
                    getattr(options, "max_cuts_per_pass", 20) if options else 20
                )
                if compacted_crossings:
                    new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                        compacted_crossings, max_cuts_per_pass=max_crossing_cuts
                    )
                    if new_cuts > 0:
                        physical_converged = False
                        continue
                    verified_incumbent = _select_verified_incumbent(
                        best_clean_extraction, best_physical_extraction
                    )
                    if verified_incumbent is not None:
                        extraction = verified_incumbent
                        phase = ClusterSolvePhase.DONE
                        break
                else:
                    needs_pre_unit_convergence = (
                        not physical_converged
                        and not unit_crossing_cuts_injected
                        and (
                            effective_inter_rgap > ratio_gap
                            or extraction.solver_status == SolverStatus.TIMEOUT
                            or pass_max_sol is not None
                        )
                    )
                    if needs_pre_unit_convergence and remaining_budget >= 1.0:
                        remaining_budget = total_cluster_budget - (time.monotonic() - t_start)
                        if remaining_budget >= 1.0:
                            try:
                                conv_budget = (
                                    min(remaining_budget, max(30.0, base_timeout))
                                    if (has_slack and slack_stage > 0)
                                    else remaining_budget
                                )
                                conv_extraction = _run_convergence_solve(
                                    formulation=formulation,
                                    options=options,
                                    solver_name=solver_name,
                                    ratio_gap=ratio_gap,
                                    remaining_budget=conv_budget,
                                )
                                conv_physical, conv_crossings, conv_compacted = (
                                    _detect_and_partition_violations(formulation, conv_extraction)
                                )
                                if conv_physical or conv_compacted:
                                    new_conv_cuts = formulation.clearance_builder.inject_clearance_cuts(
                                        [*conv_physical, *conv_compacted],
                                        extraction=conv_extraction if conv_physical else None,
                                    )
                                    remaining_budget = total_cluster_budget - (time.monotonic() - t_start)
                                    if new_conv_cuts > 0 and remaining_budget >= 1.0:
                                        physical_converged = False
                                        extraction = conv_extraction
                                        phase = (
                                            ClusterSolvePhase.PHYSICAL_CLEARANCE
                                            if conv_physical
                                            else ClusterSolvePhase.CROSSING_CLEARANCE
                                        )
                                        continue
                                    verified_incumbent = _select_verified_incumbent(
                                        best_clean_extraction, best_physical_extraction
                                    )
                                    extraction = (
                                        verified_incumbent
                                        if verified_incumbent is not None
                                        else conv_extraction
                                    )
                                    phase = ClusterSolvePhase.DONE
                                    break
                                else:
                                    physical_converged = True
                                    conv_obj = _extract_objective_value(formulation)
                                    if best_clean_extraction is None or not conv_crossings:
                                        best_physical_extraction, best_physical_obj = (
                                            _select_better_incumbent(
                                                best_physical_extraction,
                                                best_physical_obj,
                                                conv_extraction,
                                                conv_obj,
                                            )
                                        )
                                    if not conv_crossings:
                                        best_clean_extraction, best_clean_obj = (
                                            _select_better_incumbent(
                                                best_clean_extraction,
                                                best_clean_obj,
                                                conv_extraction,
                                                conv_obj,
                                            )
                                        )
                                        extraction = best_clean_extraction
                                        phase = ClusterSolvePhase.DONE
                                        break
                                    extraction = conv_extraction
                                    corridor_crossings = conv_crossings
                                    remaining_budget = total_cluster_budget - (time.monotonic() - t_start)
                            except LayoutSolverError:
                                physical_converged = True
                                remaining_budget = total_cluster_budget - (time.monotonic() - t_start)

                    if remaining_budget >= unit_crossing_budget_floor:
                        new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                            corridor_crossings, max_cuts_per_pass=max_crossing_cuts
                        )
                        if new_cuts > 0:
                            unit_crossing_cuts_injected = True
                            continue

                    if physical_converged:
                        verified_incumbent = _select_verified_incumbent(
                            best_clean_extraction, best_physical_extraction
                        )
                        extraction = verified_incumbent if verified_incumbent is not None else extraction
                        phase = ClusterSolvePhase.DONE
                        break

            phase = ClusterSolvePhase.CONVERGENCE
            elapsed = time.monotonic() - t_start
            remaining_budget = total_cluster_budget - elapsed
            if remaining_budget >= 1.0:
                try:
                    conv_budget = (
                        min(remaining_budget, max(30.0, base_timeout))
                        if (has_slack and slack_stage > 0)
                        else remaining_budget
                    )
                    final_extraction = _run_convergence_solve(
                        formulation=formulation,
                        options=options,
                        solver_name=solver_name,
                        ratio_gap=ratio_gap,
                        remaining_budget=conv_budget,
                    )
                    final_physical, final_crossings, final_compacted = (
                        _detect_and_partition_violations(formulation, final_extraction)
                    )
                    if final_physical:
                        phase = ClusterSolvePhase.PHYSICAL_CLEARANCE
                        extraction = final_extraction
                        new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                            [*final_physical, *final_compacted],
                            extraction=final_extraction,
                        )
                        remaining_budget = total_cluster_budget - (time.monotonic() - t_start)
                        if new_cuts == 0 or remaining_budget < 1.0:
                            if best_clean_extraction is not None:
                                extraction = best_clean_extraction
                                phase = ClusterSolvePhase.DONE
                                break
                            _raise_collision_error(
                                formulation,
                                node.node_id,
                                f"MILP solver encountered unresolvable geometric collisions for cluster '{node.node_id}'.",
                                fallback_collisions=final_physical if new_cuts == 0 else (),
                            )
                        physical_converged = False
                        continue
                    elif final_crossings:
                        phase = ClusterSolvePhase.CROSSING_CLEARANCE
                        extraction = final_extraction
                        max_crossing_cuts = (
                            getattr(options, "max_cuts_per_pass", 20) if options else 20
                        )
                        remaining_budget = total_cluster_budget - (time.monotonic() - t_start)
                        if final_compacted:
                            new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                                final_compacted, max_cuts_per_pass=max_crossing_cuts
                            )
                            if new_cuts > 0:
                                physical_converged = False
                                continue
                        elif remaining_budget >= unit_crossing_budget_floor:
                            new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                                final_crossings, max_cuts_per_pass=max_crossing_cuts
                            )
                            if new_cuts > 0:
                                unit_crossing_cuts_injected = True
                                continue
                        if best_clean_extraction is not None:
                            extraction = best_clean_extraction
                        elif final_compacted and best_physical_extraction is not None:
                            extraction = best_physical_extraction
                        phase = ClusterSolvePhase.DONE
                        break
                    else:
                        final_obj = _extract_objective_value(formulation)
                        if (
                            best_clean_extraction is not None
                            and final_obj is not None
                            and best_clean_obj is not None
                            and final_obj > best_clean_obj
                        ):
                            extraction = best_clean_extraction
                        else:
                            if (
                                final_extraction.solver_status == SolverStatus.TIMEOUT
                                and extraction.solver_status == SolverStatus.OPTIMAL
                            ):
                                final_extraction = replace(
                                    final_extraction,
                                    solver_status=SolverStatus.OPTIMAL,
                                )
                            extraction = final_extraction
                        phase = ClusterSolvePhase.DONE
                        break
                except LayoutSolverError:
                    # If final optimization pass fails but current extraction is verified clean, accept incumbent.
                    if best_clean_extraction is not None:
                        extraction = best_clean_extraction
                    phase = ClusterSolvePhase.DONE
                    break
            else:
                if best_clean_extraction is not None:
                    extraction = best_clean_extraction
                phase = ClusterSolvePhase.DONE
                break

        if phase != ClusterSolvePhase.DONE or extraction is None:
            verified_incumbent = _select_verified_incumbent(
                best_clean_extraction, best_physical_extraction
            )
            if verified_incumbent is not None:
                extraction = verified_incumbent
            else:
                _raise_collision_error(
                    formulation,
                    node.node_id,
                    f"MILP solver failed to resolve all geometric collisions within {max_iterations} iterations for cluster '{node.node_id}'.",
                )
    finally:
        if has_slack:
            formulation.clear_transverse_slack_lower_bounds()
            if slack_stage < 2:
                formulation.unlock_transverse_slack()
    return extraction


def solve_cluster(
    node: SolveTreeNode,
    child_solutions: Mapping[str, NodeSolution] | LayoutGraph | None = None,
    graph: LayoutGraph | None = None,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
    debug_solve_dir: Path | str | None = None,
) -> NodeSolution:
    """Solve layout cluster embedding constituent rooms and child macro-boxes via MILP.

    Coordinates the canonical MILP layout optimization pipeline:
      1. Inventory & Interface Classification (inventory_cluster)
      2. Basic MILP Formulation (formulate_cluster_model)
      3. Iterative Cutting-Plane Clearance & Solve (detect_cluster_collisions & solve_and_extract)
      4. Coordinate Normalization (normalize_cluster_solution)

    Args:
        node: SolveTreeNode representing the cluster to lay out.
        child_solutions: Mapping of child node_id to already-solved NodeSolution,
            or LayoutGraph if solving a leaf cluster without children.
        graph: Underlying LayoutGraph with room definitions and exit edges.
        options: Optional layout configuration parameters.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').
        debug_solve_dir: Optional filesystem directory for exporting debug solve SVGs.

    Returns:
        NodeSolution containing local coordinates and child offsets in metadata.

    Raises:
        LayoutSolverError: If the MILP solver is unavailable, infeasible, or solving fails.
    """
    if isinstance(child_solutions, LayoutGraph):
        graph = child_solutions
        child_solutions = {}
    elif child_solutions is None:
        child_solutions = {}

    if graph is None:
        graph = LayoutGraph(rooms={}, edges=())
    rooms = sorted(node.rooms)
    children = list(node.children)
    boundary_ports = frozenset(get_child_boundary_ports(node))

    if not rooms and not children:
        return normalize_empty_cluster(node)

    if node.node_type == TreeNodeType.WORLD_ROOT:
        from romutil.layout.milp.inventory import _collect_extended_footprints

        coord_unit, _room_scales, room_extents = _collect_extended_footprints(graph, options)
        return pack_world_root_components(
            node=node,
            child_solutions=child_solutions,
            options=options,
            coord_unit=coord_unit,
            room_extents=room_extents,
        )

    if len(rooms) == 1 and not children and not boundary_ports:
        r = rooms[0]
        return normalize_single_room_cluster(
            node=node,
            room=r,
            is_boundary_port=False,
        )

    try:
        # Stage 1: Inventory & Interface Classification
        inventory = inventory_cluster(
            node=node,
            child_solutions=child_solutions,
            graph=graph,
            options=options,
            boundary_ports=boundary_ports,
        )

        n_sub_boxes = sum(len(sbs) for sbs in inventory.child_sub_boxes.values())
        t_cluster = compute_adaptive_cluster_timeout(
            node, options, n_sub_boxes=n_sub_boxes
        )

        # Stage 2: Basic MILP Formulation
        formulation = formulate_cluster_model(inventory=inventory, options=options)

        # Stage 3: Iterative Cutting-Plane Clearance & Solve
        extraction = solve_cluster_with_cutting_planes(
            formulation=formulation,
            node=node,
            options=options,
            solver_name=solver_name,
            total_cluster_budget=float(t_cluster),
        )

        assert extraction is not None

        debug_dir = (
            debug_solve_dir
            if debug_solve_dir is not None
            else (options.debug_solve_dir if options is not None else None)
        )
        if debug_dir is not None:
            from romutil.layout.milp.debug import render_cluster_solve_svg

            out_dir = Path(debug_dir)
            out_file = out_dir / f"cluster_{node.node_id}.svg"
            render_cluster_solve_svg(
                extraction=extraction,
                inventory=inventory,
                output_path=out_file,
                title=f"Cluster {node.node_id}",
                graph=graph,
            )

        # Stage 5: Coordinate Normalization
        return normalize_cluster_solution(
            extraction=extraction,
            node=node,
            children=inventory.children,
            child_solutions=child_solutions,
            child_sub_boxes=inventory.child_sub_boxes,
            active_edges=inventory.active_edges,
            graph=graph,
            room_extents=inventory.room_extents,
        )
    except LayoutSolverError:
        raise
    except Exception as exc:
        raise LayoutSolverError(
            f"MILP solver failed for cluster '{node.node_id}': {exc}"
        ) from exc


class HierarchicalClusterSolver:
    """Solver for layout clusters (both leaf and composite parent clusters) via MILP."""

    def __init__(self, solver_name: str = "cbc") -> None:
        self.solver_name = solver_name

    def solve(
        self,
        node: SolveTreeNode,
        child_solutions: Mapping[str, NodeSolution] | LayoutGraph | None = None,
        graph: LayoutGraph | None = None,
        options: LayoutOptions | None = None,
        debug_solve_dir: Path | str | None = None,
    ) -> NodeSolution:
        """Solve cluster layout non-overlapping coordinates and child offsets."""
        return solve_cluster(
            node=node,
            child_solutions=child_solutions,
            graph=graph,
            options=options,
            solver_name=self.solver_name,
            debug_solve_dir=debug_solve_dir,
        )


__all__ = [
    "ClusterSolvePhase",
    "HierarchicalClusterSolver",
    "UnresolvableCollisionError",
    "compute_adaptive_cluster_timeout",
    "solve_cluster",
    "solve_cluster_with_cutting_planes",
]
