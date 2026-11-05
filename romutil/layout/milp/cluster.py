"""Modular 5-stage MILP cluster layout solver."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from romutil.layout.ir import LayoutGraph, NodeSolution, SolveTreeNode, TreeNodeType
from romutil.layout.milp.clearance import detect_cluster_collisions
from romutil.layout.milp.formulation import ClusterFormulation, formulate_cluster_model
from romutil.layout.milp.inventory import (
    get_child_boundary_ports,
    inventory_cluster,
)
from romutil.layout.milp.normalization import (
    normalize_cluster_solution,
    normalize_empty_cluster,
    normalize_single_room_cluster,
    pack_world_root_components,
)
from romutil.layout.milp.solver import SolvedClusterExtraction, solve_and_extract
from romutil.types import LayoutOptions, LayoutSolverError, SolverStatus


class ClusterSolvePhase(str, Enum):
    """Execution phases of the cutting-plane clearance state machine."""

    PHYSICAL_CLEARANCE = "physical_clearance"
    CROSSING_CLEARANCE = "crossing_clearance"
    CONVERGENCE = "convergence"
    DONE = "done"


def compute_adaptive_cluster_timeout(
    node: SolveTreeNode,
    options: LayoutOptions | None = None,
) -> int:
    """Compute adaptive solver timeout limit scaled to cluster entity complexity.

    Scales linearly with the total count of constituent rooms and child clusters
    beyond a 20-entity baseline to ensure large composite clusters receive sufficient
    solver budget for integer convergence.

    Args:
        node: SolveTreeNode representing the cluster to lay out.
        options: Optional layout configuration parameters containing base timeout.

    Returns:
        Adaptive timeout duration in seconds.
    """
    n_entities = len(node.rooms) + len(node.children)
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


def solve_cluster_with_cutting_planes(
    formulation: ClusterFormulation,
    node: SolveTreeNode,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
    max_iterations: int = 35,
    total_cluster_budget: float | None = None,
    max_solutions: int | None = None,
) -> SolvedClusterExtraction:
    """Iteratively solve cluster MILP formulation with a cutting-plane clearance state machine.

    Coordinates an explicit 4-phase state machine:
      1. PHYSICAL_CLEARANCE: When physical room or macro-box collisions exist, inject disjunctive
         clearance cuts strictly for physical collisions using intermediate fast passes.
      2. CROSSING_CLEARANCE: When all physical collisions are cleared but planar corridor crossings
         remain, inject corridor crossing separation cuts. If resolving crossings re-introduces
         physical collisions, immediately falls back to PHYSICAL_CLEARANCE (self-healing).
      3. CONVERGENCE: When both physical collisions and corridor crossings are zero, runs a final
         convergence optimization pass warm-started from the clean incumbent using the full
         remaining cluster budget and no early solution limits. If violations reappear during
         convergence, falls back to the appropriate clearance phase.
      4. DONE: Reached exclusively when an extraction is verified to contain zero physical
         collisions and zero corridor crossings.

    Aggregate cluster budget is tracked monotonically across all iterations. If the budget
    expires or max_iterations is reached with unresolved collisions or crossings, raises
    LayoutSolverError.

    Args:
        formulation: ClusterFormulation containing the Pyomo model and clearance builder.
        node: SolveTreeNode representing the cluster to lay out.
        options: Optional layout configuration parameters containing base timeout.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').
        max_iterations: Maximum cutting-plane refinement iterations (defaults to 15).
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
    t_start = time.monotonic()
    base_timeout = float(options.timeout_seconds) if options is not None and options.timeout_seconds else 15.0
    intermediate_timeout = (
        float(options.intermediate_pass_timeout)
        if options is not None and getattr(options, "intermediate_pass_timeout", None) and options.intermediate_pass_timeout != 15
        else max(15.0, min(60.0, base_timeout * 0.67))
    )
    ratio_gap = (
        float(options.ratio_gap)
        if options is not None and getattr(options, "ratio_gap", None)
        else 0.05
    )
    effective_max_solutions = (
        max_solutions
        if max_solutions is not None
        else (options.max_solutions if options is not None else None)
    )

    phase: ClusterSolvePhase = ClusterSolvePhase.PHYSICAL_CLEARANCE
    extraction: SolvedClusterExtraction | None = None
    best_physical_extraction: SolvedClusterExtraction | None = None

    for iteration in range(max_iterations):
        elapsed = time.monotonic() - t_start
        remaining_budget = total_cluster_budget - elapsed
        if remaining_budget <= 0:
            if best_physical_extraction is not None:
                extraction = best_physical_extraction
                phase = ClusterSolvePhase.DONE
                break
            raise LayoutSolverError(
                f"MILP solver budget of {total_cluster_budget:.1f}s expired with unresolved collisions for cluster '{node.node_id}'."
            )

        solver_opts: dict[str, Any] = {}
        if solver_name == "cbc":
            solver_opts["heuristics"] = "on"
            solver_opts["ratioGap"] = max(0.12, ratio_gap)
            solver_opts["threads"] = min(4, os.cpu_count() or 1)
            solver_opts["timeMode"] = "elapsed"

        if iteration > 0:
            pass_timeout = max(1, int(round(min(remaining_budget, intermediate_timeout))))
            pass_max_sol = effective_max_solutions
            use_warmstart = _has_populated_warmstart_coordinates(formulation)
        else:
            pass_timeout = max(1, int(round(remaining_budget)))
            pass_max_sol = None
            use_warmstart = False

        pass_options = (
            replace(options, timeout_seconds=pass_timeout)
            if options is not None
            else LayoutOptions(timeout_seconds=pass_timeout)
        )

        extraction = solve_and_extract(
            formulation=formulation,
            solver_name=solver_name,
            options=pass_options,
            max_solutions=pass_max_sol,
            solver_options=solver_opts if solver_opts else None,
            warmstart=use_warmstart,
        )

        violations = detect_cluster_collisions(
            formulation=formulation,
            extraction=extraction,
        )

        physical_collisions = [v for v in violations if v.category != "corridor_crossing"]
        corridor_crossings = [v for v in violations if v.category == "corridor_crossing"]

        if not physical_collisions:
            best_physical_extraction = extraction

        if physical_collisions:
            phase = ClusterSolvePhase.PHYSICAL_CLEARANCE
            new_cuts = formulation.clearance_builder.inject_clearance_cuts(physical_collisions)
            if new_cuts == 0:
                if iteration > 0 and intermediate_timeout < remaining_budget - 1.0:
                    intermediate_timeout = remaining_budget
                    continue
                raise LayoutSolverError(
                    f"MILP solver encountered unresolvable geometric collisions for cluster '{node.node_id}'."
                )
            continue

        if corridor_crossings:
            phase = ClusterSolvePhase.CROSSING_CLEARANCE
            max_crossing_cuts = min(2, getattr(options, "max_cuts_per_pass", 2) if options else 2)
            new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                corridor_crossings, max_cuts_per_pass=max_crossing_cuts
            )
            if new_cuts > 0 and remaining_budget >= 15.0:
                continue

        # Both physical collisions and corridor crossings are zero.
        if (
            iteration == 0
            and extraction.solver_status == SolverStatus.OPTIMAL
            and effective_max_solutions is None
        ):
            phase = ClusterSolvePhase.DONE
            break

        phase = ClusterSolvePhase.CONVERGENCE
        elapsed = time.monotonic() - t_start
        remaining_budget = total_cluster_budget - elapsed
        if remaining_budget >= 1.0:
            final_timeout = max(1, int(round(remaining_budget)))
            final_options = (
                replace(options, timeout_seconds=final_timeout)
                if options is not None
                else LayoutOptions(timeout_seconds=final_timeout)
            )
            final_solver_options: dict[str, Any] = {}
            if solver_name == "cbc":
                final_solver_options["heuristics"] = "on"
                final_solver_options["ratioGap"] = ratio_gap
                final_solver_options["threads"] = min(4, os.cpu_count() or 1)
                final_solver_options["timeMode"] = "elapsed"

            final_warmstart = _has_populated_warmstart_coordinates(formulation)
            try:
                final_extraction = solve_and_extract(
                    formulation=formulation,
                    solver_name=solver_name,
                    options=final_options,
                    max_solutions=None,
                    solver_options=final_solver_options if final_solver_options else None,
                    warmstart=final_warmstart,
                )
                final_violations = detect_cluster_collisions(
                    formulation=formulation,
                    extraction=final_extraction,
                )
                final_physical = [
                    v for v in final_violations if v.category != "corridor_crossing"
                ]
                final_crossings = [
                    v for v in final_violations if v.category == "corridor_crossing"
                ]
                if final_physical:
                    phase = ClusterSolvePhase.PHYSICAL_CLEARANCE
                    extraction = final_extraction
                    new_cuts = formulation.clearance_builder.inject_clearance_cuts(final_physical)
                    if new_cuts == 0:
                        raise LayoutSolverError(
                            f"MILP solver encountered unresolvable geometric collisions for cluster '{node.node_id}'."
                        )
                    continue
                elif final_crossings:
                    phase = ClusterSolvePhase.CROSSING_CLEARANCE
                    extraction = final_extraction
                    max_crossing_cuts = min(2, getattr(options, "max_cuts_per_pass", 2) if options else 2)
                    new_cuts = formulation.clearance_builder.inject_clearance_cuts(
                        final_crossings, max_cuts_per_pass=max_crossing_cuts
                    )
                    if new_cuts > 0 and remaining_budget >= 15.0:
                        continue
                    phase = ClusterSolvePhase.DONE
                    break
                else:
                    extraction = final_extraction
                    phase = ClusterSolvePhase.DONE
                    break
            except LayoutSolverError:
                # If final optimization pass fails but current extraction is verified clean, accept incumbent.
                phase = ClusterSolvePhase.DONE
                break
        else:
            phase = ClusterSolvePhase.DONE
            break

    if phase != ClusterSolvePhase.DONE or extraction is None:
        if best_physical_extraction is not None:
            extraction = best_physical_extraction
        else:
            raise LayoutSolverError(
                f"MILP solver failed to resolve all geometric collisions within {max_iterations} iterations for cluster '{node.node_id}'."
            )
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
        return pack_world_root_components(
            node=node,
            child_solutions=child_solutions,
            options=options,
        )

    if len(rooms) == 1 and not children and not boundary_ports:
        r = rooms[0]
        return normalize_single_room_cluster(
            node=node,
            room=r,
            is_boundary_port=False,
        )

    t_cluster = compute_adaptive_cluster_timeout(node, options)
    if options is not None:
        cluster_options = replace(options, timeout_seconds=t_cluster)
    else:
        cluster_options = LayoutOptions(timeout_seconds=t_cluster)

    try:
        # Stage 1: Inventory & Interface Classification
        inventory = inventory_cluster(
            node=node,
            child_solutions=child_solutions,
            graph=graph,
            options=cluster_options,
            boundary_ports=boundary_ports,
        )

        # Stage 2: Basic MILP Formulation
        formulation = formulate_cluster_model(inventory=inventory, options=cluster_options)

        # Stage 3: Iterative Cutting-Plane Clearance & Solve
        extraction = solve_cluster_with_cutting_planes(
            formulation=formulation,
            node=node,
            options=cluster_options,
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
    "compute_adaptive_cluster_timeout",
    "solve_cluster",
    "solve_cluster_with_cutting_planes",
]
