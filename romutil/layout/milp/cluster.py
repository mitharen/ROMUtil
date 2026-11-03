"""Modular 5-stage MILP cluster layout solver."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from romutil.layout.ir import LayoutGraph, NodeSolution, SolveTreeNode, TreeNodeType
from romutil.layout.milp.clearance import detect_cluster_collisions
from romutil.layout.milp.formulation import formulate_cluster_model
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


def solve_cluster_with_cutting_planes(
    formulation: ClusterFormulation,
    node: SolveTreeNode,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
    max_iterations: int = 15,
    total_cluster_budget: float | None = None,
    max_solutions: int | None = None,
) -> SolvedClusterExtraction:
    """Iteratively solve cluster MILP formulation with disjunctive cutting planes.

    Executes a two-phase cutting-plane solve strategy:
    1. Intermediate passes: Uses tight per-pass timeout limits and optional early incumbent
       termination (max_solutions) with heuristics to quickly obtain integer-feasible coordinates,
       allowing collision detection to identify geometric overlaps and inject clearance cuts.
    2. Final convergence pass: Once all collisions are resolved (zero violations detected),
       runs a final optimization pass without early solution limits using the remaining
       cluster budget to tighten edge lengths and spring penalties with stagnation guards.

    Aggregate cluster budget is tracked monotonically across all iterations to prevent
    per-pass timeout resets from multiplying the total cluster runtime across iterations.

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
        SolvedClusterExtraction containing valid coordinates and child offsets.

    Raises:
        LayoutSolverError: If geometric collisions cannot be resolved within budget or iterations.
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

    extraction: SolvedClusterExtraction | None = None
    cuts_injected = False
    for _ in range(max_iterations):
        elapsed = time.monotonic() - t_start
        remaining_budget = max(1.0, total_cluster_budget - elapsed)

        if cuts_injected:
            pass_timeout = max(1, int(round(min(remaining_budget, intermediate_timeout))))
            pass_max_sol = effective_max_solutions
            solver_opts: dict[str, Any] = {}
            if solver_name == "cbc":
                solver_opts["heuristics"] = "on"
                solver_opts["ratioGap"] = ratio_gap
        else:
            pass_timeout = max(1, int(round(remaining_budget)))
            pass_max_sol = None
            solver_opts = {}

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
        )

        violations = detect_cluster_collisions(
            formulation=formulation,
            extraction=extraction,
        )

        # On intermediate early-exit passes or timeout passes, ignore planar corridor crossings
        # when evaluating whether physical room and macro-box collisions have been resolved.
        is_intermediate_pass = pass_max_sol is not None
        if extraction.solver_status == SolverStatus.TIMEOUT or is_intermediate_pass or cuts_injected:
            hard_violations = [v for v in violations if v.category != "corridor_crossing"]
        else:
            hard_violations = list(violations)

        if not hard_violations:
            # All hard physical collisions resolved!
            # If this was an early-terminated intermediate pass, run final convergence pass.
            if is_intermediate_pass and extraction.solver_status == SolverStatus.FEASIBLE:
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

                    try:
                        final_extraction = solve_and_extract(
                            formulation=formulation,
                            solver_name=solver_name,
                            options=final_options,
                            max_solutions=None,
                            solver_options=final_solver_options,
                        )
                        final_violations = detect_cluster_collisions(
                            formulation=formulation,
                            extraction=final_extraction,
                        )
                        if final_extraction.solver_status == SolverStatus.TIMEOUT:
                            final_violations = [
                                v for v in final_violations if v.category != "corridor_crossing"
                            ]
                        if not final_violations:
                            extraction = final_extraction
                            break
                        new_cuts = formulation.clearance_builder.inject_clearance_cuts(final_violations)
                        if new_cuts > 0:
                            cuts_injected = True
                            continue
                    except LayoutSolverError:
                        pass
            break

        new_cuts = formulation.clearance_builder.inject_clearance_cuts(hard_violations)
        if new_cuts == 0:
            if extraction.solver_status == SolverStatus.TIMEOUT and remaining_budget > pass_timeout:
                intermediate_timeout = remaining_budget
                cuts_injected = True
                continue
            raise LayoutSolverError(
                f"MILP solver encountered unresolvable geometric collisions for cluster '{node.node_id}'."
            )
        cuts_injected = True
    else:
        raise LayoutSolverError(
            f"MILP solver failed to resolve all geometric collisions within {max_iterations} iterations for cluster '{node.node_id}'."
        )

    assert extraction is not None
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
    "HierarchicalClusterSolver",
    "compute_adaptive_cluster_timeout",
    "solve_cluster",
    "solve_cluster_with_cutting_planes",
]
