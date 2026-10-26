"""Modular 5-stage MILP cluster layout solver."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from romutil.layout.ir import LayoutGraph, NodeSolution, SolveTreeNode
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
        max_iterations = 15
        extraction: SolvedClusterExtraction | None = None
        for _ in range(max_iterations):
            extraction = solve_and_extract(
                formulation=formulation,
                solver_name=solver_name,
                options=cluster_options,
            )
            violations = detect_cluster_collisions(
                formulation=formulation,
                extraction=extraction,
            )
            if not violations:
                break

            if extraction.solver_status == SolverStatus.TIMEOUT:
                violations = [v for v in violations if v.category != "corridor_crossing"]
                if not violations:
                    break

            new_cuts = formulation.clearance_builder.inject_clearance_cuts(violations)
            if new_cuts == 0:
                raise LayoutSolverError(
                    f"MILP solver encountered unresolvable geometric collisions for cluster '{node.node_id}'."
                )
        else:
            raise LayoutSolverError(
                f"MILP solver failed to resolve all geometric collisions within {max_iterations} iterations for cluster '{node.node_id}'."
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
]
