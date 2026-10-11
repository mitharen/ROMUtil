"""Stage 4: Solve and extraction for layout cluster MILP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.geometry import BoundingBox3D, decompose_into_orthogonal_sub_boxes
from romutil.layout.ir import NodeSolution, SolveTreeNode
from romutil.layout.milp.formulation import ClusterFormulation
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus


@dataclass(frozen=True)
class SolvedClusterExtraction:
    """Solved unnormalized room coordinates, child offsets, and sub-boxes."""

    raw_parent_coords: Mapping[int, tuple[int, int, int]]
    raw_child_offsets: Mapping[str, tuple[int, int, int]]
    raw_sub_boxes: tuple[BoundingBox3D, ...]
    solver_status: SolverStatus


def decompose_cluster_sub_boxes(
    raw_parent_coords: Mapping[int, tuple[int, int, int]],
    raw_child_offsets: Mapping[str, tuple[int, int, int]],
    node: SolveTreeNode,
    children: Sequence[SolveTreeNode],
    child_solutions: Mapping[str, NodeSolution],
    boundary_ports: frozenset[int],
) -> tuple[BoundingBox3D, ...]:
    """Decompose interior rooms and translated child sub-boxes into orthogonal sub-boxes.

    Boundary ports (e.g. gateway poles) are excluded from the cluster's internal
    bounding footprint. If the cluster is a leaf (no child clusters), maximal-volume
    orthogonal decomposition partitions non-convex clusters into minimal disjoint boxes.
    If composite, the enclosing bounding box is returned.

    Args:
        raw_parent_coords: Mapping of parent room IDs to 3D coordinate tuples.
        raw_child_offsets: Mapping of child node IDs to 3D origin tuples.
        node: SolveTreeNode representing the current cluster.
        children: Sequence of child solve tree nodes.
        child_solutions: Mapping of child node IDs to solved NodeSolutions.
        boundary_ports: Set of boundary port room IDs to exclude.

    Returns:
        Tuple of disjoint BoundingBox3D instances representing the internal volume.
    """
    interior_cells: set[Coord3D] = {
        Coord3D(*raw_parent_coords[u])
        for u in node.rooms
        if u not in boundary_ports and u in raw_parent_coords
    }
    for child in children:
        cid = child.node_id
        c_sol = child_solutions.get(cid)
        if c_sol:
            cx, cy, cz = raw_child_offsets[cid]
            for sb in c_sol.sub_boxes:
                for x in range(sb.min_x + cx, sb.max_x + cx + 1):
                    for y in range(sb.min_y + cy, sb.max_y + cy + 1):
                        for z in range(sb.min_z + cz, sb.max_z + cz + 1):
                            interior_cells.add(Coord3D(x, y, z))

    if interior_cells:
        enc = BoundingBox3D.from_coordinates(interior_cells)
        if not children:
            return decompose_into_orthogonal_sub_boxes(interior_cells)
        return (enc,)
    return ()


def solve_and_extract(
    formulation: ClusterFormulation,
    solver_name: str = "cbc",
    options: LayoutOptions | None = None,
) -> SolvedClusterExtraction:
    """Invoke the MILP solver and extract raw coordinates, child offsets, and sub-boxes.

    Handles solver availability, timeouts, infeasibility, and value extraction errors,
    raising LayoutSolverError on non-optimal solver exits or missing variables.

    Args:
        formulation: ClusterFormulation containing the configured Pyomo model.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').
        options: Optional layout configuration parameters (e.g. timeout_seconds).

    Returns:
        SolvedClusterExtraction containing raw coordinates, offsets, and sub-boxes.

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

    if options and options.timeout_seconds:
        if solver_name == "cbc":
            solver.options["seconds"] = options.timeout_seconds
        elif solver_name == "glpk":
            solver.options["tmlim"] = options.timeout_seconds

    results = solver.solve(formulation.model, tee=False)

    valid_status = (
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
            )
            and results.solver.status
            in (
                PyomoSolverStatus.ok,
                PyomoSolverStatus.aborted,
                PyomoSolverStatus.warning,
            )
        )
    )
    if not valid_status:
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
    elif results.solver.termination_condition in (
        TerminationCondition.maxTimeLimit,
        TerminationCondition.intermediateNonInteger,
    ):
        solver_status = SolverStatus.TIMEOUT

    raw_sub_boxes = decompose_cluster_sub_boxes(
        raw_parent_coords=raw_parent_coords,
        raw_child_offsets=raw_child_offsets,
        node=node,
        children=children,
        child_solutions=formulation.inventory.child_solutions,
        boundary_ports=formulation.inventory.boundary_ports,
    )

    return SolvedClusterExtraction(
        raw_parent_coords=raw_parent_coords,
        raw_child_offsets=raw_child_offsets,
        raw_sub_boxes=raw_sub_boxes,
        solver_status=solver_status,
    )


__all__ = [
    "SolvedClusterExtraction",
    "decompose_cluster_sub_boxes",
    "solve_and_extract",
]
