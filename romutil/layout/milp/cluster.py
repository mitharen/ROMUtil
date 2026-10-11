"""Modular 5-stage MILP cluster layout solver and hierarchical tree coordinator."""

from __future__ import annotations

from typing import Mapping

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.milp.clearance import apply_clearance_constraints
from romutil.layout.milp.formulation import formulate_cluster_model
from romutil.layout.milp.inventory import (
    ActiveEdge,
    EndpointRef,
    get_child_boundary_ports,
    inventory_cluster,
)
from romutil.layout.milp.normalization import (
    normalize_cluster_solution,
    normalize_empty_cluster,
    normalize_single_room_cluster,
)
from romutil.layout.milp.solver import solve_and_extract
from romutil.layout.protocols import TreeSolver
from romutil.models import Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError


_get_child_boundary_ports = get_child_boundary_ports


def solve_cluster(
    node: SolveTreeNode,
    child_solutions: Mapping[str, NodeSolution] | LayoutGraph | None = None,
    graph: LayoutGraph | None = None,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
) -> NodeSolution:
    """Solve layout cluster embedding constituent rooms and child macro-boxes via MILP.

    Coordinates the canonical 5-stage MILP pipeline:
      1. Inventory & Interface Classification (inventory_cluster)
      2. Basic MILP Formulation (formulate_cluster_model)
      3. Geometric Clearance & Separation Constraints (apply_clearance_constraints)
      4. Solve & Extraction (solve_and_extract)
      5. Coordinate Normalization (normalize_cluster_solution)

    Args:
        node: SolveTreeNode representing the cluster to lay out.
        child_solutions: Mapping of child node_id to already-solved NodeSolution,
            or LayoutGraph if solving a leaf cluster without children.
        graph: Underlying LayoutGraph with room definitions and exit edges.
        options: Optional layout configuration parameters.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').

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

    if not rooms and not children:
        return normalize_empty_cluster(node)

    if len(rooms) == 1 and not children:
        r = rooms[0]
        boundary_ports = get_child_boundary_ports(node)
        return normalize_single_room_cluster(
            node=node,
            room=r,
            is_boundary_port=(r in boundary_ports),
        )

    try:
        # Stage 1: Inventory & Interface Classification
        inventory = inventory_cluster(
            node=node,
            child_solutions=child_solutions,
            graph=graph,
            options=options,
        )

        # Stage 2: Basic MILP Formulation
        formulation = formulate_cluster_model(inventory=inventory)

        # Stage 3: Geometric Clearance & Separation Constraints
        apply_clearance_constraints(formulation=formulation)

        # Stage 4: Solve & Extraction
        extraction = solve_and_extract(
            formulation=formulation,
            solver_name=solver_name,
            options=options,
        )

        # Stage 5: Coordinate Normalization
        return normalize_cluster_solution(
            extraction=extraction,
            node=node,
            children=inventory.children,
            child_solutions=child_solutions,
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
    ) -> NodeSolution:
        """Solve cluster layout non-overlapping coordinates and child offsets."""
        return solve_cluster(
            node=node,
            child_solutions=child_solutions,
            graph=graph,
            options=options,
            solver_name=self.solver_name,
        )


class HierarchicalTreeSolver(TreeSolver):
    """Hierarchical tree solver that solves tree nodes bottom-up via MILP."""

    def __init__(
        self,
        graph: LayoutGraph | None = None,
        solver_name: str = "cbc",
    ) -> None:
        self.graph = graph
        self.solver_name = solver_name

    def solve_tree(
        self,
        tree: SolveTree,
        options: LayoutOptions | None = None,
    ) -> TreeSolution:
        """Recursively solve tree nodes bottom-up (post-order traversal)."""
        layout_graph = self.graph if self.graph is not None else tree.graph
        if layout_graph is None:
            all_rooms: set[int] = set()

            def _collect_rooms(n: SolveTreeNode) -> None:
                all_rooms.update(n.rooms)
                for ch in n.children:
                    _collect_rooms(ch)

            _collect_rooms(tree.root)
            for comp in tree.components:
                _collect_rooms(comp)

            layout_graph = LayoutGraph(
                rooms={v: Room(vnum=v, name=f"Room {v}", description="", exits=()) for v in all_rooms},
                edges=(),
            )

        solutions: dict[str, NodeSolution] = {}

        def _solve_node(node: SolveTreeNode, g: LayoutGraph) -> NodeSolution:
            if node.node_id in solutions:
                return solutions[node.node_id]

            # Post-order traversal: solve all children first
            child_solutions_map: dict[str, NodeSolution] = {}
            for child in node.children:
                child_sol = _solve_node(child, g)
                child_solutions_map[child.node_id] = child_sol

            cluster_sol = solve_cluster(
                node=node,
                child_solutions=child_solutions_map,
                graph=g,
                options=options,
                solver_name=self.solver_name,
            )

            solutions[node.node_id] = cluster_sol
            return cluster_sol

        _solve_node(tree.root, layout_graph)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp, layout_graph)

        return TreeSolution(node_solutions=solutions)


__all__ = [
    "ActiveEdge",
    "EndpointRef",
    "HierarchicalClusterSolver",
    "HierarchicalTreeSolver",
    "solve_cluster",
]
