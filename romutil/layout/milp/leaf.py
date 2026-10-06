"""MILP formulation and solver delegate for irreducible leaf clusters."""

from __future__ import annotations

from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.layout.ir import (
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.milp.parent import solve_parent_cluster
from romutil.layout.protocols import TreeSolver
from romutil.models import Room
from romutil.types import Coord3D, LayoutOptions


def solve_leaf_cluster(
    node: SolveTreeNode,
    graph: LayoutGraph,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
) -> NodeSolution:
    """Solve 3D integer coordinates for all rooms in a leaf cluster via MILP.

    Args:
        node: Leaf SolveTreeNode representing the cluster to lay out.
        graph: Underlying LayoutGraph with room definitions and exit edges.
        options: Optional layout configuration parameters (e.g. timeout).
        solver_name: Name of the Pyomo MILP solver (defaults to 'cbc').

    Returns:
        NodeSolution containing solved non-overlapping 3D coordinates and bounding box.
    """
    return solve_parent_cluster(
        node=node,
        child_solutions={},
        graph=graph,
        options=options,
        solver_name=solver_name,
    )


class LeafTreeSolver(TreeSolver):
    """Hierarchical tree solver that solves leaf clusters via MILP and aggregates non-leaf boxes."""

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
        """Solve local coordinates for leaf nodes and enclose non-leaf bounding boxes."""
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

        def _solve_node(node: SolveTreeNode) -> NodeSolution:
            if node.node_id in solutions:
                return solutions[node.node_id]

            if not node.children:
                sol = solve_leaf_cluster(
                    node,
                    layout_graph,
                    options,
                    solver_name=self.solver_name,
                )
                solutions[node.node_id] = sol
                return sol

            child_solutions: list[NodeSolution] = [_solve_node(child) for child in node.children]

            child_corners: list[Coord3D] = []
            child_coords: dict[int, Coord3D] = {}
            for c_sol in child_solutions:
                child_coords.update(c_sol.local_coordinates)
                if c_sol.bounding_box is not None:
                    child_corners.extend(c_sol.bounding_box)

            if child_corners:
                enclosing_box = BoundingBox3D.from_coordinates(child_corners)
                bbox: tuple[Coord3D, Coord3D] | None = (
                    Coord3D(enclosing_box.min_x, enclosing_box.min_y, enclosing_box.min_z),
                    Coord3D(enclosing_box.max_x, enclosing_box.max_y, enclosing_box.max_z),
                )
            else:
                bbox = None

            node_local_coords = {
                v: child_coords.get(v, Coord3D(0, 0, 0))
                for v in node.rooms
            }

            sol = NodeSolution(
                node_id=node.node_id,
                local_coordinates=node_local_coords,
                bounding_box=bbox,
            )
            solutions[node.node_id] = sol
            return sol

        _solve_node(tree.root)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp)

        return TreeSolution(node_solutions=solutions)
