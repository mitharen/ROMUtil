"""Bottom-up tree traversal and MILP solving for hierarchical layout trees."""

from __future__ import annotations

from romutil.layout.ir import (
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.milp.cluster import solve_cluster
from romutil.layout.protocols import TreeSolver
from romutil.models import Room
from romutil.types import LayoutOptions


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

        if options and options.debug_solve_dir:
            from romutil.layout.milp.debug import render_debug_solve_tree_index

            render_debug_solve_tree_index(tree, options.debug_solve_dir)

        return TreeSolution(node_solutions=solutions)


__all__ = ["HierarchicalTreeSolver"]
