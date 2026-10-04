"""Pipeline engine and baseline stage implementations for hierarchical layout solving."""

from __future__ import annotations

import dataclasses
import time

from romutil.interfaces import BaseLayoutSolver
from romutil.layout.assembly import HierarchicalLayoutAssembler
from romutil.layout.ir import (
    LayoutGraph,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.protocols import (
    GraphSimplifier,
    LayoutAssembler,
    TreeBuilder,
    TreeSolver,
)
from romutil.layout.simplification.hallways import expand_corridor_coordinates
from romutil.types import (
    Coord3D,
    LayoutDiagnostics,
    LayoutOptions,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


class PassthroughSimplifier:
    """Default stage 2 simplifier passing the layout graph through unchanged."""

    def simplify(self, graph: LayoutGraph, options: LayoutOptions) -> SimplifiedGraph:
        """Return a SimplifiedGraph wrapping the unmodified LayoutGraph."""
        return SimplifiedGraph(graph=graph)


class TrivialTreeBuilder:
    """Default stage 3 tree builder grouping all rooms into a single root node."""

    def build_tree(self, simplified: SimplifiedGraph, options: LayoutOptions) -> SolveTree:
        """Construct a single-node SolveTree from the simplified graph rooms."""
        root = SolveTreeNode(
            node_id="root",
            node_type="root",
            rooms=frozenset(simplified.graph.rooms.keys()),
        )
        return SolveTree(root=root, components=(root,), graph=simplified.graph)


class PlaceholderTreeSolver:
    """Default stage 4 tree solver placing all rooms at origin coordinates."""

    def solve_tree(self, tree: SolveTree, options: LayoutOptions) -> TreeSolution:
        """Assign origin coordinates to rooms for each node in the tree."""
        nodes: dict[str, SolveTreeNode] = {}

        def _traverse(node: SolveTreeNode) -> None:
            if node.node_id not in nodes:
                nodes[node.node_id] = node
                for child in node.children:
                    _traverse(child)

        _traverse(tree.root)
        for comp in tree.components:
            _traverse(comp)

        node_solutions: dict[str, NodeSolution] = {}
        for node_id, node in nodes.items():
            local_coords = {vnum: Coord3D(0, 0, 0) for vnum in node.rooms}
            bbox = (Coord3D(0, 0, 0), Coord3D(0, 0, 0)) if local_coords else None
            node_solutions[node_id] = NodeSolution(
                node_id=node_id,
                local_coordinates=local_coords,
                bounding_box=bbox,
            )

        return TreeSolution(node_solutions=node_solutions)


class IdentityLayoutAssembler:
    """Default stage 5 assembler aggregating coordinates from all node solutions."""

    def assemble(
        self,
        tree: SolveTree,
        solution: TreeSolution,
        simplified: SimplifiedGraph,
        options: LayoutOptions,
    ) -> SolvedLayout:
        """Aggregate local coordinates into a unified SolvedLayout."""
        all_coords: dict[int, Coord3D] = {}
        for node_sol in solution.node_solutions.values():
            all_coords.update(node_sol.local_coordinates)

        for corridor in simplified.corridors:
            if corridor.start_vnum in all_coords and corridor.end_vnum in all_coords:
                intermediate_coords = expand_corridor_coordinates(
                    corridor,
                    all_coords[corridor.start_vnum],
                    all_coords[corridor.end_vnum],
                )
                all_coords.update(intermediate_coords)

        return SolvedLayout(
            coordinates=all_coords,
            diagnostics=LayoutDiagnostics(
                status=SolverStatus.FEASIBLE,
                duration_seconds=0.0,
            ),
        )


class HierarchicalLayoutSolver:
    """Pipeline layout solver implementing BaseLayoutSolver via decoupled stages."""

    def __init__(
        self,
        simplifier: GraphSimplifier | None = None,
        tree_builder: TreeBuilder | None = None,
        tree_solver: TreeSolver | None = None,
        assembler: LayoutAssembler | None = None,
    ) -> None:
        self.simplifier: GraphSimplifier = simplifier if simplifier is not None else PassthroughSimplifier()
        self.tree_builder: TreeBuilder = tree_builder if tree_builder is not None else TrivialTreeBuilder()
        self.tree_solver: TreeSolver = tree_solver if tree_solver is not None else PlaceholderTreeSolver()
        self.assembler: LayoutAssembler = assembler if assembler is not None else HierarchicalLayoutAssembler()

    @classmethod
    def production(
        cls,
        simplifier: GraphSimplifier | None = None,
        tree_builder: TreeBuilder | None = None,
        tree_solver: TreeSolver | None = None,
        assembler: LayoutAssembler | None = None,
    ) -> HierarchicalLayoutSolver:
        """Construct a HierarchicalLayoutSolver configured with standard production stages."""
        from romutil.layout.assembly import HierarchicalLayoutAssembler
        from romutil.layout.hierarchy import SpqrTreeBuilder
        from romutil.layout.milp import HierarchicalTreeSolver
        from romutil.layout.simplification import (
            ChainedSimplifier,
            HallwayCompactor,
            WarpSeverer,
        )

        actual_simplifier = (
            simplifier
            if simplifier is not None
            else ChainedSimplifier([HallwayCompactor(), WarpSeverer()])
        )
        actual_tree_builder = tree_builder if tree_builder is not None else SpqrTreeBuilder()
        actual_tree_solver = tree_solver if tree_solver is not None else HierarchicalTreeSolver()
        actual_assembler = assembler if assembler is not None else HierarchicalLayoutAssembler()

        return cls(
            simplifier=actual_simplifier,
            tree_builder=actual_tree_builder,
            tree_solver=actual_tree_solver,
            assembler=actual_assembler,
        )

    def solve(
        self,
        graph: WorldGraph,
        options: LayoutOptions | None = None,
    ) -> SolvedLayout:
        """Execute the 5-stage layout pipeline to compute 3D spatial layout coordinates."""
        opts = options if options is not None else LayoutOptions()
        start_time = time.perf_counter()

        # Stage 1: Ingestion
        layout_graph = LayoutGraph.from_world_graph(graph)

        # Stage 2: Simplification
        simplified = self.simplifier.simplify(layout_graph, opts)

        # Stage 3: Tree Building
        tree = self.tree_builder.build_tree(simplified, opts)

        # Stage 4: Tree Solving
        solution = self.tree_solver.solve_tree(tree, opts)

        # Stage 5: Assembly
        layout = self.assembler.assemble(tree, solution, simplified, opts)

        duration = time.perf_counter() - start_time
        updated_diagnostics = dataclasses.replace(layout.diagnostics, duration_seconds=duration)
        return dataclasses.replace(layout, diagnostics=updated_diagnostics)
