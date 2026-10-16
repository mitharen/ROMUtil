"""Pipeline engine and baseline stage implementations for hierarchical layout solving."""

from __future__ import annotations

import dataclasses
import time

from romutil.interfaces import BaseLayoutSolver
from romutil.layout.assembly import HierarchicalLayoutAssembler
from romutil.layout.ir import LayoutGraph, SimplifiedGraph, SolveTree, TreeSolution
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
        from romutil.layout.assembly import HierarchicalLayoutAssembler
        from romutil.layout.hierarchy import PipelineTreeBuilder
        from romutil.layout.milp import HierarchicalTreeSolver
        from romutil.layout.simplification import (
            ChainedSimplifier,
            HallwayCompactor,
            WarpSeverer,
        )

        self.simplifier: GraphSimplifier = (
            simplifier
            if simplifier is not None
            else ChainedSimplifier([HallwayCompactor(), WarpSeverer()])
        )
        self.tree_builder: TreeBuilder = (
            tree_builder if tree_builder is not None else PipelineTreeBuilder()
        )
        self.tree_solver: TreeSolver = (
            tree_solver if tree_solver is not None else HierarchicalTreeSolver()
        )
        self.assembler: LayoutAssembler = (
            assembler if assembler is not None else HierarchicalLayoutAssembler()
        )

    @classmethod
    def production(
        cls,
        simplifier: GraphSimplifier | None = None,
        tree_builder: TreeBuilder | None = None,
        tree_solver: TreeSolver | None = None,
        assembler: LayoutAssembler | None = None,
    ) -> HierarchicalLayoutSolver:
        """Construct a HierarchicalLayoutSolver configured with standard production stages."""
        return cls(
            simplifier=simplifier,
            tree_builder=tree_builder,
            tree_solver=tree_solver,
            assembler=assembler,
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
