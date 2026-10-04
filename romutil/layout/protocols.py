"""Protocols defining the stages of the hierarchical layout solver pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from romutil.layout.ir import LayoutGraph, SimplifiedGraph, SolveTree, TreeSolution
from romutil.types import LayoutOptions, SolvedLayout


@runtime_checkable
class GraphSimplifier(Protocol):
    """Protocol for simplifying a LayoutGraph (e.g. collapsing corridors, severing chords)."""

    def simplify(self, graph: LayoutGraph, options: LayoutOptions) -> SimplifiedGraph:
        """Simplify a LayoutGraph according to layout options."""
        ...


@runtime_checkable
class TreeBuilder(Protocol):
    """Protocol for constructing a hierarchical SolveTree from a SimplifiedGraph."""

    def build_tree(self, simplified: SimplifiedGraph, options: LayoutOptions) -> SolveTree:
        """Decompose a simplified graph into a SolveTree."""
        ...


@runtime_checkable
class TreeSolver(Protocol):
    """Protocol for solving 3D spatial layouts across nodes of a SolveTree."""

    def solve_tree(self, tree: SolveTree, options: LayoutOptions) -> TreeSolution:
        """Solve local coordinates for nodes in the solve tree."""
        ...


@runtime_checkable
class LayoutAssembler(Protocol):
    """Protocol for assembling a unified SolvedLayout from tree solutions and graph."""

    def assemble(
        self,
        tree: SolveTree,
        solution: TreeSolution,
        simplified: SimplifiedGraph,
        options: LayoutOptions,
    ) -> SolvedLayout:
        """Assemble a global SolvedLayout from tree solutions and topology."""
        ...
