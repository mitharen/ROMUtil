"""Hierarchical layout solver pipeline framework."""

from __future__ import annotations

from romutil.layout.engine import (
    HierarchicalLayoutSolver,
    IdentityLayoutAssembler,
    PassthroughSimplifier,
    PlaceholderTreeSolver,
    TrivialTreeBuilder,
)
from romutil.layout.ir import (
    CorridorSegment,
    LayoutEdge,
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

__all__ = [
    "CorridorSegment",
    "GraphSimplifier",
    "HierarchicalLayoutSolver",
    "IdentityLayoutAssembler",
    "LayoutAssembler",
    "LayoutEdge",
    "LayoutGraph",
    "NodeSolution",
    "PassthroughSimplifier",
    "PlaceholderTreeSolver",
    "SimplifiedGraph",
    "SolveTree",
    "SolveTreeNode",
    "TreeBuilder",
    "TreeSolution",
    "TreeSolver",
    "TrivialTreeBuilder",
]
