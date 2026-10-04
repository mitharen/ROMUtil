"""Hierarchical layout solver pipeline framework."""

from __future__ import annotations

from romutil.layout.engine import (
    HierarchicalLayoutSolver,
    IdentityLayoutAssembler,
    PassthroughSimplifier,
    PlaceholderTreeSolver,
    TrivialTreeBuilder,
)
from romutil.layout.hierarchy import (
    BlockCutTreeBuilder,
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
from romutil.layout.simplification import (
    ChainedSimplifier,
    HallwayCompactor,
    WarpSeverer,
    direction_to_displacement,
    expand_corridor_coordinates,
    is_euclidean_satisfiable,
)

__all__ = [
    "BlockCutTreeBuilder",
    "ChainedSimplifier",
    "CorridorSegment",
    "GraphSimplifier",
    "HallwayCompactor",
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
    "WarpSeverer",
    "direction_to_displacement",
    "expand_corridor_coordinates",
    "is_euclidean_satisfiable",
]
