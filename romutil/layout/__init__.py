"""Hierarchical layout solver pipeline framework."""

from __future__ import annotations

from romutil.layout.engine import (
    HierarchicalLayoutSolver,
    IdentityLayoutAssembler,
    PassthroughSimplifier,
    PlaceholderTreeSolver,
    TrivialTreeBuilder,
)
from romutil.layout.geometry import (
    BoundingBox3D,
    Segment3D,
    point_in_segment_interior,
    segment_box_penetration,
    segment_room_penetration,
    segments_intersect_3d,
)
from romutil.layout.hierarchy import (
    BlockCutTreeBuilder,
    SpqrTreeBuilder,
    build_block_spqr_tree,
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
    "BoundingBox3D",
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
    "Segment3D",
    "SimplifiedGraph",
    "SolveTree",
    "SolveTreeNode",
    "SpqrTreeBuilder",
    "TreeBuilder",
    "TreeSolution",
    "TreeSolver",
    "TrivialTreeBuilder",
    "WarpSeverer",
    "build_block_spqr_tree",
    "direction_to_displacement",
    "expand_corridor_coordinates",
    "is_euclidean_satisfiable",
    "point_in_segment_interior",
    "segment_box_penetration",
    "segment_room_penetration",
    "segments_intersect_3d",
]
