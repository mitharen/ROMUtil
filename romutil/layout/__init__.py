"""Hierarchical layout solver pipeline framework."""

from __future__ import annotations

from romutil.layout.assembly import HierarchicalLayoutAssembler
from romutil.layout.engine import (
    HierarchicalLayoutSolver,
    IdentityLayoutAssembler,
    PassthroughSimplifier,
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
    AnchorStrippingPass,
    ArticulationChildAttachmentPass,
    BlockCutTreeBuilder,
    SplitPairAllowlistFusionPass,
    SpqrTreeBuilder,
    TransverseSNodeSplitPass,
    TreeTransformationPass,
    TreeTransformationPipeline,
    build_block_spqr_tree,
)
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CorridorSegment,
    CutType,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
    TreeSolution,
)
from romutil.layout.milp import (
    HierarchicalClusterSolver,
    HierarchicalTreeSolver,
    solve_cluster,
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
from romutil.types import LayoutSolverError

production = HierarchicalLayoutSolver.production

__all__ = [
    "AnchorStrippingPass",
    "ArticulationChildAttachmentPass",
    "ArticulationPointCut",
    "BlockCutTreeBuilder",
    "BoundingBox3D",
    "BridgeCut",
    "ChainedSimplifier",
    "CorridorSegment",
    "CutType",
    "GraphSimplifier",
    "HallwayCompactor",
    "HierarchicalClusterSolver",
    "HierarchicalLayoutAssembler",
    "HierarchicalLayoutSolver",
    "HierarchicalTreeSolver",
    "IdentityLayoutAssembler",
    "LayoutAssembler",
    "LayoutEdge",
    "LayoutGraph",
    "LayoutSolverError",
    "NodeSolution",
    "PassthroughSimplifier",
    "production",
    "Segment3D",
    "SimplifiedGraph",
    "SolveTree",
    "SolveTreeNode",
    "SplitPairAllowlistFusionPass",
    "SplitPairCut",
    "SpqrTreeBuilder",
    "TransverseSNodeSplitPass",
    "TreeBuilder",
    "TreeNodeType",
    "TreeSolution",
    "TreeSolver",
    "TreeTransformationPass",
    "TreeTransformationPipeline",
    "WarpSeverer",
    "build_block_spqr_tree",
    "direction_to_displacement",
    "expand_corridor_coordinates",
    "is_euclidean_satisfiable",
    "point_in_segment_interior",
    "segment_box_penetration",
    "segment_room_penetration",
    "segments_intersect_3d",
    "solve_cluster",
]
