"""Hierarchical decomposition tree builders and transformation passes."""

from __future__ import annotations

from romutil.layout.hierarchy.block_cut import (
    BlockCutDecompositionPass,
    BlockCutTreeBuilder,
)
from romutil.layout.hierarchy.pipeline import (
    PipelineTreeBuilder,
    SpqrTreeBuilder,
)
from romutil.layout.hierarchy.spqr import (
    SpqrDecompositionPass,
    build_block_spqr_tree,
)
from romutil.layout.hierarchy.transforms import (
    AnchorStrippingPass,
    ArticulationChildAttachmentPass,
    SplitPairAllowlistFusionPass,
    TransverseSNodeSplitPass,
    TreeTransformationPass,
    TreeTransformationPipeline,
    attach_articulation_children,
    strip_anchor,
)

__all__ = [
    "AnchorStrippingPass",
    "ArticulationChildAttachmentPass",
    "BlockCutDecompositionPass",
    "BlockCutTreeBuilder",
    "PipelineTreeBuilder",
    "SplitPairAllowlistFusionPass",
    "SpqrDecompositionPass",
    "SpqrTreeBuilder",
    "TransverseSNodeSplitPass",
    "TreeTransformationPass",
    "TreeTransformationPipeline",
    "attach_articulation_children",
    "build_block_spqr_tree",
    "strip_anchor",
]
