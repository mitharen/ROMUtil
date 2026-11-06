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
    reroot_spqr_tree,
)
from romutil.layout.hierarchy.transforms import (
    AnchorStrippingPass,
    TransverseSNodeSplitPass,
    TreeTransformationPass,
    TreeTransformationPipeline,
    attach_articulation_children,
    strip_anchor,
)

__all__ = [
    "AnchorStrippingPass",
    "BlockCutDecompositionPass",
    "BlockCutTreeBuilder",
    "PipelineTreeBuilder",
    "SpqrDecompositionPass",
    "SpqrTreeBuilder",
    "TransverseSNodeSplitPass",
    "TreeTransformationPass",
    "TreeTransformationPipeline",
    "attach_articulation_children",
    "build_block_spqr_tree",
    "reroot_spqr_tree",
    "strip_anchor",
]
