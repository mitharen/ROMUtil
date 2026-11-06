"""Unified Stage 3 hierarchical tree transformation pipeline builder."""

from __future__ import annotations

from typing import Any

from romutil.layout.hierarchy.block_cut import BlockCutDecompositionPass
from romutil.layout.hierarchy.spqr import SpqrDecompositionPass
from romutil.layout.hierarchy.transforms import (
    TransverseSNodeSplitPass,
    TreeTransformationPass,
    TreeTransformationPipeline,
)
from romutil.layout.ir import LayoutGraph, SimplifiedGraph, SolveTree
from romutil.layout.protocols import TreeBuilder
from romutil.types import LayoutOptions


class PipelineTreeBuilder(TreeBuilder):
    """Stage 3 tree builder executing a sequential pipeline of tree transformation passes."""

    def __init__(
        self,
        pipeline: TreeTransformationPipeline | None = None,
        *,
        block_cut_builder: Any = None,
    ) -> None:
        if pipeline is not None:
            self._pipeline = pipeline
        elif block_cut_builder is not None:
            if hasattr(block_cut_builder, "_pass") and isinstance(
                block_cut_builder._pass, TreeTransformationPass
            ):
                first_pass: TreeTransformationPass = block_cut_builder._pass
            else:
                first_pass = BlockCutDecompositionPass()
            self._pipeline = TreeTransformationPipeline([
                first_pass,
                SpqrDecompositionPass(),
                TransverseSNodeSplitPass(),
            ])
        else:
            self._pipeline = self.default_pipeline()

    @property
    def pipeline(self) -> TreeTransformationPipeline:
        """Underlying transformation pipeline."""
        return self._pipeline

    @classmethod
    def default_pipeline(cls, graph: LayoutGraph | None = None) -> TreeTransformationPipeline:
        """Create canonical hierarchical tree transformation pipeline."""
        return TreeTransformationPipeline([
            BlockCutDecompositionPass(graph=graph),
            SpqrDecompositionPass(graph=graph),
            TransverseSNodeSplitPass(graph=graph),
        ])

    def build_tree(
        self,
        simplified: SimplifiedGraph,
        options: LayoutOptions | None = None,
    ) -> SolveTree:
        """Decompose a simplified graph into a hierarchical SolveTree via the pipeline."""
        seed = SolveTree.seed(simplified.graph)
        return self._pipeline.apply(seed)


SpqrTreeBuilder = PipelineTreeBuilder
