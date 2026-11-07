"""End-to-end integration tests for area layout solving and composite area decomposition."""

from __future__ import annotations

from pathlib import Path
import pytest

from romutil.layout import (
    ArticulationPointCut,
    HierarchicalLayoutSolver,
    LayoutGraph,
    PipelineTreeBuilder,
)
from romutil.models import merge_areas
from romutil.parser import parse_file
from romutil.types import WorldGraph

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "areas"


class TestSewerAreaIntegration:
    """Integration tests verifying full spatial layout computation for sewer.are."""

    def test_sewer_spqr_anchor_decomposition_hierarchy(self) -> None:
        """Verify sewer biconnected block reroots SPQR tree at articulation anchor component."""
        area = parse_file(FIXTURES_DIR / "sewer.are")
        world_graph = WorldGraph.from_area_data(area)
        layout_graph = LayoutGraph.from_world_graph(world_graph)

        solver = HierarchicalLayoutSolver.production()
        simplified = solver.simplifier.simplify(layout_graph)
        tree = solver.tree_builder.build_tree(simplified)

        # Locate the sewer biconnected block (block 9 decomposed into SPQR tree)
        target_node = None

        def _find_node(node):
            nonlocal target_node
            if node.node_id == "comp_0_block_9_spqr_0":
                target_node = node
                return
            for child in node.children:
                _find_node(child)

        _find_node(tree.root)
        assert target_node is not None, "Expected comp_0_block_9_spqr_0 in solve tree"

        # The SPQR root must have had anchor 7402 stripped from its room set
        assert 7402 not in target_node.rooms
        # The SPQR root must directly own the ingress rooms incident to anchor 7402
        assert 7403 in target_node.rooms
        assert 7418 in target_node.rooms
        # Contract must preserve articulation point gateway cut
        assert isinstance(target_node.cut_contract, ArticulationPointCut)
        assert target_node.cut_contract.anchor_vnum == 7402


class TestMidgaardSewerCompositeIntegration:
    """Integration tests verifying multi-area composite merging and decomposition for Midgaard + Sewers."""

    def test_midgaard_sewer_composite_merging_and_hierarchy(self) -> None:
        """Verify Midgaard and Sewers merge into a unified WorldGraph with cross-area exits and valid solve tree."""
        midgaard = parse_file(FIXTURES_DIR / "midgaard.are")
        sewer = parse_file(FIXTURES_DIR / "sewer.are")
        composite = merge_areas([midgaard, sewer])

        world_graph = WorldGraph.from_area_data(composite)
        assert len(world_graph.rooms) == 320

        # Verify cross-area exits between Midgaard and Sewers are preserved
        r3002 = world_graph.rooms[3002]
        r3030 = world_graph.rooms[3030]
        r7030 = world_graph.rooms[7030]

        ex_3002 = [ex for ex in r3002.exits if ex.dst_vnum == 7026]
        assert len(ex_3002) == 1
        ex_3030 = [ex for ex in r3030.exits if ex.dst_vnum == 7030]
        assert len(ex_3030) == 1
        ex_7030 = [ex for ex in r7030.exits if ex.dst_vnum == 3030]
        assert len(ex_7030) == 1

        # Verify hierarchical tree decomposition completes cleanly
        solver = HierarchicalLayoutSolver.production()
        layout_graph = LayoutGraph.from_world_graph(world_graph)
        simplified = solver.simplifier.simplify(layout_graph)
        tree = solver.tree_builder.build_tree(simplified)

        assert tree.root is not None
        assert len(tree.components) >= 1

        # Locate the sewer biconnected block in composite (block 46 decomposed into SPQR tree)
        target_node = None

        def _find_node(node):
            nonlocal target_node
            if node.node_id == "comp_0_block_46_spqr_0":
                target_node = node
                return
            for child in node.children:
                _find_node(child)

        _find_node(tree.root)
        assert target_node is not None, "Expected comp_0_block_46_spqr_0 in composite solve tree"

        # Anchor 7402 must be stripped from rooms and ingress rooms 7403, 7418 owned directly by root
        assert 7402 not in target_node.rooms
        assert 7403 in target_node.rooms
        assert 7418 in target_node.rooms
        assert isinstance(target_node.cut_contract, ArticulationPointCut)
        assert target_node.cut_contract.anchor_vnum == 7402
