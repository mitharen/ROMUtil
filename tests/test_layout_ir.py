"""Unit tests for layout intermediate representations and domain types."""

from __future__ import annotations

import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.models import Room
from romutil.types import Coord3D


class TestDomainEnums:
    """Tests for TreeNodeType and CutType canonical enums."""

    def test_tree_node_type_members(self) -> None:
        """Verify TreeNodeType enum variants inherit from str and match domain values."""
        assert TreeNodeType.WORLD_ROOT == "world_root"
        assert TreeNodeType.RIGID_CLUSTER == "rigid_cluster"
        assert TreeNodeType.SERIES_LOOP == "series_loop"
        assert TreeNodeType.SPLIT_PAIR_CONTAINER == "split_pair_container"
        assert TreeNodeType.BLOCK == "block"
        assert TreeNodeType.BICONNECTED_BLOCK == "biconnected_block"
        assert TreeNodeType.EMPTY == "empty"

        assert isinstance(TreeNodeType.RIGID_CLUSTER, str)
        assert isinstance(TreeNodeType.RIGID_CLUSTER, TreeNodeType)

    def test_cut_type_members(self) -> None:
        """Verify CutType enum variants inherit from str and match domain values."""
        assert CutType.ARTICULATION_POINT == "articulation_point"
        assert CutType.BRIDGE == "bridge"
        assert CutType.SPLIT_PAIR == "split_pair"
        assert CutType.WARP_CYCLE == "warp_cycle"

        assert isinstance(CutType.SPLIT_PAIR, str)
        assert isinstance(CutType.SPLIT_PAIR, CutType)


class TestCutContracts:
    """Tests for cut contract dataclasses."""

    def test_split_pair_cut(self) -> None:
        """Verify SplitPairCut normalization and properties."""
        cut = SplitPairCut(poles=(10, 20))
        assert cut.poles == (10, 20)

        # List input is normalized to tuple
        cut_from_list = SplitPairCut(poles=[30, 40])  # type: ignore[arg-type]
        assert cut_from_list.poles == (30, 40)
        assert isinstance(cut_from_list.poles, tuple)

    def test_articulation_point_cut(self) -> None:
        """Verify ArticulationPointCut initialization."""
        cut = ArticulationPointCut(anchor_vnum=42)
        assert cut.anchor_vnum == 42

    def test_bridge_cut(self) -> None:
        """Verify BridgeCut normalization and defaults."""
        cut = BridgeCut(bridge_edge=(1, 2), direction=0)
        assert cut.bridge_edge == (1, 2)
        assert cut.direction == 0
        assert cut.weight == 1

        cut_weighted = BridgeCut(bridge_edge=[3, 4], direction=1, weight=5)  # type: ignore[arg-type]
        assert cut_weighted.bridge_edge == (3, 4)
        assert isinstance(cut_weighted.bridge_edge, tuple)
        assert cut_weighted.weight == 5


class TestSolveTreeNodeDomainProperties:
    """Tests for SolveTreeNode type normalization and cut inspection properties."""

    def test_solve_tree_node_type_normalization(self) -> None:
        """SolveTreeNode normalizes valid string types to TreeNodeType enums."""
        node_str = SolveTreeNode(
            node_id="n1",
            node_type="rigid_cluster",
            rooms={1, 2},
        )
        assert isinstance(node_str.node_type, TreeNodeType)
        assert node_str.node_type is TreeNodeType.RIGID_CLUSTER

        node_enum = SolveTreeNode(
            node_id="n2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms={1, 2},
        )
        assert node_enum.node_type is TreeNodeType.SERIES_LOOP

        custom_node = SolveTreeNode(
            node_id="n3",
            node_type="custom_type",
            rooms={1},
        )
        assert custom_node.node_type == "custom_type"
        assert not isinstance(custom_node.node_type, TreeNodeType)

    def test_cut_type_and_contract_split_pair(self) -> None:
        """SolveTreeNode inspects split-pair cut contract from metadata."""
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms={1, 2, 3},
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 3),
            },
        )
        assert node.cut_type == CutType.SPLIT_PAIR
        contract = node.cut_contract
        assert isinstance(contract, SplitPairCut)
        assert contract.poles == (1, 3)

    def test_cut_type_and_contract_explicit_instance(self) -> None:
        """SolveTreeNode returns explicit cut_contract if stored in metadata."""
        contract = SplitPairCut(poles=(100, 200))
        node = SolveTreeNode(
            node_id="sp_explicit",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms={100, 200},
            metadata={"cut_contract": contract},
        )
        assert node.cut_type == CutType.SPLIT_PAIR
        assert node.cut_contract is contract

    def test_cut_type_and_contract_articulation(self) -> None:
        """SolveTreeNode inspects articulation point cut contract."""
        node = SolveTreeNode(
            node_id="ap",
            node_type=TreeNodeType.BLOCK,
            rooms={10, 20},
            metadata={
                "cut_type": "articulation_point",
                "anchor_vnum": 10,
            },
        )
        assert node.cut_type == CutType.ARTICULATION_POINT
        contract = node.cut_contract
        assert isinstance(contract, ArticulationPointCut)
        assert contract.anchor_vnum == 10

    def test_cut_type_and_contract_bridge(self) -> None:
        """SolveTreeNode inspects bridge cut contract."""
        node = SolveTreeNode(
            node_id="br",
            node_type=TreeNodeType.BLOCK,
            rooms={30, 40},
            metadata={
                "cut_type": "bridge",
                "bridge_edge": (30, 40),
                "direction": 2,
                "weight": 3,
            },
        )
        assert node.cut_type == CutType.BRIDGE
        contract = node.cut_contract
        assert isinstance(contract, BridgeCut)
        assert contract.bridge_edge == (30, 40)
        assert contract.direction == 2
        assert contract.weight == 3

    def test_uncut_node_has_none(self) -> None:
        """Node with empty metadata returns None for cut_type and cut_contract."""
        node = SolveTreeNode(
            node_id="plain",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms={1, 2},
        )
        assert node.cut_type is None
        assert node.cut_contract is None


class TestNodeSolution:
    """Tests for NodeSolution data structure, sub_boxes normalization, and bounding_box property."""

    def test_sub_boxes_default_empty(self) -> None:
        """When sub_boxes is omitted, it defaults to an empty tuple and bounding_box reflects coordinates."""
        sol_empty = NodeSolution(
            node_id="n1",
            local_coordinates={},
        )
        assert sol_empty.sub_boxes == ()
        assert sol_empty.bounding_box is None

        sol_coords = NodeSolution(
            node_id="n1",
            local_coordinates={1: Coord3D(0, 0, 0)},
        )
        assert sol_coords.sub_boxes == ()
        assert sol_coords.bounding_box == BoundingBox3D(0, 0, 0, 0, 0, 0)

    def test_sub_boxes_sequence_normalized_to_tuple(self) -> None:
        """Sequence passed to sub_boxes is normalized to a tuple."""
        box1 = BoundingBox3D(min_x=0, max_x=1, min_y=0, max_y=1, min_z=0, max_z=0)
        box2 = BoundingBox3D(min_x=2, max_x=3, min_y=0, max_y=1, min_z=0, max_z=0)
        sol = NodeSolution(
            node_id="n2",
            local_coordinates={},
            sub_boxes=[box1, box2],
        )
        assert isinstance(sol.sub_boxes, tuple)
        assert sol.sub_boxes == (box1, box2)

    def test_bounding_box_derived_property(self) -> None:
        """Bounding box dynamically computes the minimal enclosing box across all sub-boxes."""
        box1 = BoundingBox3D(min_x=1, max_x=3, min_y=2, max_y=4, min_z=0, max_z=1)
        box2 = BoundingBox3D(min_x=0, max_x=2, min_y=3, max_y=5, min_z=-1, max_z=2)
        sol = NodeSolution(
            node_id="n3",
            local_coordinates={},
            sub_boxes=(box1, box2),
        )
        bbox = sol.bounding_box
        assert bbox is not None
        assert bbox == BoundingBox3D(
            min_x=0,
            max_x=3,
            min_y=2,
            max_y=5,
            min_z=-1,
            max_z=2,
        )

    def test_single_sub_box_bounding_box(self) -> None:
        """Single sub_box returns matching minimal enclosing bounding box."""
        box = BoundingBox3D(min_x=0, max_x=5, min_y=1, max_y=2, min_z=3, max_z=4)
        sol = NodeSolution(
            node_id="n4",
            local_coordinates={},
            sub_boxes=(box,),
        )
        assert sol.bounding_box == box
