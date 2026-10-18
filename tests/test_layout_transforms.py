"""Unit tests for solve tree transformation passes and composition pipelines."""

from __future__ import annotations

import pytest

from romutil.layout.hierarchy import (
    AnchorStrippingPass,
    ArticulationChildAttachmentPass,
    BlockCutDecompositionPass,
    BlockCutTreeBuilder,
    PipelineTreeBuilder,
    SplitPairAllowlistFusionPass,
    SpqrDecompositionPass,
    SpqrTreeBuilder,
    TransverseSNodeSplitPass,
    TreeTransformationPass,
    TreeTransformationPipeline,
    attach_articulation_children,
    strip_anchor,
)
from romutil.layout.hierarchy.transforms import _get_anchor_vnum
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.models import Direction, Exit, Room


def _make_room(vnum: int) -> Room:
    """Helper to construct a simple Room."""
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=(),
    )


def _make_graph(
    rooms: list[int],
    edges: list[tuple[int, int, Direction]],
    cuts: Sequence[tuple[int, int]] | None = None,
) -> LayoutGraph:
    """Helper to construct a LayoutGraph."""
    cut_set = set(cuts) if cuts else set()
    return LayoutGraph(
        rooms={v: _make_room(v) for v in rooms},
        edges=tuple(
            LayoutEdge(
                src=u,
                dst=v,
                direction=d,
                cut=((u, v) in cut_set or (v, u) in cut_set),
            )
            for u, v, d in edges
        ),
    )


class TestTransformationPassProtocol:
    """Verify runtime protocol conformance for all transformation passes and pipelines."""

    def test_pipeline_implements_transformation_pass_protocol(self) -> None:
        pipeline = TreeTransformationPipeline()
        assert isinstance(pipeline, TreeTransformationPass)

    def test_transverse_s_node_split_pass_implements_protocol(self) -> None:
        pass_instance = TransverseSNodeSplitPass()
        assert isinstance(pass_instance, TreeTransformationPass)

    def test_articulation_child_attachment_pass_implements_protocol(self) -> None:
        pass_instance = ArticulationChildAttachmentPass()
        assert isinstance(pass_instance, TreeTransformationPass)

    def test_anchor_stripping_pass_implements_protocol(self) -> None:
        pass_instance = AnchorStrippingPass()
        assert isinstance(pass_instance, TreeTransformationPass)

    def test_split_pair_allowlist_fusion_pass_implements_protocol(self) -> None:
        pass_instance = SplitPairAllowlistFusionPass()
        assert isinstance(pass_instance, TreeTransformationPass)


class TestTreeTransformationPipeline:
    """Verify behavior of TreeTransformationPipeline."""

    def test_empty_pipeline_returns_tree_unchanged(self) -> None:
        pipeline = TreeTransformationPipeline()
        assert pipeline.passes == ()

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
        )
        tree = SolveTree(root=root, components=(root,))
        transformed = pipeline.apply(tree)
        assert transformed is tree

    def test_add_pass_and_sequential_execution(self) -> None:
        call_order: list[str] = []

        class FirstPass:
            def apply(self, tree: SolveTree) -> SolveTree:
                call_order.append("first")
                return tree

        class SecondPass:
            def apply(self, tree: SolveTree) -> SolveTree:
                call_order.append("second")
                return tree

        pipeline = TreeTransformationPipeline([FirstPass()])
        pipeline.add_pass(SecondPass())
        assert len(pipeline.passes) == 2

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
        )
        tree = SolveTree(root=root, components=(root,))
        pipeline.apply(tree)
        assert call_order == ["first", "second"]

    def test_pipeline_composition_and_nesting(self) -> None:
        log: list[str] = []

        class LoggingPass:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def apply(self, tree: SolveTree) -> SolveTree:
                log.append(self.tag)
                return tree

        inner = TreeTransformationPipeline([LoggingPass("inner1"), LoggingPass("inner2")])
        outer = TreeTransformationPipeline([LoggingPass("outer_pre"), inner, LoggingPass("outer_post")])

        root = SolveTreeNode(node_id="r", node_type=TreeNodeType.RIGID_CLUSTER, rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(root,))
        outer.apply(tree)

        assert log == ["outer_pre", "inner1", "inner2", "outer_post"]


class TestArticulationChildAttachmentPass:
    """Verify ArticulationChildAttachmentPass behavior."""

    def test_empty_tree(self) -> None:
        pass_instance = ArticulationChildAttachmentPass()
        empty_root = SolveTreeNode(node_id="empty", node_type=TreeNodeType.EMPTY, rooms=frozenset())
        tree = SolveTree(root=empty_root, components=())
        assert pass_instance.apply(tree) is tree

    def test_world_root_traversal(self) -> None:
        pass_instance = ArticulationChildAttachmentPass()
        comp1 = SolveTreeNode(node_id="c1", node_type=TreeNodeType.RIGID_CLUSTER, rooms=frozenset({1, 2}))
        comp2 = SolveTreeNode(node_id="c2", node_type=TreeNodeType.RIGID_CLUSTER, rooms=frozenset({3, 4}))
        world = SolveTreeNode(
            node_id="world",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(comp1, comp2),
        )
        tree = SolveTree(root=world, components=(comp1, comp2))
        res = pass_instance.apply(tree)
        assert res.root.node_type == TreeNodeType.WORLD_ROOT
        assert len(res.root.children) == 2

    def test_articulation_child_attached_to_spqr_container_internal_branch(self) -> None:
        # Parent container has rooms {1, 2} (poles) and child S-nodes
        s_node_1 = SolveTreeNode(
            node_id="s_1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": (1, 2)},
        )
        s_node_2 = SolveTreeNode(
            node_id="s_2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": (1, 2)},
        )
        # Articulation child attaching at room 3
        art_child = SolveTreeNode(
            node_id="art_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({5, 6, 7}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 3,
                "cut_contract": ArticulationPointCut(anchor_vnum=3),
            },
        )
        # Initially, all are children of the parent container
        parent = SolveTreeNode(
            node_id="container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(s_node_1, s_node_2, art_child),
        )
        tree = SolveTree(root=parent, components=(parent,))

        pass_instance = ArticulationChildAttachmentPass()
        transformed = pass_instance.apply(tree)

        # art_block should be re-attached under s_1 (which contains room 3)
        assert len(transformed.root.children) == 2
        s1_transformed = [c for c in transformed.root.children if c.node_id == "s_1"][0]
        assert len(s1_transformed.children) == 1
        assert s1_transformed.children[0].node_id == "art_block"

    def test_articulation_child_attached_to_pole_remains_direct_child(self) -> None:
        s_node = SolveTreeNode(
            node_id="s_1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
        )
        art_child = SolveTreeNode(
            node_id="art_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({5, 6}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 1,
            },
        )
        parent = SolveTreeNode(
            node_id="container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(s_node, art_child),
        )
        tree = SolveTree(root=parent, components=(parent,))

        pass_instance = ArticulationChildAttachmentPass()
        transformed = pass_instance.apply(tree)

        # Anchor 1 is in parent container rooms, so art_child stays on parent
        assert len(transformed.root.children) == 2
        assert any(c.node_id == "art_block" for c in transformed.root.children)

    def test_anchor_stripping_on_spqr_subtree(self) -> None:
        # A child block decomposed into SPQR retains anchor in its rooms initially
        s_node = SolveTreeNode(
            node_id="s_child",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 20}),  # contains parent anchor 10
            children=(),
        )
        spqr_block = SolveTreeNode(
            node_id="spqr_block",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({10, 30}),  # contains parent anchor 10
            children=(s_node,),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 10,
                "cut_contract": ArticulationPointCut(anchor_vnum=10),
            },
        )
        tree = SolveTree(root=spqr_block, components=(spqr_block,))

        pass_instance = ArticulationChildAttachmentPass(strip_anchors=True)
        transformed = pass_instance.apply(tree)

        # Parent anchor 10 must be stripped from spqr_block and s_child rooms
        assert 10 not in transformed.root.rooms
        assert transformed.root.rooms == frozenset({30})
        assert 10 not in transformed.root.children[0].rooms
        assert transformed.root.children[0].rooms == frozenset({20})

    def test_strip_anchors_disabled_flag(self) -> None:
        spqr_block = SolveTreeNode(
            node_id="spqr_block",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 20}),
            children=(),
            metadata={"anchor_vnum": 10},
        )
        tree = SolveTree(root=spqr_block, components=(spqr_block,))

        pass_instance = ArticulationChildAttachmentPass(strip_anchors=False)
        transformed = pass_instance.apply(tree)

        assert 10 in transformed.root.rooms


class TestAnchorStrippingPass:
    """Verify standalone AnchorStrippingPass behavior."""

    def test_empty_tree_no_op(self) -> None:
        pass_instance = AnchorStrippingPass()
        empty_root = SolveTreeNode(node_id="empty", node_type=TreeNodeType.EMPTY, rooms=frozenset())
        tree = SolveTree(root=empty_root, components=())
        assert pass_instance.apply(tree) is tree

    def test_strip_anchor_from_node_and_descendants(self) -> None:
        grandchild = SolveTreeNode(
            node_id="gc",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 30}),
            children=(),
        )
        child = SolveTreeNode(
            node_id="c",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 20}),
            children=(grandchild,),
            metadata={"anchor_vnum": 10},
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 11}),
            children=(child,),
        )
        tree = SolveTree(root=root, components=(root,))

        pass_instance = AnchorStrippingPass()
        transformed = pass_instance.apply(tree)

        # Root retains 10
        assert 10 in transformed.root.rooms
        # Child has anchor_vnum=10, so 10 is stripped from child and grandchild
        assert 10 not in transformed.root.children[0].rooms
        assert transformed.root.children[0].rooms == frozenset({20})
        assert 10 not in transformed.root.children[0].children[0].rooms
        assert transformed.root.children[0].children[0].rooms == frozenset({30})

    def test_idempotence(self) -> None:
        node = SolveTreeNode(
            node_id="n",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 20}),
            children=(),
            metadata={"anchor_vnum": 10},
        )
        tree = SolveTree(root=node, components=(node,))

        pass_instance = AnchorStrippingPass()
        t1 = pass_instance.apply(tree)
        t2 = pass_instance.apply(t1)

        assert t1.root.rooms == t2.root.rooms == frozenset({20})


class TestTransverseSNodeSplitPass:
    """Verify TransverseSNodeSplitPass behavior."""

    def test_empty_tree_or_missing_graph(self) -> None:
        pass_instance = TransverseSNodeSplitPass()
        empty_root = SolveTreeNode(node_id="empty", node_type=TreeNodeType.EMPTY, rooms=frozenset())
        tree = SolveTree(root=empty_root, components=())
        assert pass_instance.apply(tree) is tree

        # Missing graph
        root = SolveTreeNode(node_id="r", node_type=TreeNodeType.RIGID_CLUSTER, rooms=frozenset({1, 2}))
        tree_no_graph = SolveTree(root=root, components=(root,), graph=None)
        assert pass_instance.apply(tree_no_graph) is tree_no_graph

    def test_transverse_corridor_s_node_split_via_pipeline(self) -> None:
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.west),
                (10, 11, Direction.north),
                (11, 20, Direction.north),
                (2, 20, Direction.west),
            ],
        )

        child = SolveTreeNode(
            node_id="s_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)

        pipeline = TreeTransformationPipeline([TransverseSNodeSplitPass()])
        transformed = pipeline.apply(tree)

        assert len(transformed.root.children) == 2
        c_a, c_b = transformed.root.children
        assert c_a.node_type == TreeNodeType.RIGID_CLUSTER
        assert c_b.node_type == TreeNodeType.RIGID_CLUSTER
        assert isinstance(c_a.cut_contract, BridgeCut)
        assert isinstance(c_b.cut_contract, BridgeCut)
        assert {_get_anchor_vnum(c_a), _get_anchor_vnum(c_b)} == {1, 2}
        assert c_a.rooms | c_b.rooms == frozenset({10, 11, 20})

    def test_perpendicular_entrance_corridors_not_split(self) -> None:
        graph = _make_graph(
            rooms=[1, 2, 10, 20],
            edges=[
                (1, 10, Direction.west),
                (10, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_loop_perp",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 20}),
            children=(),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)

        pass_instance = TransverseSNodeSplitPass()
        transformed = pass_instance.apply(tree)
        assert len(transformed.root.children) == 1
        assert transformed.root.children[0].node_id == "s_loop_perp"

    def test_transverse_s_node_split_assigns_bridge_cut_contract(self) -> None:
        """Split transverse S-node assigns BridgeCut contracts with correct directions and weights."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_corridor",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        c_a = [c for c in res.children if c.node_id == "s_corridor_a"][0]
        c_b = [c for c in res.children if c.node_id == "s_corridor_b"][0]

        # Verify contracts
        assert isinstance(c_a.cut_contract, BridgeCut)
        assert c_a.cut_contract.bridge_edge == (1, 10)
        assert c_a.cut_contract.direction == int(Direction.north)
        assert c_a.cut_contract.weight == 1

        assert isinstance(c_b.cut_contract, BridgeCut)
        assert c_b.cut_contract.bridge_edge == (2, 20)
        assert c_b.cut_contract.direction == int(Direction.north)
        assert c_b.cut_contract.weight == 1

        # Verify metadata
        assert c_a.metadata.get("cut_type") == CutType.BRIDGE
        assert c_a.metadata.get("bridge_edge") == (1, 10)
        assert c_a.metadata.get("direction") == int(Direction.north)
        assert c_a.metadata.get("weight") == 1
        assert "anchor_vnum" not in c_a.metadata
        assert "poles" not in c_a.metadata

        assert c_b.metadata.get("cut_type") == CutType.BRIDGE
        assert c_b.metadata.get("bridge_edge") == (2, 20)
        assert c_b.metadata.get("direction") == int(Direction.north)
        assert c_b.metadata.get("weight") == 1
        assert "anchor_vnum" not in c_b.metadata
        assert "poles" not in c_b.metadata

        # Verify disjoint room ownership (poles 1 and 2 excluded)
        assert 1 not in c_a.rooms and 2 not in c_a.rooms
        assert 1 not in c_b.rooms and 2 not in c_b.rooms

    def test_transverse_s_node_split_entrance_edge_reverse_direction(self) -> None:
        """Split transverse S-node correctly handles reversed entrance edges."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (10, 1, Direction.south),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (20, 2, Direction.south),
            ],
        )
        child = SolveTreeNode(
            node_id="s_corridor_rev",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        c_a = [c for c in res.children if c.node_id == "s_corridor_rev_a"][0]
        c_b = [c for c in res.children if c.node_id == "s_corridor_rev_b"][0]

        assert isinstance(c_a.cut_contract, BridgeCut)
        assert c_a.cut_contract.bridge_edge == (1, 10)
        assert c_a.cut_contract.direction == int(Direction.north)

        assert isinstance(c_b.cut_contract, BridgeCut)
        assert c_b.cut_contract.bridge_edge == (2, 20)
        assert c_b.cut_contract.direction == int(Direction.north)

    def test_transverse_s_node_split_ambiguous_entrance_fallback_articulation_point(self) -> None:
        """Ambiguous entrance edges from poles fallback gracefully to ArticulationPointCut."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 20, 21],
            edges=[
                (1, 10, Direction.north),
                (1, 11, Direction.north),
                (10, 11, Direction.north),
                (11, 12, Direction.north),
                (12, 20, Direction.east),
                (20, 21, Direction.north),
                (2, 20, Direction.north),
                (2, 21, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_corridor_ambig",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 20, 21}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        c_a = [c for c in res.children if c.node_id == "s_corridor_ambig_a"][0]
        c_b = [c for c in res.children if c.node_id == "s_corridor_ambig_b"][0]

        assert isinstance(c_a.cut_contract, ArticulationPointCut)
        assert c_a.cut_contract.anchor_vnum == 1
        assert c_a.metadata.get("cut_type") == CutType.ARTICULATION_POINT
        assert c_a.metadata.get("anchor_vnum") == 1
        assert "poles" not in c_a.metadata

        assert isinstance(c_b.cut_contract, ArticulationPointCut)
        assert c_b.cut_contract.anchor_vnum == 2
        assert c_b.metadata.get("cut_type") == CutType.ARTICULATION_POINT
        assert c_b.metadata.get("anchor_vnum") == 2
        assert "poles" not in c_b.metadata


class TestSpqrTreeBuilderPipelineIntegration:
    """Verify SpqrTreeBuilder pipeline integration and customization."""

    def test_custom_pipeline_injection(self) -> None:
        call_log: list[str] = []

        class InterceptPass:
            def apply(self, tree: SolveTree) -> SolveTree:
                call_log.append("intercepted")
                return tree

        custom_pipeline = TreeTransformationPipeline([InterceptPass()])
        builder = SpqrTreeBuilder(pipeline=custom_pipeline)
        assert builder.pipeline is custom_pipeline

        # Single block graph
        room_defs = {v: _make_room(v) for v in (1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
        ]
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms=room_defs, edges=tuple(edges)))

        tree = builder.build_tree(simplified)
        assert call_log == ["intercepted"]
        assert isinstance(tree, SolveTree)

    def test_default_pipeline_composition(self) -> None:
        for builder_cls in (PipelineTreeBuilder, SpqrTreeBuilder):
            pipeline = builder_cls.default_pipeline()
            assert len(pipeline.passes) == 5
            assert isinstance(pipeline.passes[0], BlockCutDecompositionPass)
            assert isinstance(pipeline.passes[1], SpqrDecompositionPass)
            assert isinstance(pipeline.passes[2], SplitPairAllowlistFusionPass)
            assert isinstance(pipeline.passes[3], ArticulationChildAttachmentPass)
            assert isinstance(pipeline.passes[4], TransverseSNodeSplitPass)


class TestTransformHelpersAndEdgeCases:
    """Verify transform helper functions and edge branches."""

    def test_get_child_poles_or_anchor_contracts(self) -> None:
        from romutil.layout.hierarchy.transforms import _get_child_poles_or_anchor

        bridge = SolveTreeNode(
            node_id="br",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(1, 2), direction=0)},
        )
        assert _get_child_poles_or_anchor(bridge) == {2}

        meta_poles = SolveTreeNode(
            node_id="mp",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            metadata={"poles": (3, 4)},
        )
        assert _get_child_poles_or_anchor(meta_poles) == {3, 4}

        meta_anchor = SolveTreeNode(
            node_id="ma",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            metadata={"anchor_vnum": 5},
        )
        assert _get_child_poles_or_anchor(meta_anchor) == {5}

        no_contract = SolveTreeNode(
            node_id="nc",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
        )
        assert _get_child_poles_or_anchor(no_contract) == set()

    def test_is_articulation_child_and_get_anchor(self) -> None:
        from romutil.layout.hierarchy.transforms import _get_anchor_vnum, _is_articulation_child

        node_by_type = SolveTreeNode(
            node_id="at",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            metadata={"cut_type": "articulation_point"},
        )
        assert _is_articulation_child(node_by_type) is True
        assert _get_anchor_vnum(node_by_type) is None

        node_by_meta_anchor = SolveTreeNode(
            node_id="ma",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            metadata={"anchor_vnum": "42"},
        )
        assert _is_articulation_child(node_by_meta_anchor) is True
        assert _get_anchor_vnum(node_by_meta_anchor) == 42

        node_by_bridge = SolveTreeNode(
            node_id="br",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(1, 10), direction=0)},
        )
        assert _is_articulation_child(node_by_bridge) is True
        assert _get_anchor_vnum(node_by_bridge) == 1

        node_by_bridge_meta = SolveTreeNode(
            node_id="bm",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10}),
            metadata={"bridge_edge": (2, 10)},
        )
        assert _is_articulation_child(node_by_bridge_meta) is True
        assert _get_anchor_vnum(node_by_bridge_meta) == 2

    def test_attach_articulation_children_unmatched_anchor(self) -> None:
        spqr_node = SolveTreeNode(
            node_id="spqr",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(
                SolveTreeNode(node_id="ch1", node_type=TreeNodeType.RIGID_CLUSTER, rooms=frozenset({3})),
            ),
        )
        art_child = SolveTreeNode(
            node_id="orphan",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({99}),
            metadata={"anchor_vnum": 50},  # Neither in rooms nor in ch1
        )
        res = attach_articulation_children(spqr_node, [art_child])
        # Unmatched anchor stays on direct children
        assert any(c.node_id == "orphan" for c in res.children)

    def test_transverse_s_node_split_reverse_edges_and_axes(self) -> None:
        # Test reverse-directed entrances (dst == p0 / p1) and vertical entrance axis
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (10, 1, Direction.south),  # dst == p0, dir=south -> disp=(0, -1, 0), axis 1
                (10, 11, Direction.east),   # transverse step along X
                (11, 20, Direction.north),
                (20, 2, Direction.south),  # dst == p1, dir=south -> disp=(0, -1, 0), axis 1
            ],
        )
        child = SolveTreeNode(
            node_id="s_vert",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(),
            metadata={"poles": [1, 2]},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2

    def test_transverse_s_node_split_mismatched_entrance_axis(self) -> None:
        # Entrance at p0 is West (axis 0), at p1 is North (axis 1) -> axis mismatch
        graph = _make_graph(
            rooms=[1, 2, 10, 20],
            edges=[
                (1, 10, Direction.west),
                (10, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_mismatch",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 20}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 1

    def test_transverse_s_node_split_child_partition_fallback(self) -> None:
        # Nested child with no poles or anchor -> falls back to subtree overlap partition
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.west),
                (10, 11, Direction.north),
                (11, 20, Direction.north),
                (2, 20, Direction.west),
            ],
        )
        nested = SolveTreeNode(
            node_id="nested_orphan",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10}),  # Belongs to comp_a
            children=(),
        )
        child = SolveTreeNode(
            node_id="s_with_nested",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(nested,),
            metadata={"poles": (1, 2)},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        c_1 = [c for c in res.children if _get_anchor_vnum(c) == 1][0]
        assert len(c_1.children) == 1
        assert c_1.children[0].node_id == "nested_orphan"

    def test_transverse_s_node_split_sibling_bubbling_past_container(self) -> None:
        """S-node under split-pair container splits, bubbles to owning grandparent, and collapses container."""
        # Graph with poles 1, 2 owned by grandparent.
        # Inside s_split (poles 1, 2): 1 -> 10 (North), 10 -> 11 (East), 11 -> 20 (South), 20 -> 2 (South invert).
        # Inside s_other (poles 1, 2): 1 -> 30 (East), 30 -> 2 (East).
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20, 30],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (2, 20, Direction.north),
                (1, 30, Direction.east),
                (30, 2, Direction.east),
            ],
        )
        s_split = SolveTreeNode(
            node_id="s_split",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        s_other = SolveTreeNode(
            node_id="s_other",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({30}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        container = SolveTreeNode(
            node_id="p_container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(s_split, s_other),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        grandparent = SolveTreeNode(
            node_id="grandparent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(container,),
        )

        res = TransverseSNodeSplitPass(graph).transform_node(grandparent)

        # Container is collapsed into promoted s_other, and s_split is bubbled into two articulation clusters
        child_ids = {c.node_id for c in res.children}
        assert "s_other" in child_ids
        assert "s_split_a" in child_ids
        assert "s_split_b" in child_ids
        assert len(res.children) == 3

        c_a = [c for c in res.children if c.node_id == "s_split_a"][0]
        c_b = [c for c in res.children if c.node_id == "s_split_b"][0]
        assert isinstance(c_a.cut_contract, BridgeCut)
        assert isinstance(c_b.cut_contract, BridgeCut)
        assert c_a.cut_contract.bridge_edge == (1, 10)
        assert c_b.cut_contract.bridge_edge == (2, 20)
        assert _get_anchor_vnum(c_a) == 1
        assert _get_anchor_vnum(c_b) == 2
        assert c_a.rooms == frozenset({10})
        assert c_b.rooms == frozenset({11, 20})

    def test_transverse_s_node_split_direct_edge_cycle_preserved(self) -> None:
        """S-node whose poles are directly connected by an edge preserves tight Euclidean cycle and is never severed."""
        # Poles 1 and 2 are directly connected by an edge in graph
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 2, Direction.north),
                (1, 10, Direction.west),
                (10, 11, Direction.north),
                (11, 20, Direction.east),
                (2, 20, Direction.west),
            ],
        )
        child = SolveTreeNode(
            node_id="s_cycle",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )

        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        # s_cycle must remain intact as a single child
        assert len(res.children) == 1
        assert res.children[0].node_id == "s_cycle"

    def test_transverse_s_node_split_compound_child_transverse_span(self) -> None:
        """S-node whose internal spine path is bridged transversely by a child node severs across that child span."""
        # S-node connects to parent 1, 2 via North entrances:
        # 1 -> 10 (North), 2 -> 20 (North).
        # Between 10 and 20, span is bridged by child_bridge (poles 10, 20) with East entrance: 10 -> 30 (East), 30 -> 20 (East).
        graph = _make_graph(
            rooms=[1, 2, 10, 20, 30],
            edges=[
                (1, 10, Direction.north),
                (2, 20, Direction.north),
                (10, 30, Direction.east),
                (30, 20, Direction.east),
            ],
        )
        child_bridge = SolveTreeNode(
            node_id="bridge_child",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({30}),
            children=(),
            metadata={"poles": (10, 20), "cut_type": CutType.SPLIT_PAIR},
        )
        s_compound = SolveTreeNode(
            node_id="s_compound",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 20}),
            children=(child_bridge,),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(s_compound,),
        )

        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        c_a = [c for c in res.children if c.node_id == "s_compound_a"][0]
        c_b = [c for c in res.children if c.node_id == "s_compound_b"][0]
        assert c_a.rooms == frozenset({10})
        assert c_b.rooms == frozenset({20})
        assert len(c_a.children) == 1
        assert c_a.children[0].node_id == "bridge_child"

    def test_transverse_s_node_split_no_owning_ancestor_adopts_at_block_root(self) -> None:
        """When no ancestor owns both boundary poles, split branches are adopted at the block root level."""
        graph = _make_graph(
            rooms=[10, 11, 20],
            edges=[
                (99, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (100, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="orphan_s",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (99, 100)},
        )
        parent = SolveTreeNode(
            node_id="empty_parent",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        assert {c.node_id for c in res.children} == {"orphan_s_a", "orphan_s_b"}

    def test_transverse_s_node_split_child_partition_fallback_side_b(self) -> None:
        """Subtree fallback partition assigns nested child to side B when overlapping with rooms_b."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.west),
                (10, 11, Direction.north),
                (11, 20, Direction.north),
                (2, 20, Direction.west),
            ],
        )
        nested = SolveTreeNode(
            node_id="nested_on_b",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({20}),  # Belongs to comp_b
            children=(),
        )
        child = SolveTreeNode(
            node_id="s_b_nested",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 20}),
            children=(nested,),
            metadata={"poles": (1, 2)},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        c_2 = [c for c in res.children if _get_anchor_vnum(c) == 2][0]
        assert len(c_2.children) == 1
        assert c_2.children[0].node_id == "nested_on_b"

    def test_transverse_s_node_split_empty_internals_and_invalid_poles(self) -> None:
        """S-node with no internal rooms or invalid pole metadata is skipped without error."""
        graph = _make_graph(rooms=[1, 2], edges=[])
        # Only poles in rooms, no internal rooms
        c1 = SolveTreeNode(
            node_id="s_empty_internals",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        # Invalid poles metadata (not length 2)
        c2 = SolveTreeNode(
            node_id="s_bad_poles",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 3}),
            children=(),
            metadata={"poles": (1,)},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2

    def test_transverse_s_node_split_block_root_bubble_adoption(self) -> None:
        """Articulations that bubble past intermediate containers are adopted at block root."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        s_child = SolveTreeNode(
            node_id="s_deep",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        container1 = SolveTreeNode(
            node_id="cont1",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(s_child,),
        )
        container2 = SolveTreeNode(
            node_id="cont2",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(container1,),
        )
        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(container2,),
        )
        tree = SolveTree(root=world_root, components=(container2,), graph=graph)
        res = TransverseSNodeSplitPass(graph).apply(tree)
        # Block root container adopts the bubbles rather than allowing them to escape to world root
        assert len(res.root.children) == 1
        block_root = res.root.children[0]
        assert len(block_root.children) >= 2
        child_ids = {c.node_id for c in block_root.children}
        assert "s_deep_a" in child_ids
        assert "s_deep_b" in child_ids

    def test_transverse_s_node_split_world_root_direct_adoption(self) -> None:
        """When an S-node is a direct component of world root, split branches are adopted at world root."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        s_child = SolveTreeNode(
            node_id="s_comp",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(s_child,),
        )
        tree = SolveTree(root=world_root, components=(s_child,), graph=graph)
        res = TransverseSNodeSplitPass(graph).apply(tree)
        assert len(res.root.children) == 2
        assert {c.node_id for c in res.root.children} == {"s_comp_a", "s_comp_b"}

    def test_transverse_s_node_split_cut_edges_and_contract_poles(self) -> None:
        """Cut edges and invalid contract poles are handled gracefully."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 20, Direction.north),
                (2, 20, Direction.north),
            ],
            cuts=[(1, 10)],  # Cut edge from p0
        )
        s_cut = SolveTreeNode(
            node_id="s_cut",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        s_bad_contract = SolveTreeNode(
            node_id="s_bad_contract",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"cut_contract": SplitPairCut(poles=(1,))},  # Bad length
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(s_cut, s_bad_contract),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2

    def test_transverse_s_node_split_boundary_ports_severed_edges(self) -> None:
        """TransverseSNodeSplitPass exports severed corridor boundary ports on node_a and node_b excluding local cluster rooms."""
        graph = _make_graph(
            rooms=[1, 2, 10, 20, 100, 200],
            edges=[
                (1, 10, Direction.north),
                (2, 20, Direction.north),
                (10, 20, Direction.east),
                (10, 100, Direction.east),
                (100, 200, Direction.east),
                (200, 20, Direction.east),
            ],
        )
        child_a = SolveTreeNode(
            node_id="child_a",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({100}),
            children=(),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=10)},
        )
        child_b = SolveTreeNode(
            node_id="child_b",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({200}),
            children=(),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=20)},
        )
        s_node = SolveTreeNode(
            node_id="s_node",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 20}),
            children=(child_a, child_b),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(s_node,),
        )
        res = TransverseSNodeSplitPass(graph).transform_node(parent)
        assert len(res.children) == 2
        node_a = [c for c in res.children if c.node_id == "s_node_a"][0]
        node_b = [c for c in res.children if c.node_id == "s_node_b"][0]

        assert node_a.rooms == frozenset({10})
        assert node_b.rooms == frozenset({20})

        # Boundary ports must include 100 on node_a and 200 on node_b, but NOT local rooms 10 or 20
        assert node_a.metadata.get("boundary_ports") == (100,)
        assert node_b.metadata.get("boundary_ports") == (200,)
        assert node_a.metadata.get("severed_bridge_edges") == ((10, 20), (100, 200))
        assert node_b.metadata.get("severed_bridge_edges") == ((10, 20), (100, 200))
        assert res.metadata.get("severed_bridge_edges") == ((10, 20), (100, 200))

    def test_p_node_block_root_adopts_split_branches_preserving_elasticity(self) -> None:
        """When a P-node is the block root, severed S-node branches are adopted directly at the root level."""
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20, 30],
            edges=[
                # Direct edge between poles
                (1, 2, Direction.north),
                # Elastic S-node corridor branch across poles {1, 2}
                (1, 10, Direction.west),
                (10, 11, Direction.north),
                (11, 20, Direction.north),
                (2, 20, Direction.west),
                # Second series loop branch across poles {1, 2}
                (1, 30, Direction.east),
                (30, 2, Direction.north),
            ],
        )
        s_elastic = SolveTreeNode(
            node_id="s_elastic",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR, "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        s_other = SolveTreeNode(
            node_id="s_other",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({30}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR, "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        p_root = SolveTreeNode(
            node_id="p_block_root",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(s_elastic, s_other),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR, "cut_contract": SplitPairCut(poles=(1, 2))},
        )

        res = TransverseSNodeSplitPass(graph).transform_node(p_root)
        # P-node block root retains its split-pair identity and adopts the severed sibling clusters
        assert res.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert res.rooms == frozenset({1, 2})
        assert len(res.children) == 3
        child_ids = {c.node_id for c in res.children}
        assert "s_other" in child_ids
        assert "s_elastic_a" in child_ids
        assert "s_elastic_b" in child_ids
        assert res.metadata.get("severed_bridge_edges") == ((10, 11),)

    def test_s_node_block_root_adopts_split_branches_from_inner_container(self) -> None:
        """When an S-node is the block root, bubbled split branches from child containers are adopted at block root."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4, 10, 11, 20],
            edges=[
                # Outer cycle: 1 -> 2 -> 3 -> 4 -> 1
                (1, 2, Direction.north),
                (2, 3, Direction.east),
                (3, 4, Direction.south),
                (4, 1, Direction.west),
                # Elastic S-node corridor across poles {1, 2}
                (1, 10, Direction.west),
                (10, 11, Direction.north),
                (11, 20, Direction.north),
                (2, 20, Direction.west),
            ],
        )
        s_elastic = SolveTreeNode(
            node_id="s_elastic",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR, "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        inner_container = SolveTreeNode(
            node_id="inner_container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(s_elastic,),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        s_root = SolveTreeNode(
            node_id="s_block_root",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 3, 4}),
            children=(inner_container,),
        )

        res = TransverseSNodeSplitPass(graph).transform_node(s_root)
        # S-node block root adopts the severed clusters directly
        assert res.node_type == TreeNodeType.SERIES_LOOP
        assert res.rooms == frozenset({1, 2, 3, 4})
        child_ids = {c.node_id for c in res.children}
        assert "s_elastic_a" in child_ids
        assert "s_elastic_b" in child_ids
        assert res.metadata.get("severed_bridge_edges") == ((10, 11),)


class TestSplitPairAllowlistFusionPass:
    """Unit tests for SplitPairAllowlistFusionPass safe 2-cut allowlist evaluation and branch fusion."""

    def test_empty_tree_unchanged(self) -> None:
        """Empty solve tree is returned unchanged."""
        empty_root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.EMPTY,
            rooms=frozenset(),
        )
        tree = SolveTree(root=empty_root, components=())
        pass_instance = SplitPairAllowlistFusionPass()
        assert pass_instance.apply(tree) is tree

    def test_none_graph_returns_unchanged(self) -> None:
        """When no graph is provided, node and tree return unchanged."""
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(),
        )
        pass_instance = SplitPairAllowlistFusionPass(graph=None)
        assert pass_instance.transform_node(node) is node

        tree = SolveTree(root=node, components=(node,), graph=None)
        assert pass_instance.apply(tree) is tree

    def test_non_split_pair_node_unchanged(self) -> None:
        """Rigid clusters and series loops are returned unchanged without evaluation."""
        graph = _make_graph([1, 2], [(1, 2, Direction.north)])
        node = SolveTreeNode(
            node_id="rc",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(),
        )
        pass_instance = SplitPairAllowlistFusionPass(graph)
        res = pass_instance.transform_node(node)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER
        assert res.rooms == frozenset({1, 2})

    def test_world_root_recursion(self) -> None:
        """World root component recursively transforms and processes child components."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.west),
                (4, 2, Direction.east),
            ],
        )
        branch_a = SolveTreeNode(
            node_id="b_a",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        branch_b = SolveTreeNode(
            node_id="b_b",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(branch_a, branch_b),
            metadata={"poles": (1, 2), "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        world_root = SolveTreeNode(
            node_id="world",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(container,),
        )
        tree = SolveTree(root=world_root, components=(container,), graph=graph)
        transformed = SplitPairAllowlistFusionPass(graph).apply(tree)
        assert transformed.root.node_type == TreeNodeType.WORLD_ROOT
        assert len(transformed.root.children) == 1
        assert transformed.root.children[0].node_type == TreeNodeType.SPLIT_PAIR_CONTAINER

    def test_safe_branches_remain_split_pair_container(self) -> None:
        """Valid elastic series loop branches in disjoint half-spaces satisfy contract and remain split pair."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.west),
                (4, 2, Direction.east),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
            metadata={"poles": (1, 2), "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER
        assert res.rooms == frozenset({1, 2})
        assert len(res.children) == 2

    def test_unsafe_rigid_branch_fuses(self) -> None:
        """Candidate branch that is not a series loop (e.g. rigid cluster) fuses container into rigid cluster."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4, 5],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 4, Direction.north),
                (4, 2, Direction.west),
                (1, 5, Direction.west),
                (5, 2, Direction.east),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({3, 4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({5}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
            metadata={"poles": (1, 2), "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER
        assert res.rooms == frozenset({1, 2, 3, 4, 5})
        assert res.children == ()
        assert res.metadata.get("poles") == (1, 2)

    def test_unsafe_nested_branch_children_fuses(self) -> None:
        """Candidate series loop branch containing nested SPQR children fails elasticity and fuses."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4, 5],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.west),
                (4, 2, Direction.east),
            ],
        )
        nested = SolveTreeNode(
            node_id="nested_spqr",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({5}),
            children=(),
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(nested,),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
            metadata={"poles": (1, 2), "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER
        assert res.rooms == frozenset({1, 2, 3, 4, 5})

    def test_unsafe_empty_internal_rooms_fuses(self) -> None:
        """Candidate branch with no internal rooms fails contract and fuses."""
        graph = _make_graph(
            rooms=[1, 2],
            edges=[(1, 2, Direction.north)],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2}),  # internal rooms = rooms - {1, 2} is empty
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1,),
            metadata={"poles": (1, 2)},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER

    def test_unsafe_multi_exit_ports_fuses(self) -> None:
        """Branch connecting to pole with multiple exit directions fails port contract and fuses."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 2, Direction.north),
                # Branch 1 connects to pole 1 via both East (1->3) and North (1->4)
                (1, 3, Direction.east),
                (1, 4, Direction.north),
                (3, 4, Direction.north),
                (4, 2, Direction.west),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3, 4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1,),
            metadata={"poles": (1, 2)},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER

    def test_unsafe_disconnected_path_fuses(self) -> None:
        """Branch without simple path reaching pole 2 fails path tracing and fuses."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                # 3 and 4 are disconnected; 4 connects to 2
                (4, 2, Direction.west),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3, 4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1,),
            metadata={"poles": (1, 2)},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER

    def test_unsafe_transverse_bracketing_fuses(self) -> None:
        """Branch expanding into both positive and negative transverse half-spaces fails contract and fuses."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4, 5, 6, 7, 8],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                # Outer loop expanding both east (+X) and west (-X)
                (1, 4, Direction.south),
                (4, 5, Direction.east),
                (5, 6, Direction.north),
                (6, 7, Direction.west),
                (7, 8, Direction.south),
                (8, 2, Direction.east),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4, 5, 6, 7, 8}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
            metadata={"poles": (1, 2)},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER
        assert res.rooms == frozenset({1, 2, 3, 4, 5, 6, 7, 8})

    def test_unsafe_overlapping_halfspaces_fuses(self) -> None:
        """Two candidate branches expanding into the same half-space fail disjointness and fuse."""
        graph = _make_graph(
            rooms=[1, 2, 3, 30, 4, 5, 6, 7],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 30, Direction.north),
                (30, 2, Direction.west),
                (1, 4, Direction.south),
                (4, 5, Direction.east),
                (5, 6, Direction.north),
                (6, 7, Direction.west),
                (7, 2, Direction.south),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3, 30}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4, 5, 6, 7}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
            metadata={"poles": (1, 2)},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER
        assert res.rooms == frozenset({1, 2, 3, 30, 4, 5, 6, 7})

    def test_child_container_preserves_parent_poles_and_contracts(self) -> None:
        """Child container attached to parent poles preserves cut contract and excludes parent poles on fusion."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.east),  # same half-space -> fails allowlist
                (4, 2, Direction.west),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        # Child container has rooms = frozenset() because parent owns poles (1, 2)
        child_container = SolveTreeNode(
            node_id="child_sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(b1, b2),
            metadata={
                "cut_type": CutType.SPLIT_PAIR,
                "poles": (1, 2),
                "cut_contract": SplitPairCut(poles=(1, 2)),
            },
        )
        parent = SolveTreeNode(
            node_id="parent_cluster",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child_container,),
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(parent)
        assert len(res.children) == 1
        fused_child = res.children[0]
        assert fused_child.node_type == TreeNodeType.RIGID_CLUSTER
        # Internal rooms only (parent owns poles 1 and 2)
        assert fused_child.rooms == frozenset({3, 4})
        assert fused_child.metadata.get("cut_type") == CutType.SPLIT_PAIR
        assert fused_child.metadata.get("poles") == (1, 2)
        assert isinstance(fused_child.cut_contract, SplitPairCut)

    def test_container_with_articulation_children_preserved(self) -> None:
        """When unsafe split pair container fuses, articulation children are preserved on the fused node."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4, 10, 11],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.east),  # same half-space -> fails allowlist
                (4, 2, Direction.west),
                # Articulation child on room 1
                (1, 10, Direction.south),
                (10, 11, Direction.east),
                (11, 1, Direction.north),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
            metadata={"poles": (1, 2)},
        )
        art_child = SolveTreeNode(
            node_id="art_block",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 11}),
            children=(),
            metadata={"cut_type": CutType.ARTICULATION_POINT, "anchor_vnum": 1},
        )
        container = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2, art_child),
            metadata={"poles": (1, 2), "cut_contract": SplitPairCut(poles=(1, 2))},
        )
        res = SplitPairAllowlistFusionPass(graph).transform_node(container)
        assert res.node_type == TreeNodeType.RIGID_CLUSTER
        assert res.rooms == frozenset({1, 2, 3, 4})
        assert len(res.children) == 1
        assert res.children[0].node_id == "art_block"

    def test_pipeline_integration_order(self) -> None:
        """Default hierarchical pipeline executes decomposition passes then fusion then downstream passes."""
        pipeline = PipelineTreeBuilder.default_pipeline()
        passes = pipeline.passes
        assert len(passes) == 5
        assert isinstance(passes[0], BlockCutDecompositionPass)
        assert isinstance(passes[1], SpqrDecompositionPass)
        assert isinstance(passes[2], SplitPairAllowlistFusionPass)
        assert isinstance(passes[3], ArticulationChildAttachmentPass)
        assert isinstance(passes[4], TransverseSNodeSplitPass)

    def test_allowlist_poles_not_len_2_returns_false(self) -> None:
        """Poles tuple with length != 2 fails allowlist contract."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1}),
            children=(),
        )
        assert not _is_safe_2cut_allowlist(node, (1,), ())  # type: ignore[arg-type]

    def test_allowlist_no_candidate_branches_returns_false(self) -> None:
        """Split-pair container with no candidate branches fails allowlist contract."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(),
        )
        assert not _is_safe_2cut_allowlist(node, (1, 2), ())

    def test_allowlist_cut_edges_ignored(self) -> None:
        """Cut edges are ignored during exit port inspection and path tracing."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=2, direction=Direction.west),
            LayoutEdge(src=1, dst=3, direction=Direction.south, cut=True),
        ]
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
        )
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1,),
        )
        assert _is_safe_2cut_allowlist(node, (1, 2), edges)

    def test_allowlist_incomplete_path_visit(self) -> None:
        """Branch containing disconnected unvisited room fails path tracing."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=2, direction=Direction.west),
        ]
        # Room 99 is disconnected from the path 1 -> 3 -> 2
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3, 99}),
            children=(),
        )
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1,),
        )
        assert not _is_safe_2cut_allowlist(node, (1, 2), edges)

    def test_allowlist_opposing_u_v_steps(self) -> None:
        """Branch with opposing transverse exit steps fails contract."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        # u step is East (+X), v step is East into v (meaning step from last room to v is West, so v_step is -X)
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=4, direction=Direction.north),
            LayoutEdge(src=4, dst=2, direction=Direction.east),
        ]
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3, 4}),
            children=(),
        )
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1,),
        )
        assert not _is_safe_2cut_allowlist(node, (1, 2), edges)

    def test_allowlist_multiple_zero_signatures(self) -> None:
        """Multiple branches aligned along inter-pole axis with zero transverse signature fail contract."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        # Two branches along North-South axis without transverse expansion
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=3, direction=Direction.north),
            LayoutEdge(src=3, dst=2, direction=Direction.north),
            LayoutEdge(src=1, dst=4, direction=Direction.north),
            LayoutEdge(src=4, dst=2, direction=Direction.north),
        ]
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
        )
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
        )
        assert not _is_safe_2cut_allowlist(node, (1, 2), edges)

    def test_allowlist_conflicting_candidate_axes_without_direct_edge(self) -> None:
        """Conflicting inferred inter-pole axes without a direct edge fail contract."""
        from romutil.layout.hierarchy.transforms import _is_safe_2cut_allowlist

        # No direct edge between 1 and 2
        # Branch 1 suggests X axis as primary (1 -> 3 east, 3 -> 2 east)
        # Branch 2 suggests Y axis as primary (1 -> 4 north, 4 -> 2 north)
        edges = [
            LayoutEdge(src=1, dst=3, direction=Direction.east),
            LayoutEdge(src=3, dst=2, direction=Direction.east),
            LayoutEdge(src=1, dst=4, direction=Direction.north),
            LayoutEdge(src=4, dst=2, direction=Direction.north),
        ]
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
        )
        node = SolveTreeNode(
            node_id="sp",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
        )
        assert not _is_safe_2cut_allowlist(node, (1, 2), edges)

    def test_poles_fallback_extraction_from_metadata_list_and_rooms(self) -> None:
        """Poles are extracted from metadata list or node.rooms when cut_contract is absent."""
        graph = _make_graph(
            rooms=[1, 2, 3, 4],
            edges=[
                (1, 2, Direction.north),
                (1, 3, Direction.east),
                (3, 2, Direction.west),
                (1, 4, Direction.west),
                (4, 2, Direction.east),
            ],
        )
        b1 = SolveTreeNode(
            node_id="b1",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({3}),
            children=(),
        )
        b2 = SolveTreeNode(
            node_id="b2",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({4}),
            children=(),
        )
        # Case 1: metadata["poles"] is a list
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(b1, b2),
            metadata={"poles": [1, 2]},
        )
        pass_instance = SplitPairAllowlistFusionPass(graph)
        res1 = pass_instance.transform_node(c1)
        assert res1.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER

        # Case 2: no metadata poles, but len(rooms) == 2
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(b1, b2),
        )
        res2 = pass_instance.transform_node(c2)
        assert res2.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER


class TestUnifiedHierarchicalPipeline:
    """Verify SolveTree.seed, decomposition passes, and unified pipeline tree construction."""

    def test_solve_tree_seed_empty(self) -> None:
        graph = LayoutGraph(rooms={}, edges=())
        tree = SolveTree.seed(graph)
        assert tree.root.node_id == "root"
        assert tree.root.node_type == TreeNodeType.EMPTY
        assert tree.root.rooms == frozenset()
        assert tree.components == ()
        assert tree.graph is graph

    def test_solve_tree_seed_with_rooms(self) -> None:
        rooms = {1: _make_room(1), 2: _make_room(2), 3: _make_room(3)}
        graph = LayoutGraph(rooms=rooms, edges=())
        tree = SolveTree.seed(graph)
        assert tree.root.node_id == "seed_root"
        assert tree.root.node_type == TreeNodeType.BLOCK
        assert tree.root.rooms == frozenset({1, 2, 3})
        assert tree.components == (tree.root,)
        assert tree.graph is graph

    def test_block_cut_decomposition_pass_empty(self) -> None:
        graph = LayoutGraph(rooms={}, edges=())
        seed = SolveTree.seed(graph)
        decomp = BlockCutDecompositionPass()
        result = decomp.apply(seed)
        assert result.root.node_id == "empty"
        assert result.root.node_type == TreeNodeType.EMPTY
        assert result.root.rooms == frozenset()
        assert result.components == ()

    def test_block_cut_decomposition_pass_single_room(self) -> None:
        graph = LayoutGraph(rooms={42: _make_room(42)}, edges=())
        seed = SolveTree.seed(graph)
        result = BlockCutDecompositionPass().apply(seed)
        assert len(result.components) == 1
        assert result.root == result.components[0]
        assert result.root.node_id == "comp_0_block_0"
        assert result.root.node_type == TreeNodeType.BICONNECTED_BLOCK
        assert result.root.rooms == frozenset({42})
        assert result.root.children == ()

    def test_block_cut_decomposition_pass_disconnected_components(self) -> None:
        rooms = {10: _make_room(10), 20: _make_room(20)}
        graph = LayoutGraph(rooms=rooms, edges=())
        seed = SolveTree.seed(graph)
        result = BlockCutDecompositionPass().apply(seed)
        assert result.root.node_id == "world_root"
        assert result.root.node_type == TreeNodeType.WORLD_ROOT
        assert len(result.components) == 2
        assert result.components == result.root.children
        c0, c1 = result.components
        assert c0.node_id == "comp_0_block_0"
        assert c0.rooms == frozenset({10})
        assert c1.node_id == "comp_1_block_0"
        assert c1.rooms == frozenset({20})

    def test_block_cut_decomposition_pass_cut_vertex(self) -> None:
        # Two triangles 1-2-3 and 3-4-5 meeting at articulation point 3
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4, 5)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=1, direction=Direction.south, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.east, cut=False),
            LayoutEdge(src=4, dst=5, direction=Direction.north, cut=False),
            LayoutEdge(src=5, dst=3, direction=Direction.west, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        seed = SolveTree.seed(graph)
        pass_instance = BlockCutDecompositionPass()
        assert pass_instance.graph is None
        result = pass_instance.apply(seed)
        assert len(result.components) == 1
        root = result.root
        assert root.node_type == TreeNodeType.BLOCK
        assert len(root.children) == 1
        child = root.children[0]
        assert child.node_type == TreeNodeType.BLOCK
        assert child.metadata.get("cut_type") == CutType.ARTICULATION_POINT
        assert child.metadata.get("anchor_vnum") == 3
        assert isinstance(child.cut_contract, ArticulationPointCut)
        assert child.cut_contract.anchor_vnum == 3

    def test_spqr_decomposition_pass_empty(self) -> None:
        graph = LayoutGraph(rooms={}, edges=())
        seed = SolveTree.seed(graph)
        pass_instance = SpqrDecompositionPass(graph=graph)
        assert pass_instance.graph is graph
        result = pass_instance.apply(seed)
        assert result.root.node_type == TreeNodeType.EMPTY

    def test_spqr_decomposition_pass_small_block(self) -> None:
        # Triangle 1-2-3 has 3 rooms (< 4), remains untouched
        rooms = {v: _make_room(v) for v in (1, 2, 3)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=1, direction=Direction.south, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        seed = SolveTree.seed(graph)
        bct_tree = BlockCutDecompositionPass().apply(seed)
        spqr_tree = SpqrDecompositionPass().apply(bct_tree)
        assert spqr_tree.root.node_type == TreeNodeType.BICONNECTED_BLOCK
        assert spqr_tree.root.rooms == frozenset({1, 2, 3})

    def test_spqr_decomposition_pass_4cycle(self) -> None:
        # 4-cycle 1-2-3-4 decomposes into SERIES_LOOP
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        seed = SolveTree.seed(graph)
        bct_tree = BlockCutDecompositionPass().apply(seed)
        spqr_tree = SpqrDecompositionPass().apply(bct_tree)
        assert spqr_tree.root.node_type == TreeNodeType.SERIES_LOOP

    def test_spqr_decomposition_pass_k4(self) -> None:
        # K4 complete graph decomposes into RIGID_CLUSTER
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
            LayoutEdge(src=1, dst=3, direction=Direction.northeast, cut=False),
            LayoutEdge(src=2, dst=4, direction=Direction.southeast, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        seed = SolveTree.seed(graph)
        bct_tree = BlockCutDecompositionPass().apply(seed)
        spqr_tree = SpqrDecompositionPass().apply(bct_tree)
        assert spqr_tree.root.node_type == TreeNodeType.RIGID_CLUSTER

    def test_spqr_decomposition_pass_world_root(self) -> None:
        # Two independent 4-cycles under a WORLD_ROOT
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4, 10, 20, 30, 40)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
            LayoutEdge(src=10, dst=20, direction=Direction.north, cut=False),
            LayoutEdge(src=20, dst=30, direction=Direction.east, cut=False),
            LayoutEdge(src=30, dst=40, direction=Direction.south, cut=False),
            LayoutEdge(src=40, dst=10, direction=Direction.west, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        seed = SolveTree.seed(graph)
        bct_tree = BlockCutDecompositionPass().apply(seed)
        assert bct_tree.root.node_type == TreeNodeType.WORLD_ROOT
        spqr_tree = SpqrDecompositionPass().apply(bct_tree)
        assert spqr_tree.root.node_type == TreeNodeType.WORLD_ROOT
        assert len(spqr_tree.root.children) == 2
        for child in spqr_tree.root.children:
            assert child.node_type == TreeNodeType.SERIES_LOOP

    def test_pipeline_tree_builder_custom_pipeline(self) -> None:
        class DummyPass:
            def apply(self, tree: SolveTree) -> SolveTree:
                return tree

        dummy = DummyPass()
        custom_pipe = TreeTransformationPipeline([dummy])
        builder = PipelineTreeBuilder(pipeline=custom_pipe)
        assert builder.pipeline is custom_pipe

    def test_pipeline_tree_builder_custom_block_cut_injection(self) -> None:
        custom_bct = BlockCutTreeBuilder()
        builder = PipelineTreeBuilder(block_cut_builder=custom_bct)
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        simplified = SimplifiedGraph(graph=graph)
        tree = builder.build_tree(simplified)
        assert tree.root.node_type == TreeNodeType.SERIES_LOOP

        # Builder without _pass falls back to default BlockCutDecompositionPass
        class DummyBuilder:
            pass

        fallback_builder = PipelineTreeBuilder(block_cut_builder=DummyBuilder())  # type: ignore[arg-type]
        assert isinstance(fallback_builder.pipeline.passes[0], BlockCutDecompositionPass)

    def test_module_getattr_exports(self) -> None:
        import romutil.layout.hierarchy.spqr as spqr_mod
        import romutil.layout.hierarchy.transforms as transforms_mod

        assert spqr_mod.SpqrTreeBuilder is PipelineTreeBuilder
        with pytest.raises(AttributeError):
            getattr(spqr_mod, "non_existent_attribute")

        assert transforms_mod.BlockCutDecompositionPass is BlockCutDecompositionPass
        assert transforms_mod.SpqrDecompositionPass is SpqrDecompositionPass
        with pytest.raises(AttributeError):
            getattr(transforms_mod, "non_existent_attribute")

    @pytest.mark.parametrize("fixture_name", ["school", "thalos", "arachnos", "midgaard"])
    def test_pipeline_tree_builder_fixtures(self, fixture_name: str) -> None:
        from romutil.layout.hierarchy.transforms import _collect_subtree_rooms
        from romutil.parser import parse_file
        from romutil.types import WorldGraph

        area = parse_file(f"tests/fixtures/areas/{fixture_name}.are")
        world_graph = WorldGraph.from_area_data(area)
        layout_graph = LayoutGraph.from_world_graph(world_graph)
        simplified = SimplifiedGraph(graph=layout_graph)

        builder = PipelineTreeBuilder()
        tree = builder.build_tree(simplified)

        assert isinstance(tree, SolveTree)
        assert tree.root.node_type in (
            TreeNodeType.WORLD_ROOT,
            TreeNodeType.RIGID_CLUSTER,
            TreeNodeType.SERIES_LOOP,
            TreeNodeType.BLOCK,
            TreeNodeType.BICONNECTED_BLOCK,
        )

        all_tree_rooms = _collect_subtree_rooms(tree.root)
        expected_rooms = set(simplified.graph.rooms.keys())
        assert all_tree_rooms == expected_rooms

        # SpqrTreeBuilder alias produces identical tree structure
        legacy_builder = SpqrTreeBuilder()
        legacy_tree = legacy_builder.build_tree(simplified)
        assert _collect_subtree_rooms(legacy_tree.root) == expected_rooms
        assert legacy_tree.root.node_type == tree.root.node_type
        assert len(legacy_tree.components) == len(tree.components)
