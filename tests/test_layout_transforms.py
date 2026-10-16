"""Unit tests for solve tree transformation passes and composition pipelines."""

from __future__ import annotations

import pytest

from romutil.layout.hierarchy import (
    AnchorStrippingPass,
    ArticulationChildAttachmentPass,
    BlockCutTreeBuilder,
    SpqrTreeBuilder,
    TransverseSNodeSplitPass,
    TreeTransformationPass,
    TreeTransformationPipeline,
    attach_articulation_children,
    strip_anchor,
)
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


def _make_graph(rooms: list[int], edges: list[tuple[int, int, Direction]]) -> LayoutGraph:
    """Helper to construct a LayoutGraph."""
    return LayoutGraph(
        rooms={v: _make_room(v) for v in rooms},
        edges=tuple(
            LayoutEdge(src=u, dst=v, direction=d, cut=False)
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
        assert {c_a.metadata.get("anchor_vnum"), c_b.metadata.get("anchor_vnum")} == {1, 2}
        assert c_a.rooms | c_b.rooms == frozenset({1, 2, 10, 11, 20})

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
        pipeline = SpqrTreeBuilder.default_pipeline()
        assert len(pipeline.passes) == 2
        assert isinstance(pipeline.passes[0], ArticulationChildAttachmentPass)
        assert isinstance(pipeline.passes[1], TransverseSNodeSplitPass)


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
        assert _get_child_poles_or_anchor(bridge) == {1, 2}

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
        c_1 = [c for c in res.children if c.metadata.get("anchor_vnum") == 1][0]
        assert len(c_1.children) == 1
        assert c_1.children[0].node_id == "nested_orphan"
