"""Unit tests for solve tree transformation passes and composition pipelines."""

from __future__ import annotations

import pytest

from romutil.layout.hierarchy import (
    AnchorStrippingPass,
    BlockCutDecompositionPass,
    BlockCutTreeBuilder,
    OrthogonalBridgePromotionPass,
    PipelineTreeBuilder,
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


class TestSpqrDirectArticulationAttachment:
    """Verify direct articulation child attachment and anchor stripping in SPQR decomposition."""

    def test_articulation_child_attached_to_spqr_container_internal_branch(self) -> None:
        """1-cut child is directly attached under the SPQR descendant branch owning its anchor."""
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4, 10, 11)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
            LayoutEdge(src=3, dst=10, direction=Direction.north, cut=False),
            LayoutEdge(src=10, dst=11, direction=Direction.east, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))

        art_child = SolveTreeNode(
            node_id="art_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 3,
                "cut_contract": ArticulationPointCut(anchor_vnum=3),
            },
        )
        block_node = SolveTreeNode(
            node_id="parent_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2, 3, 4}),
            children=(art_child,),
        )
        tree = SolveTree(root=block_node, components=(block_node,), graph=graph)

        pass_instance = SpqrDecompositionPass(graph=graph)
        transformed = pass_instance.apply(tree)

        assert transformed.root.node_type == TreeNodeType.SERIES_LOOP
        assert any(c.node_id == "art_block" for c in transformed.root.children)

    def test_articulation_child_attached_to_internal_split_pair_branch(self) -> None:
        """When an SPQR tree produces a split-pair container, child attaching at internal branch attaches to that branch."""
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
        parent = SolveTreeNode(
            node_id="container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2}),
            children=(s_node_1, s_node_2),
        )

        attached = attach_articulation_children(parent, [art_child])
        assert len(attached.children) == 2
        s1_transformed = [c for c in attached.children if c.node_id == "s_1"][0]
        assert len(s1_transformed.children) == 1
        assert s1_transformed.children[0].node_id == "art_block"

    def test_articulation_child_attached_to_pole_remains_direct_child(self) -> None:
        """When anchor is in the split-pair container's rooms (the poles), child remains a direct child of the container."""
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
            children=(s_node,),
        )

        attached = attach_articulation_children(parent, [art_child])
        assert len(attached.children) == 2
        assert any(c.node_id == "art_block" for c in attached.children)

    def test_anchor_stripping_on_decomposed_spqr_block(self) -> None:
        """Parent gateway anchor is stripped from the decomposed SPQR subtree to enforce disjoint room ownership."""
        rooms = {v: _make_room(v) for v in (10, 1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=10, dst=1, direction=Direction.north, cut=False),
            LayoutEdge(src=1, dst=2, direction=Direction.east, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.south, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.west, cut=False),
            LayoutEdge(src=4, dst=10, direction=Direction.north, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))

        block_node = SolveTreeNode(
            node_id="child_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2, 3, 4}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 10,
                "cut_contract": ArticulationPointCut(anchor_vnum=10),
            },
        )
        tree = SolveTree(root=block_node, components=(block_node,), graph=graph)

        transformed = SpqrDecompositionPass(graph=graph).apply(tree)
        assert 10 not in transformed.root.rooms
        assert transformed.root.metadata["anchor_vnum"] == 10
        assert isinstance(transformed.root.cut_contract, ArticulationPointCut)
        assert transformed.root.cut_contract.anchor_vnum == 10

    def test_spqr_decomposition_preserves_other_non_articulation_children(self) -> None:
        """Non-1-cut child nodes are retained at the SPQR root level."""
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.south, cut=False),
            LayoutEdge(src=4, dst=1, direction=Direction.west, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))

        other_child = SolveTreeNode(
            node_id="custom_child",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({100}),
            children=(),
        )
        block_node = SolveTreeNode(
            node_id="block_parent",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2, 3, 4}),
            children=(other_child,),
        )
        tree = SolveTree(root=block_node, components=(block_node,), graph=graph)

        transformed = SpqrDecompositionPass(graph=graph).apply(tree)
        assert any(c.node_id == "custom_child" for c in transformed.root.children)

    def test_small_block_anchor_stripping_disjoint_ownership(self) -> None:
        """When a small block (< 4 rooms) contains its gateway anchor, the anchor is stripped."""
        # 3-room block with anchor 1 included in its rooms
        small_block = SolveTreeNode(
            node_id="small_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2, 3}),
            children=(),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 1,
                "cut_contract": ArticulationPointCut(anchor_vnum=1),
            },
        )
        tree = SolveTree(root=small_block, components=(small_block,))

        transformed = SpqrDecompositionPass().apply(tree)
        assert transformed.root.node_type == TreeNodeType.BLOCK
        assert 1 not in transformed.root.rooms
        assert transformed.root.rooms == frozenset({2, 3})


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

        pass_instance = TransverseSNodeSplitPass(graph)
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
            assert len(pipeline.passes) == 4
            assert isinstance(pipeline.passes[0], BlockCutDecompositionPass)
            assert isinstance(pipeline.passes[1], SpqrDecompositionPass)
            assert isinstance(pipeline.passes[2], TransverseSNodeSplitPass)
            assert isinstance(pipeline.passes[3], OrthogonalBridgePromotionPass)


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

    def test_internal_step_in_covered_direction_not_split(self) -> None:
        """Internal spine steps whose directions are covered by boundary exit directions are not severed."""
        # 1 -> 10 (West), 10 -> 20 (West), 20 -> 2 (West, so 2 -> 20 is East)
        # Forward boundary exit directions along the spine:
        # p0 -> 10: West.
        # 20 -> p1: West.
        # Covered directions: {West}.
        # Step (10, 20) is West, which is covered by {West}.
        # Therefore, s_covered remains intact as a single child.
        graph = _make_graph(
            rooms=[1, 2, 10, 20],
            edges=[
                (1, 10, Direction.west),
                (10, 20, Direction.west),
                (20, 2, Direction.west),
            ],
        )
        child = SolveTreeNode(
            node_id="s_covered",
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
        assert res.children[0].node_id == "s_covered"

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

    def test_complete_bidirectional_transverse_reachability(self) -> None:
        """S-node spanning both transverse axes severs at most once per orthogonal axis."""
        # Poles 1 and 2 with North entrance corridors (Y axis).
        # Transverse steps cover East (+X), West (-X), Up (+Z), and Down (-Z).
        # Under orthogonal axis expansion joints, X is cut at most once and Z is cut at most once,
        # yielding at most 2 splits and 3 sub-nodes total.
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 13, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 12, Direction.west),
                (12, 13, Direction.up),
                (13, 20, Direction.down),
                (2, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_corridor_3d",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 13, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # 2 orthogonal transverse axes (X and Z) severed into 3 sub-nodes:
        # Step (10, 11) severs X axis.
        # Step (11, 12) is on X axis (already covered) -> not severed.
        # Step (12, 13) severs Z axis.
        # Step (13, 20) is on Z axis (already covered) -> not severed.
        assert len(transformed.root.children) == 3
        child_ids = [c.node_id for c in transformed.root.children]
        assert child_ids == [
            "s_corridor_3d_a",
            "s_corridor_3d_b",
            "s_corridor_3d_c",
        ]
        # Severed bridge edges recorded in parent container metadata
        expected_severed = ((10, 11), (12, 13))
        assert transformed.root.metadata.get("severed_bridge_edges") == expected_severed

        # End sub-nodes anchored to boundary poles
        c_a = transformed.root.children[0]
        c_c = transformed.root.children[2]
        assert isinstance(c_a.cut_contract, BridgeCut)
        assert c_a.cut_contract.bridge_edge == (1, 10)
        assert isinstance(c_c.cut_contract, BridgeCut)
        assert c_c.cut_contract.bridge_edge == (2, 20)

    def test_snaking_corridor_cohesion_limits_to_one_cut_per_direction(self) -> None:
        """Snaking corridors with repeated turns in the same direction are severed at most once per direction."""
        # Entrance corridors are North. The spine alternates East and North repeatedly:
        # 1 -> 10 (North), 10 -> 11 (East), 11 -> 12 (North), 12 -> 13 (East),
        # 13 -> 14 (North), 14 -> 15 (East), 15 -> 20 (North), 2 -> 20 (North).
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 13, 14, 15, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 12, Direction.north),
                (12, 13, Direction.east),
                (13, 14, Direction.north),
                (14, 15, Direction.east),
                (15, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_snake",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 13, 14, 15, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # Only the first East rung (10, 11) is severed; repeated East rungs are preserved
        assert len(transformed.root.children) == 2
        c_a = transformed.root.children[0]
        c_b = transformed.root.children[1]
        assert c_a.node_id == "s_snake_a"
        assert c_b.node_id == "s_snake_b"
        assert c_a.rooms == frozenset({10})
        assert c_b.rooms == frozenset({11, 12, 13, 14, 15, 20})
        assert transformed.root.metadata.get("severed_bridge_edges") == ((10, 11),)

    def test_three_dimensional_vertical_step_decoupling(self) -> None:
        """S-node with orthogonal horizontal and vertical steps decouples into 3D degrees of freedom."""
        # 1 -> 10 (North), 10 -> 11 (East), 11 -> 12 (North), 12 -> 13 (Down), 13 -> 20 (North), 2 -> 20 (North).
        # Entrance is North (Y). Transverse axes are X (East) and Z (Down).
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 13, 20],
            edges=[
                (1, 10, Direction.north),
                (10, 11, Direction.east),
                (11, 12, Direction.north),
                (12, 13, Direction.down),
                (13, 20, Direction.north),
                (2, 20, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_vertical",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 13, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # Decomposes into 3 sub-nodes:
        # a: {10}
        # b: {11, 12} (severed at East (10, 11))
        # c: {13, 20} (severed at Down (12, 13))
        assert len(transformed.root.children) == 3
        c_a, c_b, c_c = transformed.root.children
        assert c_a.node_id == "s_vertical_a"
        assert c_b.node_id == "s_vertical_b"
        assert c_c.node_id == "s_vertical_c"
        assert c_a.rooms == frozenset({10})
        assert c_b.rooms == frozenset({11, 12})
        assert c_c.rooms == frozenset({13, 20})
        assert transformed.root.metadata.get("severed_bridge_edges") == ((10, 11), (12, 13))

    def test_longitudinal_steps_not_severed_when_boundary_exits_cover_axis(self) -> None:
        """Boundary exits covering Y axis (South) cover all Y steps, severing only orthogonal transverse axes."""
        # Forward boundary exit directions along spine:
        # 1 -> 10 (South), 20 -> 2 (South).
        # Covered axes: {Y axis (1)}.
        # Step (10, 11) is North (Y axis) -> covered by boundary ports axis, not severed.
        # Step (11, 12) is East (X axis) -> orthogonal to boundary ports axis, severed!
        # Step (12, 20) is South (Y axis) -> covered axis, not severed.
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 20],
            edges=[
                (1, 10, Direction.south),
                (10, 11, Direction.north),
                (11, 12, Direction.east),
                (12, 20, Direction.south),
                (20, 2, Direction.south),
            ],
        )
        child = SolveTreeNode(
            node_id="s_opposing",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # Decomposes into 2 sub-nodes severed along X axis at (11, 12):
        # a: {10, 11} (anchored to 1 via South bridge)
        # b: {12, 20} (anchored to 2 via South bridge)
        assert len(transformed.root.children) == 2
        c_a, c_b = transformed.root.children
        assert c_a.node_id == "s_opposing_a"
        assert c_b.node_id == "s_opposing_b"
        assert c_a.rooms == frozenset({10, 11})
        assert c_b.rooms == frozenset({12, 20})
        assert transformed.root.metadata.get("severed_bridge_edges") == ((11, 12),)

    def test_snaking_corridor_opposing_cardinal_directions_same_axis_split_at_most_once(self) -> None:
        """Snaking corridor with opposing cardinal directions along the same orthogonal axis is cut at most once."""
        # Boundary exits along X axis (East).
        # Internal spine makes turns North (+Y), then East (+X), then South (-Y).
        # North and South are on the same orthogonal axis (Y axis, index 1).
        # The first turn (North) severs the Y axis. The subsequent turn (South) on Y axis must NOT be cut again.
        # Spine: 1 -> 10 (East), 10 -> 11 (North), 11 -> 12 (East), 12 -> 13 (South), 13 -> 20 (East), 20 -> 2 (East).
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 13, 20],
            edges=[
                (1, 10, Direction.east),
                (10, 11, Direction.north),
                (11, 12, Direction.east),
                (12, 13, Direction.south),
                (13, 20, Direction.east),
                (20, 2, Direction.east),
            ],
        )
        child = SolveTreeNode(
            node_id="s_opposing_cardinal",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 13, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # Y axis severed at most once at (10, 11), yielding 2 sub-nodes:
        # a: {10}
        # b: {11, 12, 13, 20}
        assert len(transformed.root.children) == 2
        c_a, c_b = transformed.root.children
        assert c_a.node_id == "s_opposing_cardinal_a"
        assert c_b.node_id == "s_opposing_cardinal_b"
        assert c_a.rooms == frozenset({10})
        assert c_b.rooms == frozenset({11, 12, 13, 20})
        assert transformed.root.metadata.get("severed_bridge_edges") == ((10, 11),)

    def test_perpendicular_boundary_exits_sever_orthogonal_axis(self) -> None:
        """Perpendicular boundary exits (X and Y axes) sever orthogonal steps along Z axis."""
        # Boundary exit 0 is along X (East).
        # Boundary exit 1 is along Y (North).
        # Covered axes = {X, Y}.
        # Step (10, 11) is along Z (Down). Orthogonal to both boundary exits -> severed!
        # Step (11, 20) is along Y (North). Covered by exit 1 -> not severed.
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 20],
            edges=[
                (1, 10, Direction.east),
                (10, 11, Direction.down),
                (11, 20, Direction.north),
                (20, 2, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_perp_3d",
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
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # Decomposes into 2 sub-nodes severed at Z step (10, 11):
        assert len(transformed.root.children) == 2
        c_a, c_b = transformed.root.children
        assert c_a.node_id == "s_perp_3d_a"
        assert c_b.node_id == "s_perp_3d_b"
        assert c_a.rooms == frozenset({10})
        assert c_b.rooms == frozenset({11, 20})
        assert transformed.root.metadata.get("severed_bridge_edges") == ((10, 11),)

    def test_perpendicular_boundary_exits_no_cut_when_spine_only_traverses_exit_axes(self) -> None:
        """Perpendicular boundary exits (X and Y axes) introduce zero cuts if spine only traverses X and Y."""
        # Boundary exit 0 is along X (East).
        # Boundary exit 1 is along Y (North).
        # Covered axes = {X, Y}.
        # Internal spine only traverses X (East) and Y (North).
        # All internal steps are on covered axes, so no splits are introduced.
        graph = _make_graph(
            rooms=[1, 2, 10, 11, 12, 20],
            edges=[
                (1, 10, Direction.east),
                (10, 11, Direction.north),
                (11, 12, Direction.east),
                (12, 20, Direction.north),
                (20, 2, Direction.north),
            ],
        )
        child = SolveTreeNode(
            node_id="s_perp_2d",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({1, 2, 10, 11, 12, 20}),
            children=(),
            metadata={"poles": (1, 2), "cut_type": CutType.SPLIT_PAIR},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,), graph=graph)
        transformed = TransverseSNodeSplitPass(graph).apply(tree)

        # No cuts introduced; child preserved intact
        assert len(transformed.root.children) == 1
        assert transformed.root.children[0].node_id == child.node_id
        assert transformed.root.children[0].rooms == child.rooms
        assert "severed_bridge_edges" not in transformed.root.metadata


class TestUnifiedHierarchicalPipeline:
    """Verify SolveTree.seed, decomposition passes, and unified pipeline tree construction."""

    def test_pipeline_integration_order(self) -> None:
        """Default hierarchical pipeline executes decomposition passes then downstream passes."""
        pipeline = PipelineTreeBuilder.default_pipeline()
        passes = pipeline.passes
        assert len(passes) == 4
        assert isinstance(passes[0], BlockCutDecompositionPass)
        assert isinstance(passes[1], SpqrDecompositionPass)
        assert isinstance(passes[2], TransverseSNodeSplitPass)
        assert isinstance(passes[3], OrthogonalBridgePromotionPass)

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

    def test_block_cut_decomposition_pass_bridge_cut(self) -> None:
        # Triangle 1-2-3 connected to room 4 via unique bridge edge 3 -> 4
        rooms = {v: _make_room(v) for v in (1, 2, 3, 4)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=Direction.north, cut=False),
            LayoutEdge(src=2, dst=3, direction=Direction.east, cut=False),
            LayoutEdge(src=3, dst=1, direction=Direction.south, cut=False),
            LayoutEdge(src=3, dst=4, direction=Direction.east, cut=False),
        ]
        graph = LayoutGraph(rooms=rooms, edges=tuple(edges))
        seed = SolveTree.seed(graph)
        result = BlockCutDecompositionPass().apply(seed)
        assert len(result.components) == 1
        root = result.root
        assert root.node_type == TreeNodeType.BLOCK
        assert len(root.children) == 1
        child = root.children[0]
        assert child.metadata.get("cut_type") == CutType.BRIDGE
        assert child.metadata.get("anchor_vnum") == 3
        assert isinstance(child.cut_contract, BridgeCut)
        assert child.cut_contract.bridge_edge == (3, 4)
        assert child.cut_contract.direction == int(Direction.east)
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


class TestOrthogonalBridgePromotionPass:
    """Verify orthogonal 1-cut bridge promotion across single and multi-hop collinear spines."""

    def test_orthogonal_bridge_promotion_pass_implements_protocol(self) -> None:
        pass_instance = OrthogonalBridgePromotionPass()
        assert isinstance(pass_instance, TreeTransformationPass)

    def test_empty_tree_returned_unchanged(self) -> None:
        empty = SolveTree(
            root=SolveTreeNode(
                node_id="empty",
                node_type=TreeNodeType.EMPTY,
                rooms=frozenset(),
            ),
            components=(),
        )
        assert OrthogonalBridgePromotionPass().apply(empty) is empty

    def test_invalid_bridge_direction_ignored(self) -> None:
        child = SolveTreeNode(
            node_id="c",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(2, 3), direction=99, weight=1, anchor_vnum=2
                )
            },
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2}),
            children=(child,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 2), direction=int(Direction.north), weight=1, anchor_vnum=1
                )
            },
        )
        root = SolveTreeNode(
            node_id="r",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(parent,),
        )
        tree = SolveTree(root=root, components=(root,))
        res = OrthogonalBridgePromotionPass().apply(tree)
        assert len(res.root.children) == 1
        assert res.root.children[0].children[0].node_id == "c"

    def test_multi_hop_collinear_spine_promotion(self) -> None:
        """Child bridge orthogonal to a 2-hop collinear spine bubbles to the owning ancestor."""
        gc = SolveTreeNode(
            node_id="orth_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({4}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(3, 4),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=3,
                )
            },
        )
        spine2 = SolveTreeNode(
            node_id="spine_2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3}),
            children=(gc,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(2, 3),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=2,
                )
            },
        )
        spine1 = SolveTreeNode(
            node_id="spine_1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2}),
            children=(spine2,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 2),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=1,
                )
            },
        )
        root_block = SolveTreeNode(
            node_id="root_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(spine1,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(0, 1),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=0,
                )
            },
        )
        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(root_block,),
        )
        tree = SolveTree(root=world_root, components=(root_block,))
        transformed = OrthogonalBridgePromotionPass().apply(tree)

        new_root_block = transformed.root.children[0]
        child_ids = [c.node_id for c in new_root_block.children]
        assert child_ids == ["spine_1", "orth_child"]

    def test_child_does_not_bubble_out_of_multi_room_cluster(self) -> None:
        """Bridge cut children do not bubble out of multi-room components."""
        child = SolveTreeNode(
            node_id="child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({20}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(10, 20),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=10,
                )
            },
        )
        parent = SolveTreeNode(
            node_id="parent_multi",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
            children=(child,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 10),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=1,
                )
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(parent,),
        )
        tree = SolveTree(root=root, components=(root,))
        transformed = OrthogonalBridgePromotionPass().apply(tree)

        # Child must remain inside parent_multi rather than bubbling into root
        new_parent = transformed.root.children[0]
        assert len(new_parent.children) == 1
        assert new_parent.children[0].node_id == "child"
        assert len(transformed.root.children) == 1

    def test_child_does_not_bubble_through_branching_junction(self) -> None:
        """Intermediate nodes with multiple children block child bubbling."""
        orth_child = SolveTreeNode(
            node_id="orth_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({4}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(3, 4),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=3,
                )
            },
        )
        rung2 = SolveTreeNode(
            node_id="rung_2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3}),
            children=(orth_child,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(2, 3),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=2,
                )
            },
        )
        other_branch = SolveTreeNode(
            node_id="other_branch",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({5}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(2, 5),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=2,
                )
            },
        )
        junction_rung1 = SolveTreeNode(
            node_id="junction_rung_1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2}),
            children=(rung2, other_branch),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 2),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=1,
                )
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(junction_rung1,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(0, 1),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=0,
                )
            },
        )
        tree = SolveTree(root=root, components=(root,))
        transformed = OrthogonalBridgePromotionPass().apply(tree)

        # orth_child bubbles out of rung_2 into junction_rung_1, but cannot bubble
        # through junction_rung_1 because junction_rung_1 is a branching hub.
        new_junction = transformed.root.children[0]
        assert new_junction.node_id == "junction_rung_1"
        assert len(transformed.root.children) == 1
        junction_child_ids = {c.node_id for c in new_junction.children}
        assert "orth_child" in junction_child_ids

    def test_multi_room_child_does_not_bubble(self) -> None:
        """A child cluster containing multiple rooms is never promoted across bridge cuts."""
        child_multi = SolveTreeNode(
            node_id="child_multi",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(2, 3),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=2,
                )
            },
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2}),
            children=(child_multi,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 2),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=1,
                )
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(parent,),
        )
        tree = SolveTree(root=root, components=(root,))
        transformed = OrthogonalBridgePromotionPass().apply(tree)

        new_parent = transformed.root.children[0]
        assert len(new_parent.children) == 1
        assert new_parent.children[0].node_id == "child_multi"
        assert len(transformed.root.children) == 1

    def test_bubbling_stops_at_non_collinear_intermediate_rung(self) -> None:
        """Bubbling along a hallway spine stops when an intermediate rung turns non-collinearly."""
        gc = SolveTreeNode(
            node_id="orth_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({4}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(3, 4),
                    direction=int(Direction.north),
                    weight=1,
                    anchor_vnum=3,
                )
            },
        )
        spine2 = SolveTreeNode(
            node_id="spine_2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3}),
            children=(gc,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(2, 3),
                    direction=int(Direction.east),
                    weight=1,
                    anchor_vnum=2,
                )
            },
        )
        # spine1 arrives from the south (non-collinear with spine2 which arrives from the east)
        spine1 = SolveTreeNode(
            node_id="spine_1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2}),
            children=(spine2,),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 2),
                    direction=int(Direction.south),
                    weight=1,
                    anchor_vnum=1,
                )
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(spine1,),
        )
        tree = SolveTree(root=root, components=(root,))
        transformed = OrthogonalBridgePromotionPass().apply(tree)

        # orth_child bubbles out of spine_2 to spine_1, but stops at spine_1
        # because spine_1 incoming direction (South) is not collinear with spine_2 (East).
        new_spine_1 = transformed.root.children[0]
        assert new_spine_1.node_id == "spine_1"
        assert len(transformed.root.children) == 1
        spine1_children = {c.node_id for c in new_spine_1.children}
        assert spine1_children == {"spine_2", "orth_child"}

    def test_midennir_solves_optimally_with_zero_collisions(self) -> None:
        """midennir.are solves to OPTIMAL status with 0 duplicate coordinates."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.parser import parse_file
        from romutil.types import LayoutOptions, SolverStatus, WorldGraph

        area = parse_file("tests/fixtures/areas/midennir.are")
        wg = WorldGraph.from_area_data(area)
        solver = HierarchicalLayoutSolver.production()
        res = solver.solve(wg, LayoutOptions(timeout_seconds=15))

        assert res.diagnostics.status == SolverStatus.OPTIMAL

        by_coord: dict[object, list[int]] = {}
        for v, coord in res.coordinates.items():
            by_coord.setdefault(coord, []).append(v)
        duplicates = {c: vs for c, vs in by_coord.items() if len(vs) > 1}
        assert not duplicates
