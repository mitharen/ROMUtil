"""Comprehensive unit tests for internal layout solver pipeline framework."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from romutil.interfaces import BaseLayoutSolver
from romutil.layout import (
    BoundingBox3D,
    CorridorSegment,
    GraphSimplifier,
    HierarchicalLayoutSolver,
    IdentityLayoutAssembler,
    LayoutAssembler,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    PassthroughSimplifier,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeBuilder,
    TreeSolution,
    TreeSolver,
)
from romutil.models import AreaHeader, Exit, Room
from romutil.types import Coord3D, LayoutOptions, SolvedLayout, SolverStatus, WorldGraph


class TestLayoutIR:
    """Tests for intermediate representations: immutability, defaults, conversions."""

    def test_layout_edge_immutability_and_defaults(self) -> None:
        edge = LayoutEdge(src=1, dst=2, direction=0)
        assert edge.src == 1
        assert edge.dst == 2
        assert edge.direction == 0
        assert edge.weight == 1
        assert edge.cut is False
        assert edge.metadata == {}

        with pytest.raises(FrozenInstanceError):
            edge.weight = 5  # type: ignore[misc]

        custom_edge = LayoutEdge(
            src=10,
            dst=20,
            direction=3,
            weight=10,
            cut=True,
            metadata={"tag": "secret"},
        )
        assert custom_edge.weight == 10
        assert custom_edge.cut is True
        assert custom_edge.metadata == {"tag": "secret"}

    def test_layout_graph_immutability_and_conversion(self) -> None:
        r1 = Room(vnum=1, name="R1", description="D1")
        e1 = LayoutEdge(src=1, dst=2, direction=0)
        graph = LayoutGraph(rooms={1: r1}, edges=[e1])  # passed as list

        assert graph.rooms[1] == r1
        assert isinstance(graph.edges, tuple)
        assert graph.edges == (e1,)

        with pytest.raises(FrozenInstanceError):
            graph.edges = ()  # type: ignore[misc]

    def test_layout_graph_from_world_graph(self) -> None:
        ex1 = Exit(direction=0, dst_vnum=2)
        ex2 = Exit(direction=1, dst_vnum=3)
        r1 = Room(vnum=1, name="Hall", description="Entry", exits=(ex1, ex2))
        r2 = Room(vnum=2, name="Room 2", description="North")
        r3 = Room(vnum=3, name="Room 3", description="East")
        world_graph = WorldGraph(
            rooms={1: r1, 2: r2, 3: r3},
            exits=(ex1, ex2),
            header=AreaHeader(filename="a.are", name="A", builder="B", vnum_min=1, vnum_max=3),
        )

        layout_graph = LayoutGraph.from_world_graph(world_graph)
        assert set(layout_graph.rooms.keys()) == {1, 2, 3}
        assert len(layout_graph.edges) == 2
        assert LayoutEdge(src=1, dst=2, direction=0) in layout_graph.edges
        assert LayoutEdge(src=1, dst=3, direction=1) in layout_graph.edges

    def test_layout_graph_from_empty_world_graph(self) -> None:
        world_graph = WorldGraph(rooms={}, exits=(), header=None)
        layout_graph = LayoutGraph.from_world_graph(world_graph)
        assert layout_graph.rooms == {}
        assert layout_graph.edges == ()

    def test_corridor_segment(self) -> None:
        segment = CorridorSegment(
            start_vnum=100,
            end_vnum=105,
            direction=0,
            intermediate_vnums=[101, 102, 103, 104],  # list -> tuple
        )
        assert segment.start_vnum == 100
        assert segment.end_vnum == 105
        assert segment.direction == 0
        assert isinstance(segment.intermediate_vnums, tuple)
        assert segment.intermediate_vnums == (101, 102, 103, 104)

        default_seg = CorridorSegment(start_vnum=1, end_vnum=2, direction=1)
        assert default_seg.intermediate_vnums == ()

        with pytest.raises(FrozenInstanceError):
            segment.direction = 2  # type: ignore[misc]

    def test_simplified_graph(self) -> None:
        graph = LayoutGraph(rooms={}, edges=())
        seg = CorridorSegment(start_vnum=1, end_vnum=2, direction=0)
        chord = LayoutEdge(src=1, dst=3, direction=2, cut=True)

        simp = SimplifiedGraph(
            graph=graph,
            corridors=[seg],
            severed_chords=[chord],
        )
        assert simp.graph == graph
        assert isinstance(simp.corridors, tuple)
        assert isinstance(simp.severed_chords, tuple)
        assert simp.corridors == (seg,)
        assert simp.severed_chords == (chord,)

        default_simp = SimplifiedGraph(graph=graph)
        assert default_simp.corridors == ()
        assert default_simp.severed_chords == ()

        with pytest.raises(FrozenInstanceError):
            simp.graph = graph  # type: ignore[misc]

    def test_solve_tree_node(self) -> None:
        child = SolveTreeNode(node_id="c1", node_type="leaf", rooms=[2])
        parent = SolveTreeNode(
            node_id="p1",
            node_type="cluster",
            rooms=[1],
            children=[child],
            metadata={"depth": 1},
        )
        assert parent.node_id == "p1"
        assert parent.node_type == "cluster"
        assert isinstance(parent.rooms, frozenset)
        assert parent.rooms == frozenset({1})
        assert isinstance(parent.children, tuple)
        assert parent.children == (child,)
        assert parent.metadata == {"depth": 1}

        default_node = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1, 2}))
        assert default_node.children == ()
        assert default_node.metadata == {}

        with pytest.raises(FrozenInstanceError):
            parent.node_id = "other"  # type: ignore[misc]

    def test_solve_tree(self) -> None:
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1}))
        comp = SolveTreeNode(node_id="comp1", node_type="leaf", rooms=frozenset({2}))
        tree = SolveTree(root=root, components=[comp])

        assert tree.root == root
        assert isinstance(tree.components, tuple)
        assert tree.components == (comp,)

        default_tree = SolveTree(root=root)
        assert default_tree.components == ()

        with pytest.raises(FrozenInstanceError):
            tree.root = comp  # type: ignore[misc]

    def test_node_solution_and_tree_solution(self) -> None:
        coord = Coord3D(1, 2, 3)
        bbox = BoundingBox3D(min_x=0, max_x=2, min_y=0, max_y=2, min_z=0, max_z=2)
        node_sol = NodeSolution(node_id="n1", local_coordinates={10: coord}, sub_boxes=(bbox,))
        assert node_sol.node_id == "n1"
        assert node_sol.local_coordinates[10] == coord
        assert node_sol.bounding_box == bbox
        assert node_sol.sub_boxes == (bbox,)

        default_sol = NodeSolution(node_id="n2", local_coordinates={})
        assert default_sol.bounding_box is None
        assert default_sol.sub_boxes == ()

        with pytest.raises(FrozenInstanceError):
            node_sol.node_id = "n3"  # type: ignore[misc]

        tree_sol = TreeSolution(node_solutions={"n1": node_sol, "n2": default_sol})
        assert tree_sol.node_solutions["n1"] == node_sol

        with pytest.raises(FrozenInstanceError):
            tree_sol.node_solutions = {}  # type: ignore[misc]


class TestProtocols:
    """Tests for stage protocols and runtime_checkable adherence."""

    def test_default_stages_satisfy_protocols(self) -> None:
        solver = HierarchicalLayoutSolver()
        assert isinstance(solver.simplifier, GraphSimplifier)
        assert isinstance(solver.tree_builder, TreeBuilder)
        assert isinstance(solver.tree_solver, TreeSolver)
        assert isinstance(solver.assembler, LayoutAssembler)
        assert isinstance(solver, BaseLayoutSolver)
        assert isinstance(PassthroughSimplifier(), GraphSimplifier)
        assert isinstance(IdentityLayoutAssembler(), LayoutAssembler)

    def test_custom_stage_runtime_check(self) -> None:
        class ValidSimplifier:
            def simplify(self, graph: LayoutGraph, options: LayoutOptions) -> SimplifiedGraph:
                return SimplifiedGraph(graph=graph)

        class IncompleteSimplifier:
            def other_method(self) -> None:
                pass

        assert isinstance(ValidSimplifier(), GraphSimplifier)
        assert not isinstance(IncompleteSimplifier(), GraphSimplifier)

        class ValidTreeBuilder:
            def build_tree(self, simplified: SimplifiedGraph, options: LayoutOptions) -> SolveTree:
                node = SolveTreeNode(node_id="r", node_type="t", rooms=frozenset())
                return SolveTree(root=node)

        assert isinstance(ValidTreeBuilder(), TreeBuilder)

        class ValidTreeSolver:
            def solve_tree(self, tree: SolveTree, options: LayoutOptions) -> TreeSolution:
                return TreeSolution(node_solutions={})

        assert isinstance(ValidTreeSolver(), TreeSolver)

        class ValidAssembler:
            def assemble(
                self,
                tree: SolveTree,
                solution: TreeSolution,
                simplified: SimplifiedGraph,
                options: LayoutOptions,
            ) -> SolvedLayout:
                from romutil.types import LayoutDiagnostics
                return SolvedLayout(
                    coordinates={},
                    diagnostics=LayoutDiagnostics(status=SolverStatus.FEASIBLE, duration_seconds=0.0),
                )

        assert isinstance(ValidAssembler(), LayoutAssembler)


class TestLayoutEngine:
    """Tests for default stage behaviors and HierarchicalLayoutSolver."""

    def test_passthrough_simplifier(self) -> None:
        r1 = Room(vnum=10, name="R10", description="Desc")
        lg = LayoutGraph(rooms={10: r1}, edges=())
        simp = PassthroughSimplifier().simplify(lg, LayoutOptions())
        assert simp.graph == lg
        assert simp.corridors == ()
        assert simp.severed_chords == ()

    def test_identity_layout_assembler(self) -> None:
        sol1 = NodeSolution(node_id="n1", local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(1, 0, 0)})
        sol2 = NodeSolution(node_id="n2", local_coordinates={3: Coord3D(2, 0, 0)})
        tree_sol = TreeSolution(node_solutions={"n1": sol1, "n2": sol2})
        tree = SolveTree(root=SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1, 2, 3})))
        simp = SimplifiedGraph(graph=LayoutGraph(rooms={}, edges=()))

        assembler = IdentityLayoutAssembler()
        solved = assembler.assemble(tree, tree_sol, simp, LayoutOptions())

        assert solved.coordinates[1] == Coord3D(0, 0, 0)
        assert solved.coordinates[2] == Coord3D(1, 0, 0)
        assert solved.coordinates[3] == Coord3D(2, 0, 0)
        assert solved.diagnostics.status == SolverStatus.FEASIBLE
        assert solved.diagnostics.duration_seconds == 0.0

    def test_solver_default_pipeline_end_to_end(self) -> None:
        r1 = Room(vnum=100, name="Entry", description="", exits=(Exit(direction=0, dst_vnum=101),))
        r2 = Room(vnum=101, name="Hall", description="", exits=(Exit(direction=2, dst_vnum=100),))
        wg = WorldGraph(rooms={100: r1, 101: r2}, exits=(r1.exits[0], r2.exits[0]))

        solver = HierarchicalLayoutSolver()
        result = solver.solve(wg)

        assert isinstance(result, SolvedLayout)
        assert set(result.coordinates.keys()) == {100, 101}
        assert result.coordinates[100] != result.coordinates[101]
        assert result.coordinates[101].y == result.coordinates[100].y + 1
        assert result.diagnostics.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.diagnostics.duration_seconds >= 0.0

    def test_solver_empty_world_graph(self) -> None:
        wg = WorldGraph(rooms={}, exits=(), header=None)
        solver = HierarchicalLayoutSolver()
        result = solver.solve(wg)

        assert isinstance(result, SolvedLayout)
        assert result.coordinates == {}
        assert result.diagnostics.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert result.diagnostics.duration_seconds >= 0.0

    def test_solver_custom_stage_injection_and_order(self) -> None:
        call_order: list[str] = []

        mock_simplifier = MagicMock(spec=GraphSimplifier)
        mock_tree_builder = MagicMock(spec=TreeBuilder)
        mock_tree_solver = MagicMock(spec=TreeSolver)
        mock_assembler = MagicMock(spec=LayoutAssembler)

        dummy_simp = SimplifiedGraph(graph=LayoutGraph(rooms={}, edges=()))
        dummy_tree = SolveTree(root=SolveTreeNode(node_id="r", node_type="t", rooms=frozenset()))
        dummy_sol = TreeSolution(node_solutions={})
        from romutil.types import LayoutDiagnostics
        dummy_layout = SolvedLayout(
            coordinates={42: Coord3D(1, 1, 1)},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
        )

        def simplifier_side_effect(graph: LayoutGraph, options: LayoutOptions) -> SimplifiedGraph:
            call_order.append("simplifier")
            return dummy_simp

        def tree_builder_side_effect(simplified: SimplifiedGraph, options: LayoutOptions) -> SolveTree:
            call_order.append("tree_builder")
            return dummy_tree

        def tree_solver_side_effect(tree: SolveTree, options: LayoutOptions) -> TreeSolution:
            call_order.append("tree_solver")
            return dummy_sol

        def assembler_side_effect(
            tree: SolveTree,
            solution: TreeSolution,
            simplified: SimplifiedGraph,
            options: LayoutOptions,
        ) -> SolvedLayout:
            call_order.append("assembler")
            return dummy_layout

        mock_simplifier.simplify.side_effect = simplifier_side_effect
        mock_tree_builder.build_tree.side_effect = tree_builder_side_effect
        mock_tree_solver.solve_tree.side_effect = tree_solver_side_effect
        mock_assembler.assemble.side_effect = assembler_side_effect

        custom_options = LayoutOptions(timeout_seconds=12, component_padding=5)
        solver = HierarchicalLayoutSolver(
            simplifier=mock_simplifier,
            tree_builder=mock_tree_builder,
            tree_solver=mock_tree_solver,
            assembler=mock_assembler,
        )

        r = Room(vnum=42, name="Custom", description="")
        wg = WorldGraph(rooms={42: r}, exits=())
        result = solver.solve(wg, options=custom_options)

        assert call_order == ["simplifier", "tree_builder", "tree_solver", "assembler"]

        mock_simplifier.simplify.assert_called_once()
        passed_graph, passed_opts = mock_simplifier.simplify.call_args[0]
        assert isinstance(passed_graph, LayoutGraph)
        assert passed_graph.rooms[42] == r
        assert passed_opts == custom_options

        mock_tree_builder.build_tree.assert_called_once_with(dummy_simp, custom_options)
        mock_tree_solver.solve_tree.assert_called_once_with(dummy_tree, custom_options)
        mock_assembler.assemble.assert_called_once_with(dummy_tree, dummy_sol, dummy_simp, custom_options)

        assert result.coordinates == {42: Coord3D(1, 1, 1)}
        assert result.diagnostics.status == SolverStatus.OPTIMAL
        assert result.diagnostics.duration_seconds >= 0.0
