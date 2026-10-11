"""Comprehensive unit tests for hierarchical tree solver in romutil.layout.milp.tree."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from romutil.layout.ir import (
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
    TreeSolution,
)
import romutil.layout.milp.cluster as cluster_module
from romutil.layout.milp import HierarchicalTreeSolver as TopLevelTreeSolver
from romutil.layout.milp.tree import HierarchicalTreeSolver
from romutil.layout.protocols import TreeSolver
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError


def _make_room(vnum: int, exits: dict[Direction, int] | None = None) -> Room:
    exits_dict = exits or {}
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(
            Exit(direction=d, dst_vnum=to_v)
            for d, to_v in exits_dict.items()
        ),
    )


class TestHierarchicalTreeSolverBasics:
    """Basic initialization and protocol conformance tests."""

    def test_implements_tree_solver_protocol(self) -> None:
        """HierarchicalTreeSolver is an instance of TreeSolver protocol."""
        solver = HierarchicalTreeSolver()
        assert isinstance(solver, TreeSolver)

    def test_canonical_and_top_level_exports(self) -> None:
        """HierarchicalTreeSolver is identically exported from tree module and milp package."""
        assert HierarchicalTreeSolver is TopLevelTreeSolver
        assert "HierarchicalTreeSolver" not in cluster_module.__all__

    def test_default_and_custom_initialization(self) -> None:
        """Initializes with default or custom solver name and graph."""
        solver_default = HierarchicalTreeSolver()
        assert solver_default.graph is None
        assert solver_default.solver_name == "cbc"

        g = LayoutGraph(rooms={1: _make_room(1)}, edges=())
        solver_custom = HierarchicalTreeSolver(graph=g, solver_name="glpk")
        assert solver_custom.graph is g
        assert solver_custom.solver_name == "glpk"


class TestHierarchicalTreeSolverExecution:
    """Bottom-up traversal and cluster solving execution tests."""

    def test_solve_single_leaf_root(self) -> None:
        """Solves a tree containing a single leaf cluster."""
        r1 = _make_room(1, {Direction.east: 2})
        r2 = _make_room(2, {Direction.west: 1})
        graph = LayoutGraph(rooms={1: r1, 2: r2}, edges=(LayoutEdge(src=1, dst=2, direction=Direction.east),))

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(),
        )
        tree = SolveTree(root=root, graph=graph)
        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert isinstance(sol, TreeSolution)
        assert "root" in sol.node_solutions
        root_sol = sol.node_solutions["root"]
        assert 1 in root_sol.local_coordinates
        assert 2 in root_sol.local_coordinates

    def test_solve_nested_hierarchy_post_order(self) -> None:
        """Solves nested hierarchy bottom-up, passing child solutions to parent container."""
        # Child 1: rooms 10, 11
        # Child 2: rooms 20, 21
        # Root: parent container containing child 1 and child 2
        r10 = _make_room(10, {Direction.east: 11})
        r11 = _make_room(11, {Direction.west: 10, Direction.north: 20})
        r20 = _make_room(20, {Direction.south: 11, Direction.east: 21})
        r21 = _make_room(21, {Direction.west: 20})

        graph = LayoutGraph(
            rooms={10: r10, 11: r11, 20: r20, 21: r21},
            edges=(
                LayoutEdge(src=10, dst=11, direction=Direction.east),
                LayoutEdge(src=11, dst=20, direction=Direction.north),
                LayoutEdge(src=20, dst=21, direction=Direction.east),
            ),
        )

        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 11}),
            children=(),
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({20, 21}),
            children=(),
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(),
            children=(c1, c2),
        )
        tree = SolveTree(root=root, graph=graph)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "c1" in sol.node_solutions
        assert "c2" in sol.node_solutions
        assert "root" in sol.node_solutions
        root_sol = sol.node_solutions["root"]
        child_offsets = root_sol.metadata.get("child_offsets", {})
        assert "c1" in child_offsets
        assert "c2" in child_offsets

    def test_solve_with_detached_components(self) -> None:
        """Solves both root and additional independent components in tree.components."""
        r1 = _make_room(1)
        r2 = _make_room(2)
        graph = LayoutGraph(rooms={1: r1, 2: r2}, edges=())

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        comp = SolveTreeNode(
            node_id="comp",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({2}),
            children=(),
        )
        tree = SolveTree(root=root, components=(comp,), graph=graph)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "root" in sol.node_solutions
        assert "comp" in sol.node_solutions
        assert 1 in sol.node_solutions["root"].local_coordinates
        assert 2 in sol.node_solutions["comp"].local_coordinates

    def test_synthesizes_graph_when_none_provided(self) -> None:
        """Collects all rooms across root and components when layout graph is None."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({101, 102}),
            children=(),
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100}),
            children=(c1,),
        )
        comp = SolveTreeNode(
            node_id="comp",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({200}),
            children=(),
        )
        tree = SolveTree(root=root, components=(comp,), graph=None)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "c1" in sol.node_solutions
        assert "root" in sol.node_solutions
        assert "comp" in sol.node_solutions
        assert 100 in sol.node_solutions["root"].local_coordinates
        assert 101 in sol.node_solutions["c1"].local_coordinates
        assert 200 in sol.node_solutions["comp"].local_coordinates

    def test_custom_graph_override(self) -> None:
        """Solver instance graph overrides tree graph when explicitly passed."""
        r1 = _make_room(1)
        g_custom = LayoutGraph(rooms={1: r1}, edges=())
        g_tree = LayoutGraph(rooms={}, edges=())

        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        tree = SolveTree(root=root, graph=g_tree)

        solver = HierarchicalTreeSolver(graph=g_custom)
        sol = solver.solve_tree(tree)
        assert 1 in sol.node_solutions["root"].local_coordinates

    def test_node_caching_behavior(self) -> None:
        """Returns cached node solution if the same node is encountered multiple times."""
        child = SolveTreeNode(
            node_id="shared_child",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({5}),
            children=(),
        )
        # Tree where root has the child, and components also references the same child node
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(),
            children=(child,),
        )
        tree = SolveTree(root=root, components=(child,), graph=None)

        solver = HierarchicalTreeSolver()
        sol = solver.solve_tree(tree)

        assert "shared_child" in sol.node_solutions
        assert "root" in sol.node_solutions
        assert len(sol.node_solutions) == 2

    def test_forwards_options_and_solver_name(self) -> None:
        """Forwards solver options and solver name to solve_cluster."""
        node = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        tree = SolveTree(root=node, graph=None)
        options = LayoutOptions(timeout_seconds=42)

        with patch("romutil.layout.milp.tree.solve_cluster") as mock_solve:
            mock_solve.return_value = NodeSolution(
                node_id="root",
                local_coordinates={1: Coord3D(0, 0, 0)},
            )
            solver = HierarchicalTreeSolver(solver_name="custom_solver")
            solver.solve_tree(tree, options=options)

            mock_solve.assert_called_once()
            _, kwargs = mock_solve.call_args
            assert kwargs["solver_name"] == "custom_solver"
            assert kwargs["options"] is options

    def test_propagates_solver_error(self) -> None:
        """Propagates LayoutSolverError when underlying cluster solver fails."""
        node = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1}),
            children=(),
        )
        tree = SolveTree(root=node, graph=None)

        with patch("romutil.layout.milp.tree.solve_cluster") as mock_solve:
            mock_solve.side_effect = LayoutSolverError("Simulated failure")
            solver = HierarchicalTreeSolver()
            with pytest.raises(LayoutSolverError, match="Simulated failure"):
                solver.solve_tree(tree)
