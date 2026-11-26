"""Comprehensive unit tests for hierarchical tree solver in romutil.layout.milp.tree."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from romutil.layout.assembly.hierarchical import HierarchicalLayoutAssembler
from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
    TreeSolution,
)
import romutil.layout.milp.cluster as cluster_module
from romutil.layout.milp import (
    HierarchicalTreeSolver as TopLevelTreeSolver,
    pack_world_root_components,
    solve_cluster,
)
from romutil.layout.milp.tree import (
    HierarchicalTreeSolver,
    _as_coord3d,
    _child_attachment_rooms,
    _extract_status,
    _find_bridge_ray_blockers,
    _has_outward_exit_in_direction,
    _ray_continues_beyond_neighbor,
    _ray_hits_box,
    _ray_hits_box_with_extent,
    _resynthesize_split_child_solution,
    _safe_displacement,
    _try_detach_articulation_ray_blocking_subchild,
    _try_resolve_articulation_ray_blockage,
    _try_split_ray_blocked_child,
)
from romutil.layout.protocols import TreeSolver
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus
from tests.regressions.invariants import assert_zero_coordinate_collisions


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


class TestWorldRootComponentPacking:
    """Unit and integration tests for sequential X-axis packing of disconnected components under WORLD_ROOT."""

    def test_pack_world_root_empty_children(self) -> None:
        """Constructs an empty node solution when WORLD_ROOT has no children."""
        node = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(),
        )
        sol = pack_world_root_components(node=node, child_solutions={})
        assert sol.node_id == "world_root"
        assert sol.local_coordinates == {}
        assert sol.metadata.get("child_offsets") == {}

    def test_pack_world_root_sequential_x_alignment(self) -> None:
        """Packs multiple child component solutions sequentially along X-axis with padding."""
        ch0 = SolveTreeNode(
            node_id="comp_0",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
        )
        ch1 = SolveTreeNode(
            node_id="comp_1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 11}),
        )
        ch2 = SolveTreeNode(
            node_id="comp_2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100}),
        )
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1, ch2),
        )

        sol_0 = NodeSolution(
            node_id="comp_0",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(10, 4, 1)},
            sub_boxes=(BoundingBox3D(0, 10, 0, 4, 0, 1),),
            metadata={
                "bounding_box": BoundingBox3D(0, 10, 0, 4, 0, 1),
                "all_room_coordinates": {1: Coord3D(0, 0, 0), 2: Coord3D(10, 4, 1)},
            },
        )
        sol_1 = NodeSolution(
            node_id="comp_1",
            local_coordinates={10: Coord3D(0, 0, 0), 11: Coord3D(5, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 5, 0, 2, 0, 0),),
            metadata={
                "bounding_box": BoundingBox3D(0, 5, 0, 2, 0, 0),
                "all_room_coordinates": {10: Coord3D(0, 0, 0), 11: Coord3D(5, 2, 0)},
            },
        )
        sol_2 = NodeSolution(
            node_id="comp_2",
            local_coordinates={100: Coord3D(0, 0, 0)},
            metadata={
                "bounding_box": BoundingBox3D(0, 0, 0, 0, 0, 0),
                "all_room_coordinates": {100: Coord3D(0, 0, 0)},
            },
        )

        child_solutions = {
            "comp_0": sol_0,
            "comp_1": sol_1,
            "comp_2": sol_2,
        }

        options = LayoutOptions(component_padding=3)
        sol = pack_world_root_components(node=root, child_solutions=child_solutions, options=options)

        offsets = sol.metadata["child_offsets"]
        assert offsets["comp_0"] == Coord3D(0, 0, 0)
        # comp_0 max_x is 10, so comp_1 starts at 10 + 3 - 0 = 13
        assert offsets["comp_1"] == Coord3D(13, 0, 0)
        # comp_1 max_x is 13 + 5 = 18, so comp_2 starts at 18 + 3 - 0 = 21
        assert offsets["comp_2"] == Coord3D(21, 0, 0)

        # Descendant room coordinates
        all_rooms = sol.metadata["all_room_coordinates"]
        assert all_rooms[1] == Coord3D(0, 0, 0)
        assert all_rooms[2] == Coord3D(10, 4, 1)
        assert all_rooms[10] == Coord3D(13, 0, 0)
        assert all_rooms[11] == Coord3D(18, 2, 0)
        assert all_rooms[100] == Coord3D(21, 0, 0)

        assert sol.metadata["status"] == SolverStatus.OPTIMAL
        bbox = sol.bounding_box
        assert bbox is not None
        assert bbox.min_x == 0
        assert bbox.max_x == 21

    def test_solve_cluster_delegates_world_root_to_packing(self) -> None:
        """solve_cluster bypasses MILP model formulation for WORLD_ROOT nodes."""
        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BLOCK, rooms=frozenset({1}))
        ch1 = SolveTreeNode(node_id="comp_1", node_type=TreeNodeType.BLOCK, rooms=frozenset({2}))
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1),
        )

        sol_0 = NodeSolution(
            node_id="comp_0",
            local_coordinates={1: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 4, 0, 0, 0, 0),),
        )
        sol_1 = NodeSolution(
            node_id="comp_1",
            local_coordinates={2: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 2, 0, 0, 0, 0),),
        )

        sol = solve_cluster(
            node=root,
            child_solutions={"comp_0": sol_0, "comp_1": sol_1},
            options=LayoutOptions(component_padding=2),
        )

        assert sol.node_id == "world_root"
        assert sol.metadata["child_offsets"]["comp_0"] == Coord3D(0, 0, 0)
        assert sol.metadata["child_offsets"]["comp_1"] == Coord3D(6, 0, 0)
        assert sol.metadata["status"] == SolverStatus.OPTIMAL

    def test_hierarchical_tree_solver_world_root_packing(self) -> None:
        """HierarchicalTreeSolver solves child components and packs WORLD_ROOT sequentially."""
        r1 = _make_room(1)
        r2 = _make_room(2)
        graph = LayoutGraph(rooms={1: r1, 2: r2}, edges=())

        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BICONNECTED_BLOCK, rooms=frozenset({1}))
        ch1 = SolveTreeNode(node_id="comp_1", node_type=TreeNodeType.BICONNECTED_BLOCK, rooms=frozenset({2}))
        world_root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1),
        )
        tree = SolveTree(root=world_root, components=(ch0, ch1), graph=graph)

        solver = HierarchicalTreeSolver()
        tree_sol = solver.solve_tree(tree, options=LayoutOptions(component_padding=4))

        assert "world_root" in tree_sol.node_solutions
        root_sol = tree_sol.node_solutions["world_root"]
        assert root_sol.metadata["child_offsets"]["comp_0"] == Coord3D(0, 0, 0)
        assert root_sol.metadata["child_offsets"]["comp_1"] == Coord3D(4, 0, 0)
        assert root_sol.metadata["all_room_coordinates"][1] == Coord3D(0, 0, 0)
        assert root_sol.metadata["all_room_coordinates"][2] == Coord3D(4, 0, 0)

    def test_sewer_disconnected_rooms_compact_placement(self) -> None:
        """Verify layout solver places disconnected rooms tightly adjacent without large spatial void."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.models import Direction, Exit, Room
        from romutil.types import WorldGraph

        # Main connected component: rooms 1-4 in a line
        rooms = {
            1: Room(vnum=1, name="Room 1", description="", exits=(Exit(direction=int(Direction.east), dst_vnum=2),)),
            2: Room(vnum=2, name="Room 2", description="", exits=(
                Exit(direction=int(Direction.west), dst_vnum=1),
                Exit(direction=int(Direction.east), dst_vnum=3),
            )),
            3: Room(vnum=3, name="Room 3", description="", exits=(
                Exit(direction=int(Direction.west), dst_vnum=2),
                Exit(direction=int(Direction.east), dst_vnum=4),
            )),
            4: Room(vnum=4, name="Room 4", description="", exits=(Exit(direction=int(Direction.west), dst_vnum=3),)),
            # Disconnected component rooms
            5: Room(vnum=5, name="Room 5", description="", exits=()),
            6: Room(vnum=6, name="Room 6", description="", exits=()),
        }
        world_graph = WorldGraph(rooms=rooms, exits=[e for r in rooms.values() for e in r.exits])
        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph)

        assert 5 in layout.coordinates
        assert 6 in layout.coordinates

        # Main component rooms
        main_xs = [c.x for v, c in layout.coordinates.items() if v in (1, 2, 3, 4)]
        main_max_x = max(main_xs)

        # Disconnected rooms must be placed within a small constant padding of main component
        assert layout.coordinates[5].x > main_max_x
        assert layout.coordinates[6].x > main_max_x
        assert layout.coordinates[5].x <= main_max_x + 5
        assert layout.coordinates[6].x <= main_max_x + 5

    test_sewer_layout_disconnected_rooms_placement = test_sewer_disconnected_rooms_compact_placement

    def test_pack_world_root_missing_child_solution_and_empty_rooms(self) -> None:
        """Handles child nodes with missing solutions or empty coordinates gracefully."""
        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BLOCK, rooms=frozenset())
        ch1 = SolveTreeNode(node_id="comp_1", node_type=TreeNodeType.BLOCK, rooms=frozenset())
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0, ch1),
        )

        # Neither ch0 nor ch1 is provided in child_solutions
        sol = pack_world_root_components(node=root, child_solutions={})
        assert sol.node_id == "world_root"
        assert sol.metadata["child_offsets"]["comp_0"] == Coord3D(0, 0, 0)
        assert sol.metadata["child_offsets"]["comp_1"] == Coord3D(2, 0, 0)
        assert sol.metadata["all_room_coordinates"] == {}
        assert sol.bounding_box is None

    def test_pack_world_root_all_room_coordinates_fallback(self) -> None:
        """Derives child bounding box from metadata all_room_coordinates when sub_boxes is empty."""
        ch0 = SolveTreeNode(node_id="comp_0", node_type=TreeNodeType.BLOCK, rooms=frozenset({1}))
        root = SolveTreeNode(
            node_id="world_root",
            node_type=TreeNodeType.WORLD_ROOT,
            rooms=frozenset(),
            children=(ch0,),
        )
        sol_0 = NodeSolution(
            node_id="comp_0",
            local_coordinates={},
            sub_boxes=(),
            metadata={"all_room_coordinates": {1: Coord3D(3, 4, 0)}},
        )
        sol = pack_world_root_components(node=root, child_solutions={"comp_0": sol_0})
        assert sol.bounding_box == BoundingBox3D(3, 3, 4, 4, 0, 0)

    def test_midgaard_sewers_composite_disconnected_rooms_placement(self) -> None:
        """Verify composite multi-cluster layout places disconnected rooms adjacent without void."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.models import Direction, Exit, Room
        from romutil.types import LayoutOptions, WorldGraph

        # Two distinct connected components (composite) plus disconnected rooms
        rooms = {
            101: Room(vnum=101, name="Room 101", description="", exits=(Exit(direction=int(Direction.east), dst_vnum=102),)),
            102: Room(vnum=102, name="Room 102", description="", exits=(Exit(direction=int(Direction.west), dst_vnum=101),)),
            201: Room(vnum=201, name="Room 201", description="", exits=(Exit(direction=int(Direction.south), dst_vnum=202),)),
            202: Room(vnum=202, name="Room 202", description="", exits=(Exit(direction=int(Direction.north), dst_vnum=201),)),
            301: Room(vnum=301, name="Room 301", description="", exits=()),
            302: Room(vnum=302, name="Room 302", description="", exits=()),
            303: Room(vnum=303, name="Room 303", description="", exits=()),
        }
        world_graph = WorldGraph(rooms=rooms, exits=[e for r in rooms.values() for e in r.exits])

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph, LayoutOptions(timeout_seconds=5))

        disconnected = [301, 302, 303]
        for r in disconnected:
            assert r in layout.coordinates

        main_xs = [c.x for v, c in layout.coordinates.items() if v not in disconnected]
        main_max_x = max(main_xs)

        for r in disconnected:
            assert layout.coordinates[r].x > main_max_x
            assert layout.coordinates[r].x <= main_max_x + 10


class TestDynamicBridgeRaySplit:
    """Tests for post-solve outward ray-blockage detection and dynamic orthogonal bridge splitting."""

    def test_splits_ray_blocked_bridge_child_and_resynthesizes_cleanly(self) -> None:
        """Splits a ray-blocked BridgeCut child across an orthogonal internal bridge and re-synthesizes cleanly."""
        r1 = _make_room(1, {Direction.east: 2})
        r2 = _make_room(2, {Direction.west: 1, Direction.east: 3})
        r3 = _make_room(3, {Direction.west: 2, Direction.north: 4})
        r4 = _make_room(4, {Direction.south: 3, Direction.east: 40, Direction.west: 5})
        r40 = _make_room(40, {Direction.west: 4, Direction.south: 8})
        r8 = _make_room(8, {Direction.north: 40, Direction.west: 7})
        r7 = _make_room(7, {Direction.east: 8, Direction.north: 6})
        r6 = _make_room(6, {Direction.south: 7, Direction.north: 5})
        r5 = _make_room(5, {Direction.south: 6, Direction.east: 4})

        edges = (
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
            LayoutEdge(src=4, dst=40, direction=Direction.east, weight=1),
            LayoutEdge(src=40, dst=8, direction=Direction.south, weight=2),
            LayoutEdge(src=8, dst=7, direction=Direction.west, weight=4),
            LayoutEdge(src=7, dst=6, direction=Direction.north, weight=1),
            LayoutEdge(src=6, dst=5, direction=Direction.north, weight=1),
            LayoutEdge(src=5, dst=4, direction=Direction.east, weight=3),
        )
        graph = LayoutGraph(
            rooms={r.vnum: r for r in (r1, r2, r3, r4, r40, r8, r7, r6, r5)},
            edges=edges,
        )

        child = SolveTreeNode(
            node_id="child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 40, 8, 7, 6, 5}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 2,
                "bridge_edge": (2, 3),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2),
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=root, graph=graph)

        sol = HierarchicalTreeSolver().solve_tree(tree, options=LayoutOptions(timeout_seconds=15))

        # Synthetic split piece IDs must never leak into TreeSolution or root child_offsets
        assert set(sol.node_solutions.keys()) == {"child", "root"}
        assert set(sol.node_solutions["root"].metadata["child_offsets"].keys()) == {"child"}
        assert set(sol.node_solutions["child"].local_coordinates.keys()) == {3, 4, 40, 8, 7, 6, 5}
        assert 2 in sol.node_solutions["child"].port_coordinates
        assert set(sol.node_solutions["root"].metadata["child_sub_boxes"].keys()) == {"child"}

        assembled = HierarchicalLayoutAssembler().assemble(
            tree=tree,
            solution=sol,
            simplified=SimplifiedGraph(graph=graph),
        )
        assert_zero_coordinate_collisions(assembled)
        assert assembled.coordinates[1].x < assembled.coordinates[2].x
        assert assembled.coordinates[2].x < assembled.coordinates[3].x

    def test_recursive_two_level_ray_blockage_split(self) -> None:
        """Recurses ray-blockage splitting across two levels when the Level-1 wing is also blocked."""
        r1 = _make_room(1, {Direction.east: 2})
        r2 = _make_room(2, {Direction.west: 1, Direction.east: 3})
        # Room 3 has a warp-cut outward South exit toward 20 in root
        r3 = _make_room(3, {Direction.west: 2, Direction.south: 20, Direction.north: 4})
        r20 = _make_room(20, {Direction.north: 3})
        r4 = _make_room(4, {Direction.south: 3, Direction.east: 5})
        r5 = _make_room(5, {Direction.west: 4, Direction.south: 6, Direction.north: 11})
        r6 = _make_room(6, {Direction.north: 5, Direction.west: 7})
        r7 = _make_room(7, {Direction.east: 6, Direction.west: 8})
        r8 = _make_room(8, {Direction.east: 7, Direction.north: 9})
        r9 = _make_room(9, {Direction.south: 8, Direction.north: 10})
        r10 = _make_room(10, {Direction.south: 9, Direction.east: 11})
        r11 = _make_room(11, {Direction.west: 10, Direction.south: 5})

        edges = (
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=20, direction=Direction.south, weight=1, cut=True),
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
            LayoutEdge(src=4, dst=5, direction=Direction.east, weight=1),
            LayoutEdge(src=5, dst=6, direction=Direction.south, weight=3),
            LayoutEdge(src=6, dst=7, direction=Direction.west, weight=1),
            LayoutEdge(src=7, dst=8, direction=Direction.west, weight=4),
            LayoutEdge(src=8, dst=9, direction=Direction.north, weight=2),
            LayoutEdge(src=9, dst=10, direction=Direction.north, weight=2),
            LayoutEdge(src=10, dst=11, direction=Direction.east, weight=5),
            LayoutEdge(src=11, dst=5, direction=Direction.south, weight=1),
        )
        graph = LayoutGraph(
            rooms={r.vnum: r for r in (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r20)},
            edges=edges,
        )

        child = SolveTreeNode(
            node_id="child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 5, 6, 7, 8, 9, 10, 11}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 2,
                "bridge_edge": (2, 3),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2),
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=root, graph=graph)

        solved_node_ids: list[str] = []
        orig_solve_cluster = solve_cluster

        def spy_solve_cluster(node, child_solutions=None, graph=None, options=None, solver_name="cbc", debug_solve_dir=None):
            solved_node_ids.append(node.node_id)
            return orig_solve_cluster(
                node=node,
                child_solutions=child_solutions,
                graph=graph,
                options=options,
                solver_name=solver_name,
                debug_solve_dir=debug_solve_dir,
            )

        with patch("romutil.layout.milp.tree.solve_cluster", side_effect=spy_solve_cluster):
            sol = HierarchicalTreeSolver().solve_tree(tree, options=LayoutOptions(timeout_seconds=15))

        assert "child__base" in solved_node_ids
        assert "child__wing" in solved_node_ids
        assert "child__wing__base" in solved_node_ids
        assert "child__wing__wing" in solved_node_ids
        assert set(sol.node_solutions.keys()) == {"child", "root"}
        assert set(sol.node_solutions["root"].metadata["child_offsets"].keys()) == {"child"}

        assembled = HierarchicalLayoutAssembler().assemble(
            tree=tree,
            solution=sol,
            simplified=SimplifiedGraph(graph=graph),
        )
        assert_zero_coordinate_collisions(assembled)

    def test_skips_split_when_anchor_has_no_outward_exit_in_reverse_direction(self) -> None:
        """Does not split a BridgeCut child whose anchor exits vertically instead of opposite -d."""
        r1 = _make_room(1, {Direction.down: 2})
        r2 = _make_room(2, {Direction.up: 1, Direction.east: 3})
        r3 = _make_room(3, {Direction.west: 2, Direction.north: 4})
        r4 = _make_room(4, {Direction.south: 3, Direction.west: 5})
        r5 = _make_room(5, {Direction.east: 4, Direction.west: 6})
        r6 = _make_room(6, {Direction.east: 5, Direction.south: 7})
        r7 = _make_room(7, {Direction.north: 6})

        edges = (
            LayoutEdge(src=2, dst=1, direction=Direction.up, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
            LayoutEdge(src=4, dst=5, direction=Direction.west, weight=1),
            LayoutEdge(src=5, dst=6, direction=Direction.west, weight=1),
            LayoutEdge(src=6, dst=7, direction=Direction.south, weight=1),
        )
        graph = LayoutGraph(
            rooms={r.vnum: r for r in (r1, r2, r3, r4, r5, r6, r7)},
            edges=edges,
        )
        child = SolveTreeNode(
            node_id="child_z_escape",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 5, 6, 7}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 2,
                "bridge_edge": (2, 3),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2),
            },
        )
        # Place room 7 strictly West of port 2 along y=0
        child_sol = NodeSolution(
            node_id="child_z_escape",
            local_coordinates={
                3: Coord3D(3, 0, 0),
                4: Coord3D(3, 1, 0),
                5: Coord3D(2, 1, 0),
                6: Coord3D(1, 1, 0),
                7: Coord3D(1, 0, 0),
            },
            port_coordinates={2: Coord3D(2, 0, 0)},
        )
        assert _find_bridge_ray_blockers(child, child_sol, graph) is None
        assert _try_split_ray_blocked_child(child, child_sol, graph) is None

    def test_skips_split_when_ray_is_unblocked(self) -> None:
        """Does not split a BridgeCut child when all owned geometry clears the outward ray."""
        r1 = _make_room(1, {Direction.east: 2})
        r2 = _make_room(2, {Direction.west: 1, Direction.east: 3})
        r3 = _make_room(3, {Direction.west: 2, Direction.north: 4})
        r4 = _make_room(4, {Direction.south: 3})
        edges = (
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
        )
        graph = LayoutGraph(rooms={r.vnum: r for r in (r1, r2, r3, r4)}, edges=edges)
        child = SolveTreeNode(
            node_id="child_clean",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4}),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 2,
                "bridge_edge": (2, 3),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2),
            },
        )
        child_sol = NodeSolution(
            node_id="child_clean",
            local_coordinates={3: Coord3D(1, 0, 0), 4: Coord3D(1, 1, 0)},
            port_coordinates={2: Coord3D(0, 0, 0)},
        )
        assert _find_bridge_ray_blockers(child, child_sol, graph) is None

    def test_skips_split_when_no_valid_orthogonal_bridge_exists(self) -> None:
        """Covers negative guard paths: cycles, straddling sub-children, vertical/non-cardinal cuts."""
        # Case A: Vertical BridgeCut direction is ignored
        g_base = LayoutGraph(
            rooms={1: _make_room(1), 2: _make_room(2), 3: _make_room(3)},
            edges=(LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),),
        )
        vert_child = SolveTreeNode(
            node_id="vert",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3}),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.up), 1, 2)},
        )
        assert _find_bridge_ray_blockers(vert_child, NodeSolution("vert", {3: Coord3D(0, 0, 0)}), g_base) is None

        # Case B: Blocker inside a 2-connected cycle with no orthogonal bridge edge
        edges_cycle = (
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
            LayoutEdge(src=4, dst=5, direction=Direction.west, weight=2),
            LayoutEdge(src=5, dst=6, direction=Direction.south, weight=1),
            LayoutEdge(src=6, dst=3, direction=Direction.east, weight=1),
        )
        g_cycle = LayoutGraph(rooms={v: _make_room(v) for v in (1, 2, 3, 4, 5, 6)}, edges=edges_cycle)
        cyc_child = SolveTreeNode(
            node_id="cyc",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 5, 6}),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        cyc_sol = NodeSolution(
            node_id="cyc",
            local_coordinates={3: Coord3D(2, 0, 0), 4: Coord3D(2, 1, 0), 5: Coord3D(0, 1, 0), 6: Coord3D(0, 0, 0)},
            port_coordinates={2: Coord3D(1, 0, 0)},
        )
        # Room 6 blocks 2's West ray, yet all internal edges form a 4-cycle so no bridge exists
        assert _find_bridge_ray_blockers(cyc_child, cyc_sol, g_cycle) is not None
        assert _try_split_ray_blocked_child(cyc_child, cyc_sol, g_cycle) is None

        # Case C: Orthogonal bridge exists, but a sub-child attaches to both sides via sub-box blocking ray
        sub_c = SolveTreeNode(
            node_id="sub_shared",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({99}),
            metadata={},
        )
        edges_straddle = (
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
            LayoutEdge(src=3, dst=99, direction=Direction.west, weight=1),
            LayoutEdge(src=4, dst=99, direction=Direction.south, weight=1),
        )
        g_straddle = LayoutGraph(rooms={v: _make_room(v) for v in (1, 2, 3, 4, 99)}, edges=edges_straddle)
        straddle_child = SolveTreeNode(
            node_id="straddle",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4}),
            children=(sub_c,),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        straddle_sol = NodeSolution(
            node_id="straddle",
            local_coordinates={3: Coord3D(3, 0, 0), 4: Coord3D(3, 1, 0)},
            port_coordinates={2: Coord3D(2, 0, 0)},
            metadata={
                "child_offsets": {"sub_shared": Coord3D(0, 0, 0)},
                "child_sub_boxes": {"sub_shared": (BoundingBox3D(0, 1, 0, 1, 0, 0),)},
            },
        )
        assert _try_split_ray_blocked_child(straddle_child, straddle_sol, g_straddle) is None

        # Case D: Nested sub-child resynthesis preserves sub-offsets and stretched corridor cells
        nested_sub = SolveTreeNode(node_id="sc1", node_type=TreeNodeType.RIGID_CLUSTER, rooms=frozenset({40}))
        orig_parent_child = SolveTreeNode(
            node_id="c_orig",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4}),
            children=(nested_sub,),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        p_base = SolveTreeNode(node_id="c_orig__base", node_type=TreeNodeType.BLOCK, rooms=frozenset({3}))
        p_wing = SolveTreeNode(
            node_id="c_orig__wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({4}),
            children=(nested_sub,),
            metadata={"cut_contract": BridgeCut((3, 4), int(Direction.north), 1, 3)},
        )
        s_base = NodeSolution("c_orig__base", {3: Coord3D(0, 0, 0)}, port_coordinates={2: Coord3D(-1, 0, 0)}, metadata={"status": SolverStatus.OPTIMAL})
        s_wing = NodeSolution(
            "c_orig__wing",
            {4: Coord3D(0, 0, 0)},
            metadata={
                "status": SolverStatus.OPTIMAL,
                "child_offsets": {"sc1": Coord3D(1, 0, 0)},
                "child_sub_boxes": {"sc1": (BoundingBox3D(0, 0, 0, 0, 0, 0),)},
                "all_room_coordinates": {4: Coord3D(0, 0, 0), 40: Coord3D(1, 0, 0)},
            },
        )
        s_parent = NodeSolution(
            "root",
            {1: Coord3D(0, 0, 0), 2: Coord3D(1, 0, 0)},
            metadata={
                "status": SolverStatus.OPTIMAL,
                "child_offsets": {"c_orig__base": Coord3D(2, 0, 0), "c_orig__wing": Coord3D(2, 3, 0)},
            },
        )
        sc_sol = NodeSolution("sc1", {40: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),))
        merged_sol, orig_off, orig_sbs = _resynthesize_split_child_solution(
            orig_child=orig_parent_child,
            pieces=[(p_base, s_base), (p_wing, s_wing)],
            parent_sol=s_parent,
            solutions={"sc1": sc_sol},
        )
        assert orig_off == Coord3D(2, 0, 0)
        assert merged_sol.local_coordinates[3] == Coord3D(0, 0, 0)
        assert merged_sol.local_coordinates[4] == Coord3D(0, 3, 0)
        assert merged_sol.metadata["child_offsets"]["sc1"] == Coord3D(1, 3, 0)
        assert "corridor_3_4_1" in merged_sol.metadata["all_room_coordinates"]
        assert "corridor_3_4_2" in merged_sol.metadata["all_room_coordinates"]
        assert len(orig_sbs) >= 1

    def test_dynamic_split_helper_edge_cases(self) -> None:
        """Exercises edge-case guards across dynamic ray-split helper functions."""
        # 1. _as_coord3d normalization
        assert _as_coord3d(Coord3D(1, 2, 3)) == Coord3D(1, 2, 3)
        assert _as_coord3d((4, 5, 6)) == Coord3D(4, 5, 6)
        assert _as_coord3d(None) == Coord3D(0, 0, 0)

        # 2. _safe_displacement valid & invalid directions
        assert _safe_displacement(Direction.east) == (1, 0, 0)
        assert _safe_displacement(999) is None

        # 3. _ray_hits_box across all planar directions, z mismatch, and non-planar axis
        orig = Coord3D(5, 5, 0)
        assert _ray_hits_box(orig, (1, 0, 0), (6, 8, 4, 6, 0, 0)) is True
        assert _ray_hits_box(orig, (-1, 0, 0), (1, 3, 4, 6, 0, 0)) is True
        assert _ray_hits_box(orig, (0, 1, 0), (4, 6, 6, 8, 0, 0)) is True
        assert _ray_hits_box(orig, (0, -1, 0), (4, 6, 1, 3, 0, 0)) is True
        assert _ray_hits_box(orig, (1, 0, 0), (6, 8, 4, 6, 1, 2)) is False
        assert _ray_hits_box(orig, (0, 0, 1), (0, 10, 0, 10, 0, 10)) is False

        # 4. _child_attachment_rooms skipping cut=True and matching dst in parent_rooms
        sub_n = SolveTreeNode(node_id="sub_n", node_type=TreeNodeType.BLOCK, rooms=frozenset({50}))
        g_att = LayoutGraph(
            rooms={10: _make_room(10), 11: _make_room(11), 50: _make_room(50)},
            edges=(
                LayoutEdge(src=50, dst=10, direction=Direction.east, cut=True),
                LayoutEdge(src=50, dst=11, direction=Direction.west, cut=False),
            ),
        )
        assert _child_attachment_rooms(sub_n, frozenset({10, 11}), g_att) == {11}

        # 4b. _has_outward_exit_in_direction via reverse edge direction (dst == anchor_vnum)
        g_rev = LayoutGraph(
            rooms={1: _make_room(1), 2: _make_room(2), 3: _make_room(3)},
            edges=(LayoutEdge(src=1, dst=2, direction=Direction.east, cut=False),),
        )
        assert _has_outward_exit_in_direction(2, (-1, 0, 0), {3}, g_rev) is True
        g_rev_bad = LayoutGraph(
            rooms={1: _make_room(1), 2: _make_room(2), 3: _make_room(3)},
            edges=(LayoutEdge(src=1, dst=2, direction=999, cut=False),),
        )
        assert _has_outward_exit_in_direction(2, (-1, 0, 0), {3}, g_rev_bad) is False

        # 5. _find_bridge_ray_blockers negative guards
        assert (
            _find_bridge_ray_blockers(
                SolveTreeNode(
                    node_id="bad_anchor",
                    node_type=TreeNodeType.BLOCK,
                    rooms=frozenset({3}),
                    metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 99)},
                ),
                NodeSolution("bad_anchor", {3: Coord3D(0, 0, 0)}),
                g_rev,
            )
            is None
        )
        assert (
            _find_bridge_ray_blockers(
                SolveTreeNode(
                    node_id="bad_dir",
                    node_type=TreeNodeType.BLOCK,
                    rooms=frozenset({3}),
                    metadata={"cut_contract": BridgeCut((2, 3), 999, 1, 2)},
                ),
                NodeSolution("bad_dir", {3: Coord3D(0, 0, 0)}),
                g_rev,
            )
            is None
        )
        assert (
            _find_bridge_ray_blockers(
                SolveTreeNode(
                    node_id="missing_origin",
                    node_type=TreeNodeType.BLOCK,
                    rooms=frozenset({3}),
                    metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
                ),
                NodeSolution("missing_origin", {3: Coord3D(0, 0, 0)}),
                g_rev,
            )
            is None
        )

        # 6. _try_split_ray_blocked_child: child assignment to base & wing, zero-blocker bridge skip, disconnected edge skip
        sc_base = SolveTreeNode(node_id="sc_base", node_type=TreeNodeType.BLOCK, rooms=frozenset({80}))
        sc_wing = SolveTreeNode(node_id="sc_wing", node_type=TreeNodeType.BLOCK, rooms=frozenset({81}))
        edges_split_edge_cases = (
            LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
            LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
            LayoutEdge(src=3, dst=80, direction=Direction.south, weight=1),
            LayoutEdge(src=3, dst=10, direction=Direction.south, weight=1),  # orthogonal bridge separating zero blockers
            LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),   # orthogonal bridge separating blocker 6
            LayoutEdge(src=4, dst=5, direction=Direction.west, weight=2),
            LayoutEdge(src=5, dst=6, direction=Direction.south, weight=1),
            LayoutEdge(src=6, dst=81, direction=Direction.west, weight=1),
            LayoutEdge(src=90, dst=91, direction=Direction.north, weight=1),  # disconnected internal edge
            LayoutEdge(src=3, dst=92, direction=999, weight=1),  # unsupported direction skipped cleanly
        )
        g_split_ec = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 3, 4, 5, 6, 10, 80, 81, 90, 91, 92)},
            edges=edges_split_edge_cases,
        )
        ch_ec = SolveTreeNode(
            node_id="ch_ec",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 5, 6, 10, 90, 91}),
            children=(sc_base, sc_wing),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        sol_ec = NodeSolution(
            "ch_ec",
            {
                3: Coord3D(3, 1, 0),
                4: Coord3D(3, 2, 0),
                5: Coord3D(1, 2, 0),
                6: Coord3D(1, 1, 0),
                10: Coord3D(3, 0, 0),
            },
            port_coordinates={2: Coord3D(2, 1, 0)},
        )
        split_res = _try_split_ray_blocked_child(ch_ec, sol_ec, g_split_ec)
        assert split_res is not None
        b_node, w_node = split_res
        assert b_node.children == (sc_base,)
        assert w_node.children == (sc_wing,)

        # 7. _extract_status variants
        assert _extract_status(NodeSolution("s", {}, metadata={"status": SolverStatus.OPTIMAL})) == SolverStatus.OPTIMAL
        assert _extract_status(NodeSolution("s", {}, metadata={"status": "timeout"})) == SolverStatus.TIMEOUT
        assert _extract_status(NodeSolution("s", {}, metadata={"status": "unknown"})) == SolverStatus.FEASIBLE
        assert _extract_status(NodeSolution("s", {}, metadata={"status": 123})) == SolverStatus.FEASIBLE

        # 8. _resynthesize_split_child_solution empty pieces, timeout/feasible statuses, and port-only fallback
        orig_empty = SolveTreeNode(node_id="c_emp", node_type=TreeNodeType.BLOCK, rooms=frozenset())
        p_emp_sol = NodeSolution("root", {}, metadata={"status": SolverStatus.OPTIMAL})
        ms_emp, off_emp, sb_emp = _resynthesize_split_child_solution(orig_empty, [], p_emp_sol, {})
        assert off_emp == Coord3D(0, 0, 0)
        assert sb_emp == ()
        assert ms_emp.metadata["status"] == SolverStatus.OPTIMAL

        orig_port_only = SolveTreeNode(
            node_id="c_po",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({2}),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        p_po = SolveTreeNode("c_po__base", TreeNodeType.BLOCK, frozenset({2}))
        s_po = NodeSolution("c_po__base", {2: Coord3D(0, 0, 0)}, port_coordinates={2: Coord3D(0, 0, 0)}, metadata={"status": SolverStatus.TIMEOUT})
        ms_to, _off_to, sb_to = _resynthesize_split_child_solution(orig_port_only, [(p_po, s_po)], p_emp_sol, {})
        assert ms_to.metadata["status"] == SolverStatus.TIMEOUT
        assert len(sb_to) == 1

        s_po_feas = NodeSolution("c_po__base", {2: Coord3D(0, 0, 0)}, metadata={"status": SolverStatus.FEASIBLE})
        ms_feas, _, _ = _resynthesize_split_child_solution(orig_port_only, [(p_po, s_po_feas)], p_emp_sol, {})
        assert ms_feas.metadata["status"] == SolverStatus.FEASIBLE

        # 9. Respect _MAX_DYNAMIC_BRIDGE_SPLIT_DEPTH = 0 cap
        r1 = _make_room(1, {Direction.east: 2})
        r2 = _make_room(2, {Direction.west: 1, Direction.east: 3})
        r3 = _make_room(3, {Direction.west: 2, Direction.north: 4})
        r4 = _make_room(4, {Direction.south: 3, Direction.east: 40, Direction.west: 5})
        r40 = _make_room(40, {Direction.west: 4, Direction.south: 8})
        r8 = _make_room(8, {Direction.north: 40, Direction.west: 7})
        r7 = _make_room(7, {Direction.east: 8, Direction.north: 6})
        r6 = _make_room(6, {Direction.south: 7, Direction.north: 5})
        r5 = _make_room(5, {Direction.south: 6, Direction.east: 4})
        g_cap = LayoutGraph(
            rooms={r.vnum: r for r in (r1, r2, r3, r4, r40, r8, r7, r6, r5)},
            edges=(
                LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
                LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
                LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
                LayoutEdge(src=4, dst=40, direction=Direction.east, weight=1),
                LayoutEdge(src=40, dst=8, direction=Direction.south, weight=2),
                LayoutEdge(src=8, dst=7, direction=Direction.west, weight=4),
                LayoutEdge(src=7, dst=6, direction=Direction.north, weight=1),
                LayoutEdge(src=6, dst=5, direction=Direction.north, weight=1),
                LayoutEdge(src=5, dst=4, direction=Direction.east, weight=3),
            ),
        )
        ch_cap = SolveTreeNode(
            node_id="ch_cap",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 40, 8, 7, 6, 5}),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        rt_cap = SolveTreeNode(node_id="rt_cap", node_type=TreeNodeType.BLOCK, rooms=frozenset({1, 2}), children=(ch_cap,))
        with patch("romutil.layout.milp.tree._MAX_DYNAMIC_BRIDGE_SPLIT_DEPTH", 0):
            sol_cap = HierarchicalTreeSolver().solve_tree(SolveTree(root=rt_cap, graph=g_cap), LayoutOptions(timeout_seconds=15))
            assert set(sol_cap.node_solutions.keys()) == {"ch_cap", "rt_cap"}

    def test_try_split_ray_blocked_child_groups_connected_sibling_subchildren(self) -> None:
        """Connected sibling sub-children (such as intermediate _b split pieces) stay grouped on the same side."""
        # Parent room 1 -> East -> 2 -> East -> 3 (child entry).
        # Blocked ray from anchor 2 is West (-X). Blocker 6 sits at x=1, y=1.
        # Wing sub-child sc_wing_a attaches to 6 (on side_other), while sc_wing_b only connects to sc_wing_a
        # (zero direct attachment to ch.rooms), with severed_bridge_edges=((81, 82),).
        sc_wing_a = SolveTreeNode(
            node_id="sc_wing_a",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({81}),
            metadata={
                "cut_contract": BridgeCut((6, 81), int(Direction.west), 1, 6),
                "severed_bridge_edges": ((81, 82),),
            },
        )
        sc_wing_b = SolveTreeNode(
            node_id="sc_wing_b",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({82}),
            metadata={"severed_bridge_edges": ((81, 82),)},
        )
        g_grp = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 3, 4, 5, 6, 81, 82)},
            edges=(
                LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
                LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
                LayoutEdge(src=3, dst=4, direction=Direction.north, weight=1),
                LayoutEdge(src=4, dst=5, direction=Direction.west, weight=2),
                LayoutEdge(src=5, dst=6, direction=Direction.south, weight=1),
                LayoutEdge(src=6, dst=81, direction=Direction.west, weight=1),
                LayoutEdge(src=81, dst=82, direction=Direction.south, weight=1),
            ),
        )
        ch_grp = SolveTreeNode(
            node_id="ch_grp",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({3, 4, 5, 6}),
            children=(sc_wing_a, sc_wing_b),
            metadata={"cut_contract": BridgeCut((2, 3), int(Direction.east), 1, 2)},
        )
        sol_grp = NodeSolution(
            "ch_grp",
            {
                3: Coord3D(3, 1, 0),
                4: Coord3D(3, 2, 0),
                5: Coord3D(1, 2, 0),
                6: Coord3D(1, 1, 0),
            },
            port_coordinates={2: Coord3D(2, 1, 0)},
        )
        split_res = _try_split_ray_blocked_child(ch_grp, sol_grp, g_grp)
        assert split_res is not None
        b_node, w_node = split_res
        assert b_node.children == ()
        assert {c.node_id for c in w_node.children} == {"sc_wing_a", "sc_wing_b"}
        assert w_node.metadata.get("severed_bridge_edges") == ((81, 82),)

        # If a bridge connects sc_wing_b back to room 3 (on side_v), the connected group straddles
        # the split boundary and prevents separating across (3, 4) -> split returns None.
        g_straddle = LayoutGraph(
            rooms=g_grp.rooms,
            edges=g_grp.edges + (LayoutEdge(src=3, dst=82, direction=Direction.south, weight=1),),
        )
        assert _try_split_ray_blocked_child(ch_grp, sol_grp, g_straddle) is None

    def test_detaches_articulation_ray_blocking_subchild_end_to_end(self) -> None:
        """Detaches a BridgeCut sub-child inside an ArticulationPointCut loop blocking an outward ray."""
        r106 = _make_room(106, {Direction.east: 120, Direction.west: 112, Direction.south: 105})
        r120 = _make_room(120, {Direction.west: 106})
        r112 = _make_room(112, {Direction.east: 106, Direction.south: 111})
        r111 = _make_room(111, {Direction.north: 112, Direction.east: 105})
        r105 = _make_room(105, {Direction.west: 111, Direction.north: 106, Direction.east: 110})
        r110 = _make_room(110, {Direction.west: 105, Direction.north: 119})
        r119 = _make_room(119, {Direction.south: 110})

        edges = (
            LayoutEdge(src=106, dst=120, direction=Direction.east, weight=1),
            LayoutEdge(src=106, dst=112, direction=Direction.west, weight=1),
            LayoutEdge(src=112, dst=111, direction=Direction.south, weight=1),
            LayoutEdge(src=111, dst=105, direction=Direction.east, weight=1),
            LayoutEdge(src=105, dst=106, direction=Direction.north, weight=1),
            LayoutEdge(src=105, dst=110, direction=Direction.east, weight=1),
            LayoutEdge(src=110, dst=119, direction=Direction.north, weight=1),
        )
        graph = LayoutGraph(
            rooms={r.vnum: r for r in (r106, r120, r112, r111, r105, r110, r119)},
            edges=edges,
        )

        sc = SolveTreeNode(
            node_id="sc_wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({110, 119}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 105,
                "bridge_edge": (105, 110),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((105, 110), int(Direction.east), 1, 105),
            },
        )
        ap_child = SolveTreeNode(
            node_id="ap_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({105, 111, 112}),
            children=(sc,),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 106,
                "cut_contract": ArticulationPointCut(anchor_vnum=106),
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({106, 120}),
            children=(ap_child,),
        )
        tree = SolveTree(root=root, components=(root,), graph=graph)

        sol = HierarchicalTreeSolver().solve_tree(tree, options=LayoutOptions(timeout_seconds=15))

        assert set(sol.node_solutions.keys()) == {"sc_wing", "ap_loop", "root"}
        assert sol.node_solutions["root"].metadata["status"] == SolverStatus.OPTIMAL
        assert set(sol.node_solutions["root"].metadata["child_offsets"].keys()) == {"ap_loop"}
        assert "sc_wing" in sol.node_solutions["ap_loop"].metadata["child_offsets"]
        assert "corridor_105_110_1" in sol.node_solutions["ap_loop"].metadata["all_room_coordinates"]

        assembled = HierarchicalLayoutAssembler().assemble(
            tree=tree,
            solution=sol,
            simplified=SimplifiedGraph(graph=graph),
        )
        assert_zero_coordinate_collisions(assembled)
        assert assembled.coordinates[120].x == assembled.coordinates[106].x + 1
        assert assembled.coordinates[120].y == assembled.coordinates[106].y
        assert assembled.coordinates[110].x >= assembled.coordinates[105].x + 2

    def test_try_detach_articulation_ray_blocking_subchild_guards(self) -> None:
        """Negative unit guards verify cut warps, sibling-connected sub-children, and non-blockers do not detach."""
        sc = SolveTreeNode(
            node_id="sc_wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({110, 119}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 105,
                "bridge_edge": (105, 110),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((105, 110), int(Direction.east), 1, 105),
            },
        )
        ap_child = SolveTreeNode(
            node_id="ap_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({105, 111, 112}),
            children=(sc,),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=106)},
        )
        ap_sol = NodeSolution(
            node_id="ap_loop",
            local_coordinates={
                105: Coord3D(1, 0, 0),
                111: Coord3D(0, 0, 0),
                112: Coord3D(0, 1, 0),
            },
            port_coordinates={106: Coord3D(1, 1, 0)},
            metadata={
                "child_offsets": {"sc_wing": Coord3D(2, 0, 0)},
                "child_sub_boxes": {"sc_wing": (BoundingBox3D(0, 0, 0, 1, 0, 0),)},
            },
        )
        base_rooms = {
            v: _make_room(v)
            for v in (105, 106, 110, 111, 112, 119, 120)
        }

        # 1. Cut warp outward exit from anchor 106 -> must NOT trigger detachment
        g_cut_warp = LayoutGraph(
            rooms=base_rooms,
            edges=(
                LayoutEdge(src=106, dst=120, direction=Direction.east, weight=1, cut=True),
                LayoutEdge(src=105, dst=110, direction=Direction.east, weight=1),
            ),
        )
        assert _try_detach_articulation_ray_blocking_subchild(ap_child, ap_sol, g_cut_warp) is None

        # 2. Sub-child connected to another sibling sub-child -> must NOT detach
        sc_sib = SolveTreeNode(
            node_id="sc_sib",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({130}),
            children=(),
            metadata={"cut_contract": BridgeCut((111, 130), int(Direction.south), 1, 111)},
        )
        ap_two_sibs = SolveTreeNode(
            node_id="ap_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({105, 111, 112}),
            children=(sc, sc_sib),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=106)},
        )
        g_sib_linked = LayoutGraph(
            rooms={**base_rooms, 130: _make_room(130)},
            edges=(
                LayoutEdge(src=106, dst=120, direction=Direction.east, weight=1),
                LayoutEdge(src=110, dst=130, direction=Direction.south, weight=1),
            ),
        )
        assert _try_detach_articulation_ray_blocking_subchild(ap_two_sibs, ap_sol, g_sib_linked) is None

        # 3. Non-blocking sub-child box (y=2, outside anchor y=1) -> must NOT detach
        ap_non_blocking_sol = NodeSolution(
            node_id="ap_loop",
            local_coordinates=ap_sol.local_coordinates,
            port_coordinates=ap_sol.port_coordinates,
            metadata={
                "child_offsets": {"sc_wing": Coord3D(2, 2, 0)},
                "child_sub_boxes": {"sc_wing": (BoundingBox3D(0, 0, 0, 1, 0, 0),)},
            },
        )
        g_east = LayoutGraph(
            rooms=base_rooms,
            edges=(LayoutEdge(src=106, dst=120, direction=Direction.east, weight=1),),
        )
        assert _try_detach_articulation_ray_blocking_subchild(ap_child, ap_non_blocking_sol, g_east) is None

    def test_ray_continues_beyond_neighbor_and_parallel_bridge_guard(self) -> None:
        """_ray_continues_beyond_neighbor detects continuing rays and guards parallel-bridge detachment."""
        base_rooms = {v: _make_room(v) for v in (105, 106, 110, 111, 112, 119, 120, 121)}
        subtree = {105, 106, 110, 111, 112, 119}

        # Forward continuing ray: 106 -(E)-> 120 -(E)-> 121
        g_fwd = LayoutGraph(
            rooms=base_rooms,
            edges=(
                LayoutEdge(src=106, dst=120, direction=Direction.east, weight=1),
                LayoutEdge(src=120, dst=121, direction=Direction.east, weight=1),
            ),
        )
        assert _ray_continues_beyond_neighbor(106, (1, 0, 0), subtree, g_fwd) is True
        assert _ray_continues_beyond_neighbor(106, (0, 1, 0), subtree, g_fwd) is False

        # Reverse edge orientation continuing ray: 121 -(W)-> 120 -(W)-> 106
        g_rev = LayoutGraph(
            rooms=base_rooms,
            edges=(
                LayoutEdge(src=120, dst=106, direction=Direction.west, weight=1),
                LayoutEdge(src=121, dst=120, direction=Direction.west, weight=1),
            ),
        )
        assert _ray_continues_beyond_neighbor(106, (1, 0, 0), subtree, g_rev) is True

        # Cut continuation edge does not count as continuing ray
        g_cut_cont = LayoutGraph(
            rooms=base_rooms,
            edges=(
                LayoutEdge(src=106, dst=120, direction=Direction.east, weight=1),
                LayoutEdge(src=120, dst=121, direction=Direction.east, weight=1, cut=True),
            ),
        )
        assert _ray_continues_beyond_neighbor(106, (1, 0, 0), subtree, g_cut_cont) is False

        # Parallel East BridgeCut sub-child must NOT detach when East ray continues past 120 -> 121
        sc_parallel = SolveTreeNode(
            node_id="sc_wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({110, 119}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 105,
                "bridge_edge": (105, 110),
                "direction": int(Direction.east),
                "weight": 1,
                "cut_contract": BridgeCut((105, 110), int(Direction.east), 1, 105),
            },
        )
        ap_child = SolveTreeNode(
            node_id="ap_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({105, 111, 112}),
            children=(sc_parallel,),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=106)},
        )
        ap_sol = NodeSolution(
            node_id="ap_loop",
            local_coordinates={
                105: Coord3D(1, 0, 0),
                111: Coord3D(0, 0, 0),
                112: Coord3D(0, 1, 0),
            },
            port_coordinates={106: Coord3D(1, 1, 0)},
            metadata={
                "child_offsets": {"sc_wing": Coord3D(2, 1, 0)},
                "child_sub_boxes": {"sc_wing": (BoundingBox3D(0, 0, 0, 1, 0, 0),)},
            },
        )
        assert _try_detach_articulation_ray_blocking_subchild(ap_child, ap_sol, g_fwd) is None

        # Perpendicular (North) BridgeCut sub-child blocking the same continuing East ray DOES detach
        sc_perp = SolveTreeNode(
            node_id="sc_wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({110, 119}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 105,
                "bridge_edge": (105, 110),
                "direction": int(Direction.north),
                "weight": 1,
                "cut_contract": BridgeCut((105, 110), int(Direction.north), 1, 105),
            },
        )
        ap_perp = SolveTreeNode(
            node_id="ap_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({105, 111, 112}),
            children=(sc_perp,),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=106)},
        )
        assert _try_detach_articulation_ray_blocking_subchild(ap_perp, ap_sol, g_fwd) is not None

    def test_resolve_articulation_ray_blockage_clears_continuing_north_ray_with_zero_shear(self) -> None:
        """Open-ray re-solve shifts an ArticulationPointCut loop laterally off a continuing North ray with zero shear."""
        rooms = {
            106: _make_room(106, {Direction.north: 120, Direction.east: 112, Direction.south: 118}),
            120: _make_room(120, {Direction.south: 106, Direction.north: 121}),
            121: _make_room(121, {Direction.south: 120}),
            112: _make_room(112, {Direction.west: 106, Direction.north: 111}),
            111: _make_room(111, {Direction.south: 112, Direction.north: 114}),
            114: _make_room(114, {Direction.south: 111, Direction.east: 115, Direction.north: 130}),
            115: _make_room(115, {Direction.west: 114, Direction.south: 116}),
            116: _make_room(116, {Direction.north: 115, Direction.south: 117}),
            117: _make_room(117, {Direction.north: 116, Direction.west: 118}),
            118: _make_room(118, {Direction.east: 117, Direction.north: 106}),
            130: _make_room(130, {Direction.south: 114, Direction.west: 131}),
            131: _make_room(131, {Direction.east: 130, Direction.north: 132}),
            132: _make_room(132, {Direction.south: 131}),
        }
        edges = (
            LayoutEdge(src=106, dst=120, direction=Direction.north, weight=1),
            LayoutEdge(src=120, dst=121, direction=Direction.north, weight=1),
            LayoutEdge(src=106, dst=112, direction=Direction.east, weight=1),
            LayoutEdge(src=112, dst=111, direction=Direction.north, weight=1),
            LayoutEdge(src=111, dst=114, direction=Direction.north, weight=1),
            LayoutEdge(src=114, dst=115, direction=Direction.east, weight=1),
            LayoutEdge(src=115, dst=116, direction=Direction.south, weight=1),
            LayoutEdge(src=116, dst=117, direction=Direction.south, weight=1),
            LayoutEdge(src=117, dst=118, direction=Direction.west, weight=1),
            LayoutEdge(src=118, dst=106, direction=Direction.north, weight=1),
            LayoutEdge(src=114, dst=130, direction=Direction.north, weight=1),
            LayoutEdge(src=130, dst=131, direction=Direction.west, weight=1),
            LayoutEdge(src=131, dst=132, direction=Direction.north, weight=1),
        )
        graph = LayoutGraph(rooms=rooms, edges=edges)
        sc = SolveTreeNode(
            node_id="sc_wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({130, 131, 132}),
            children=(),
            metadata={
                "cut_type": CutType.BRIDGE,
                "anchor_vnum": 114,
                "bridge_edge": (114, 130),
                "direction": int(Direction.north),
                "weight": 1,
                "cut_contract": BridgeCut((114, 130), int(Direction.north), 1, 114),
            },
        )
        ap_loop = SolveTreeNode(
            node_id="ap_loop",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({111, 112, 114, 115, 116, 117, 118}),
            children=(sc,),
            metadata={
                "cut_type": CutType.ARTICULATION_POINT,
                "anchor_vnum": 106,
                "cut_contract": ArticulationPointCut(anchor_vnum=106),
            },
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({106, 120, 121}),
            children=(ap_loop,),
        )
        tree = SolveTree(root=root, components=(root,), graph=graph)

        sol = HierarchicalTreeSolver().solve_tree(tree, options=LayoutOptions(timeout_seconds=15))
        # Parallel-North sub-child was NOT detached; ap_loop cleared the North ray internally
        assert set(sol.node_solutions.keys()) == {"sc_wing", "ap_loop", "root"}
        assert set(sol.node_solutions["root"].metadata["child_offsets"].keys()) == {"ap_loop"}

        assembled = HierarchicalLayoutAssembler().assemble(
            tree=tree,
            solution=sol,
            simplified=SimplifiedGraph(graph=graph),
        )
        assert_zero_coordinate_collisions(assembled)
        c = assembled.coordinates
        # Continuing North ray keeps zero transverse shear along X and Z
        assert c[120].x == c[106].x and c[120].z == c[106].z and c[120].y > c[106].y
        assert c[121].x == c[120].x and c[121].z == c[120].z and c[121].y > c[120].y
        # No room in ap_loop or sc_wing obstructs the North ray of anchor 106
        for vnum in (111, 112, 114, 115, 116, 117, 118, 130, 131, 132):
            assert not (c[vnum].x == c[106].x and c[vnum].y > c[106].y and c[vnum].z == c[106].z)

    def test_ray_hits_box_with_extent_and_blockage_solver_error_fallback(self) -> None:
        """Exercises _ray_hits_box_with_extent bounds and safe LayoutSolverError fallback in ray resolution."""
        orig = Coord3D(10, 10, 0)
        ext = (1, 1)  # closed anchor square [9..11, 9..11]
        assert _ray_hits_box_with_extent(orig, ext, (1, 0, 0), (12, 13, 11, 12, 0, 0)) is True
        assert _ray_hits_box_with_extent(orig, ext, (1, 0, 0), (11, 11, 10, 10, 0, 0)) is False
        assert _ray_hits_box_with_extent(orig, ext, (-1, 0, 0), (6, 8, 8, 9, 0, 0)) is True
        assert _ray_hits_box_with_extent(orig, ext, (0, 1, 0), (9, 10, 12, 14, 0, 0)) is True
        assert _ray_hits_box_with_extent(orig, ext, (0, -1, 0), (11, 12, 5, 8, 0, 0)) is True
        assert _ray_hits_box_with_extent(orig, ext, (1, 0, 0), (12, 13, 10, 10, 1, 1)) is False
        assert _ray_hits_box_with_extent(orig, ext, (0, 0, 1), (0, 20, 0, 20, 0, 20)) is False

        # Non-ArticulationPointCut and missing-origin guards
        non_ap = SolveTreeNode(node_id="non_ap", node_type=TreeNodeType.BLOCK, rooms=frozenset({1}))
        assert _try_resolve_articulation_ray_blockage(non_ap, NodeSolution("non_ap", {1: Coord3D(0, 0, 0)}), {}, LayoutGraph(rooms={1: _make_room(1)}, edges=()), None) is None

        ap_node = SolveTreeNode(
            node_id="ap_err",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({2, 3, 4}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        g_blocked = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 3, 4, 5)},
            edges=(
                LayoutEdge(src=1, dst=5, direction=Direction.north, weight=1),
                LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
                LayoutEdge(src=2, dst=3, direction=Direction.north, weight=1),
                LayoutEdge(src=3, dst=4, direction=Direction.west, weight=1),
                LayoutEdge(src=4, dst=1, direction=Direction.south, weight=1),
            ),
        )
        sol_blocked = NodeSolution(
            "ap_err",
            {2: Coord3D(1, 0, 0), 3: Coord3D(1, 1, 0), 4: Coord3D(0, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0)},
        )
        with patch(
            "romutil.layout.milp.tree.solve_cluster_with_cutting_planes",
            side_effect=LayoutSolverError("Simulated timeout during ray clear"),
        ):
            assert (
                _try_resolve_articulation_ray_blockage(
                    ap_node, sol_blocked, {}, g_blocked, LayoutOptions(timeout_seconds=5)
                )
                is None
            )

        # Unblocked layout returns None without invoking solver
        sol_clear = NodeSolution(
            "ap_err",
            {2: Coord3D(1, 1, 0), 3: Coord3D(1, 0, 0), 4: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, 1, 0)},
        )
        assert (
            _try_resolve_articulation_ray_blockage(
                ap_node, sol_clear, {}, g_blocked, LayoutOptions(timeout_seconds=5)
            )
            is None
        )

    def test_resolve_articulation_ray_blockage_east_west_envelope_and_split_hook(self) -> None:
        """Exercises East/West open-ray separation, corridor envelope calls, and split-map ray hook."""
        from romutil.layout.ir import SplitPairCut
        from romutil.layout.milp.formulation import ClusterFormulation

        # 1. East-blocked ray (1 -> 6 East) cleared by shifting elastic split-pair child West
        g_east = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 3, 4, 5, 6, 11, 20)},
            edges=(
                LayoutEdge(src=1, dst=6, direction=Direction.east, weight=1),
                LayoutEdge(src=1, dst=2, direction=Direction.north, weight=1),
                LayoutEdge(src=2, dst=3, direction=Direction.west, weight=1),
                LayoutEdge(src=3, dst=4, direction=Direction.south, weight=1),
                LayoutEdge(src=4, dst=5, direction=Direction.south, weight=1),
                LayoutEdge(src=5, dst=11, direction=Direction.east, weight=1),
                LayoutEdge(src=11, dst=1, direction=Direction.north, weight=1),
                LayoutEdge(src=3, dst=20, direction=Direction.east, weight=1),
                LayoutEdge(src=20, dst=4, direction=Direction.south, weight=1),
            ),
        )
        sp_e = SolveTreeNode(
            node_id="sp_e",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({20}),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": (3, 4), "cut_contract": SplitPairCut((3, 4))},
        )
        ap_e = SolveTreeNode(
            node_id="ap_e",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({2, 3, 4, 5, 11}),
            children=(sp_e,),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        sp_e_sol = NodeSolution(
            "sp_e",
            {20: Coord3D(2, 1, 0)},
            port_coordinates={3: Coord3D(0, 2, 0), 4: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(1, 2, 1, 1, 0, 0),),
        )
        blocked_e_sol = NodeSolution(
            "ap_e",
            {2: Coord3D(2, 3, 0), 3: Coord3D(1, 3, 0), 4: Coord3D(1, 1, 0), 5: Coord3D(1, 0, 0), 11: Coord3D(2, 0, 0)},
            port_coordinates={1: Coord3D(2, 2, 0)},
            metadata={
                "child_offsets": {"sp_e": Coord3D(1, 1, 0)},
                "child_sub_boxes": {"sp_e": (BoundingBox3D(1, 2, 1, 1, 0, 0),)},
            },
        )
        envelope_calls: list[bool] = []
        orig_envelope = ClusterFormulation.get_corridor_envelope_box

        def spy_envelope(self_form: ClusterFormulation, ae: object, *, interior: bool = True) -> tuple[object, ...]:
            envelope_calls.append(interior)
            return orig_envelope(self_form, ae, interior=interior)  # type: ignore[arg-type]

        with patch.object(ClusterFormulation, "get_corridor_envelope_box", autospec=True, side_effect=spy_envelope):
            resolved_e = _try_resolve_articulation_ray_blockage(
                ap_e, blocked_e_sol, {"sp_e": sp_e_sol}, g_east, LayoutOptions(timeout_seconds=10)
            )

        assert resolved_e is not None
        assert False in envelope_calls
        anchor_e = resolved_e.port_coordinates[1]
        sp_e_off = resolved_e.metadata["child_offsets"]["sp_e"]
        for sb in resolved_e.metadata["child_sub_boxes"]["sp_e"]:
            assert not (sp_e_off.x + sb.max_x > anchor_e.x and sp_e_off.y + sb.min_y <= anchor_e.y <= sp_e_off.y + sb.max_y)

        # 2. West-blocked ray (1 -> 7 West) cleared by shifting elastic split-pair child East
        g_west = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 3, 4, 5, 7, 11, 20)},
            edges=(
                LayoutEdge(src=1, dst=7, direction=Direction.west, weight=1),
                LayoutEdge(src=1, dst=2, direction=Direction.north, weight=1),
                LayoutEdge(src=2, dst=3, direction=Direction.east, weight=1),
                LayoutEdge(src=3, dst=4, direction=Direction.south, weight=1),
                LayoutEdge(src=4, dst=5, direction=Direction.south, weight=1),
                LayoutEdge(src=5, dst=11, direction=Direction.west, weight=1),
                LayoutEdge(src=11, dst=1, direction=Direction.north, weight=1),
                LayoutEdge(src=3, dst=20, direction=Direction.west, weight=1),
                LayoutEdge(src=20, dst=4, direction=Direction.south, weight=1),
            ),
        )
        sp_w = SolveTreeNode(
            node_id="sp_w",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({20}),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": (3, 4), "cut_contract": SplitPairCut((3, 4))},
        )
        ap_w = SolveTreeNode(
            node_id="ap_w",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({2, 3, 4, 5, 11}),
            children=(sp_w,),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        sp_w_sol = NodeSolution(
            "sp_w",
            {20: Coord3D(0, 1, 0)},
            port_coordinates={3: Coord3D(2, 2, 0), 4: Coord3D(2, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 1, 1, 0, 0),),
        )
        blocked_w_sol = NodeSolution(
            "ap_w",
            {2: Coord3D(2, 3, 0), 3: Coord3D(3, 3, 0), 4: Coord3D(3, 1, 0), 5: Coord3D(3, 0, 0), 11: Coord3D(2, 0, 0)},
            port_coordinates={1: Coord3D(2, 2, 0)},
            metadata={
                "child_offsets": {"sp_w": Coord3D(1, 1, 0)},
                "child_sub_boxes": {"sp_w": (BoundingBox3D(0, 1, 1, 1, 0, 0),)},
            },
        )
        resolved_w = _try_resolve_articulation_ray_blockage(
            ap_w, blocked_w_sol, {"sp_w": sp_w_sol}, g_west, LayoutOptions(timeout_seconds=10)
        )
        assert resolved_w is not None
        anchor_w = resolved_w.port_coordinates[1]
        sp_w_off = resolved_w.metadata["child_offsets"]["sp_w"]
        for sb in resolved_w.metadata["child_sub_boxes"]["sp_w"]:
            assert not (sp_w_off.x + sb.min_x < anchor_w.x and sp_w_off.y + sb.min_y <= anchor_w.y <= sp_w_off.y + sb.max_y)

        # 3. Verify split-map (effective_node) path invokes _try_resolve_articulation_ray_blockage
        wing_split_child = SolveTreeNode(
            node_id="br_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({20}),
            metadata={"cut_contract": BridgeCut((2, 20), int(Direction.east), 1, 2)},
        )
        ap_split = SolveTreeNode(
            node_id="ap_split",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({2, 3, 4}),
            children=(wing_split_child,),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        g_split = LayoutGraph(
            rooms={v: _make_room(v) for v in (1, 2, 3, 4, 20)},
            edges=(
                LayoutEdge(src=1, dst=2, direction=Direction.east, weight=1),
                LayoutEdge(src=2, dst=3, direction=Direction.north, weight=1),
                LayoutEdge(src=3, dst=4, direction=Direction.west, weight=1),
                LayoutEdge(src=4, dst=1, direction=Direction.south, weight=1),
                LayoutEdge(src=2, dst=20, direction=Direction.east, weight=1),
            ),
        )
        tree_split = SolveTree(root=ap_split, components=(ap_split,), graph=g_split)
        split_base = SolveTreeNode("br_child__base", TreeNodeType.BLOCK, frozenset({20}))
        split_wing = SolveTreeNode(
            "br_child__wing",
            TreeNodeType.BLOCK,
            frozenset({20}),
            metadata={"cut_contract": BridgeCut((2, 20), int(Direction.east), 1, 2)},
        )

        with (
            patch(
                "romutil.layout.milp.tree._try_split_ray_blocked_child",
                side_effect=[(split_base, split_wing), None, None],
            ),
            patch("romutil.layout.milp.tree._try_resolve_articulation_ray_blockage", return_value=None) as mock_resolve,
            patch(
                "romutil.layout.milp.tree._resynthesize_split_child_solution",
                return_value=(NodeSolution("br_child", {20: Coord3D(0, 0, 0)}), Coord3D(0, 0, 0), ()),
            ),
            patch(
                "romutil.layout.milp.tree.solve_cluster",
                side_effect=[
                    NodeSolution("br_child", {20: Coord3D(0, 0, 0)}),
                    NodeSolution("br_child__base", {20: Coord3D(0, 0, 0)}),
                    NodeSolution("br_child__wing", {20: Coord3D(0, 0, 0)}),
                    NodeSolution("ap_split", {2: Coord3D(1, 0, 0), 3: Coord3D(1, 1, 0), 4: Coord3D(0, 1, 0)}, port_coordinates={1: Coord3D(0, 0, 0)}),
                ],
            ),
        ):
            HierarchicalTreeSolver().solve_tree(tree_split, LayoutOptions(timeout_seconds=10))

        mock_resolve.assert_called_once()
        called_node = mock_resolve.call_args.kwargs["node"]
        assert [c.node_id for c in called_node.children] == ["br_child__base", "br_child__wing"]
