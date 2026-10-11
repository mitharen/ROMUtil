"""Unit tests for the modular 5-stage MILP cluster layout pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

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
from romutil.layout.milp import (
    ActiveEdge,
    ClusterFormulation,
    ClusterInventory,
    apply_clearance_constraints,
    formulate_cluster_model,
    inventory_cluster,
    normalize_cluster_solution,
    solve_and_extract,
)
from romutil.layout.milp.inventory import (
    ChildChildAlignment,
    RoomChildAlignment,
    get_child_boundary_ports,
)
from romutil.layout.milp.normalization import (
    normalize_empty_cluster,
    normalize_single_room_cluster,
)
from romutil.layout.milp.solver import (
    SolvedClusterExtraction,
    decompose_cluster_sub_boxes,
)
from romutil.models import Direction, Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus


def _make_graph(
    rooms: list[int],
    edges: list[tuple[int, int, Direction | int, int]],
    cut_edges: list[tuple[int, int, Direction | int, int]] | None = None,
) -> LayoutGraph:
    """Helper to construct a LayoutGraph."""
    room_defs = {
        v: Room(vnum=v, name=f"Room {v}", description="", exits=())
        for v in rooms
    }
    layout_edges: list[LayoutEdge] = []
    for u, v, d, w in edges:
        layout_edges.append(
            LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=False)
        )
    if cut_edges:
        for u, v, d, w in cut_edges:
            layout_edges.append(
                LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=True)
            )
    return LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))


class TestStage1Inventory:
    """Unit tests for Stage 1: Inventory and Interface Classification."""

    def test_get_child_boundary_ports_variants(self) -> None:
        """Boundary ports are correctly parsed from cut contracts and metadata."""
        node_contract = SolveTreeNode(
            node_id="c_contract",
            node_type="leaf",
            rooms=frozenset({1, 2, 3}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 3))},
        )
        assert get_child_boundary_ports(node_contract) == {1, 3}

        node_meta_enum = SolveTreeNode(
            node_id="c_enum",
            node_type="leaf",
            rooms=frozenset({4, 5}),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": (4, 5)},
        )
        assert get_child_boundary_ports(node_meta_enum) == {4, 5}

        node_meta_str = SolveTreeNode(
            node_id="c_str",
            node_type="leaf",
            rooms=frozenset({6, 7}),
            metadata={"cut_type": "split_pair", "poles": ["6", "7"]},
        )
        assert get_child_boundary_ports(node_meta_str) == {6, 7}

        node_plain = SolveTreeNode(
            node_id="c_plain",
            node_type="leaf",
            rooms=frozenset({8, 9}),
        )
        assert get_child_boundary_ports(node_plain) == set()

    def test_inventory_leaf_cluster(self) -> None:
        """Inventories leaf cluster internal edges and empty alignments."""
        node = SolveTreeNode(node_id="leaf", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)

        assert inv.rooms == (1, 2)
        assert inv.children == ()
        assert len(inv.active_edges) == 1
        ae = inv.active_edges[0]
        assert ae.src == 1
        assert ae.dst == 2
        assert ae.direction == (1, 0, 0)
        assert ae.min_length == 1
        assert not ae.allow_transverse_slack

    def test_inventory_split_pair_transverse_slack(self) -> None:
        """Split-pair boundary pole connections generate transverse slack edges."""
        p0, p1 = 1, 2
        child_rooms = {1, 2, 10}
        child = SolveTreeNode(
            node_id="c_sp",
            node_type="leaf",
            rooms=frozenset(child_rooms),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="parent_sp",
            node_type="container",
            rooms=frozenset({1, 2, 3}),
            children=(child,),
        )
        c_sol = NodeSolution(
            node_id="c_sp",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(4, 0, 0), 10: Coord3D(2, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 4, 0, 0, 0, 0),),
        )
        graph = _make_graph(
            rooms=[1, 2, 3, 10],
            edges=[
                (3, 10, Direction.north, 1),
                (1, 2, Direction.east, 4),
            ],
        )

        inv = inventory_cluster(
            node=parent,
            child_solutions={child.node_id: c_sol},
            graph=graph,
        )

        assert inv.rooms == (1, 2, 3)
        assert len(inv.room_alignments) == 1
        assert inv.room_alignments[0].child_id == "c_sp"
        # 3 -> 10 should be an external transverse active edge
        ext_edges = [e for e in inv.active_edges if e.edge_id.startswith("ext_")]
        assert len(ext_edges) == 1
        assert ext_edges[0].allow_transverse_slack is True
        assert ext_edges[0].src == 3
        assert ext_edges[0].dst == ("c_sp", 10)

    def test_inventory_articulation_point_and_bridge(self) -> None:
        """Articulation point and bridge cut children are correctly classified."""
        c_art = SolveTreeNode(
            node_id="c_art",
            node_type="leaf",
            rooms=frozenset({2, 20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=2)},
        )
        c_brg = SolveTreeNode(
            node_id="c_brg",
            node_type="leaf",
            rooms=frozenset({30}),
            metadata={
                "cut_contract": BridgeCut(
                    bridge_edge=(1, 30),
                    direction=Direction.north,
                    weight=2,
                )
            },
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({1, 2}),
            children=(c_art, c_brg),
        )
        sol_art = NodeSolution(
            node_id="c_art",
            local_coordinates={2: Coord3D(0, 0, 0), 20: Coord3D(1, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 0),),
        )
        sol_brg = NodeSolution(
            node_id="c_brg",
            local_coordinates={30: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )
        graph = _make_graph(
            rooms=[1, 2, 20, 30],
            edges=[(1, 2, Direction.east, 1), (1, 30, Direction.north, 2)],
        )

        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_art": sol_art, "c_brg": sol_brg},
            graph=graph,
        )

        assert any(a.room_vnum == 2 and a.child_id == "c_art" for a in inv.room_alignments)
        gw_edges = [e for e in inv.active_edges if e.edge_id.startswith("gw_")]
        assert len(gw_edges) == 1
        assert gw_edges[0].src == 1
        assert gw_edges[0].dst == ("c_brg", 30)
        assert gw_edges[0].min_length == 2


class TestStage2Formulation:
    """Unit tests for Stage 2: Basic MILP Formulation."""

    def test_formulate_cluster_model_variables_and_anchors(self) -> None:
        """Formulation declares room variables, fixes anchor, and attaches accessors."""
        node = SolveTreeNode(node_id="test_node", node_type="leaf", rooms=frozenset({10, 20}))
        graph = _make_graph([10, 20], [(10, 20, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)

        formulation = formulate_cluster_model(inventory=inv)
        model = formulation.model

        # Coordinates variables exist
        assert hasattr(model, "x")
        assert hasattr(model, "y")
        assert hasattr(model, "z")
        # Anchor is fixed to 0
        assert model.x[10].is_fixed()
        assert model.x[10].value == 0
        assert not model.x[20].is_fixed()

        # Accessors
        pt10 = formulation.get_point(10)
        assert pt10[0] is model.x[10]
        box10 = formulation.get_room_box(10)
        assert box10 == (model.x[10], model.x[10], model.y[10], model.y[10], model.z[10], model.z[10])

        # Objective exists
        assert hasattr(model, "obj")

    def test_formulate_cluster_model_child_only(self) -> None:
        """Formulation with no parent rooms fixes the first child's origin."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({1}))
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(node_id="p_no_rooms", node_type="container", rooms=frozenset(), children=(c1, c2))

        sol1 = NodeSolution(node_id="c1", local_coordinates={1: Coord3D(0, 0, 0)}, sub_boxes=())
        sol2 = NodeSolution(node_id="c2", local_coordinates={2: Coord3D(0, 0, 0)}, sub_boxes=())
        graph = _make_graph([1, 2], [])

        inv = inventory_cluster(node=parent, child_solutions={"c1": sol1, "c2": sol2}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)
        model = formulation.model

        assert hasattr(model, "cx")
        assert model.cx["c1"].is_fixed()
        assert model.cx["c1"].value == 0
        assert not model.cx["c2"].is_fixed()


class TestStage3Clearance:
    """Unit tests for Stage 3: Geometric Clearance & Separation Constraints."""

    def test_apply_clearance_constraints_parent_rooms(self) -> None:
        """Room-room non-overlap separation constraints are attached with prefix 'pr'."""
        node = SolveTreeNode(node_id="p_rooms", node_type="block", rooms=frozenset({1, 2, 3}))
        graph = _make_graph([1, 2, 3], [(1, 2, Direction.east, 1), (2, 3, Direction.north, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        apply_clearance_constraints(formulation=formulation)
        model = formulation.model

        # Parent-room separation variables and constraints
        assert hasattr(model, "pr_br")
        assert hasattr(model, "pr_sep_sum")
        assert len(model.pr_sep_sum) == 3  # (1,2), (1,3), (2,3)

    def test_apply_clearance_constraints_child_and_corridors(self) -> None:
        """Child sub-box and corridor clearance constraints are created."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset({1, 2, 3}), children=(c1,))
        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),),
        )
        graph = _make_graph([1, 2, 3, 10], [(1, 2, Direction.east, 2)])
        inv = inventory_cluster(node=parent, child_solutions={"c1": sol1}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        apply_clearance_constraints(formulation=formulation)
        model = formulation.model

        assert hasattr(model, "rc_br")  # Room vs child separation
        assert hasattr(model, "cwr_br")  # Corridor vs room separation


class TestStage4Solver:
    """Unit tests for Stage 4: Solve and Extraction."""

    def test_decompose_cluster_sub_boxes_leaf(self) -> None:
        """Leaf cluster decomposes non-convex cells into minimal orthogonal boxes."""
        node = SolveTreeNode(node_id="leaf_l", node_type="leaf", rooms=frozenset({1, 2, 3}))
        raw_coords = {
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (0, 1, 0),
        }
        boxes = decompose_cluster_sub_boxes(
            raw_parent_coords=raw_coords,
            raw_child_offsets={},
            node=node,
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
        )
        assert len(boxes) == 2  # L-shape decomposes into 2 disjoint boxes
        total_vol = sum(b.volume() for b in boxes)
        assert total_vol == 3

    def test_decompose_cluster_sub_boxes_composite(self) -> None:
        """Composite cluster computes enclosing bounding box over translated sub-boxes."""
        child = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="comp", node_type="container", rooms=frozenset({1}), children=(child,))
        c_sol = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 2, 0, 1, 0, 0),),
        )
        boxes = decompose_cluster_sub_boxes(
            raw_parent_coords={1: (0, 0, 0)},
            raw_child_offsets={"c1": (5, 0, 0)},
            node=parent,
            children=(child,),
            child_solutions={"c1": c_sol},
            boundary_ports=frozenset(),
        )
        assert len(boxes) == 1
        assert boxes[0].min_x == 0
        assert boxes[0].max_x == 7

    def test_solve_and_extract_unavailable_solver(self) -> None:
        """Unavailable solver raises LayoutSolverError."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = False
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError, match="is not available"):
                solve_and_extract(formulation=formulation, solver_name="cbc")

    def test_solve_and_extract_infeasible_status(self) -> None:
        """Infeasible solver exit cleanly raises LayoutSolverError."""
        node = SolveTreeNode(node_id="n", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.error
        mock_results.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_results

        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with pytest.raises(LayoutSolverError, match="returned non-optimal status"):
                solve_and_extract(formulation=formulation, solver_name="cbc")


class TestStage5Normalization:
    """Unit tests for Stage 5: Coordinate Normalization."""

    def test_normalize_empty_cluster(self) -> None:
        """Normalizes degenerate empty cluster into empty NodeSolution."""
        node = SolveTreeNode(node_id="empty", node_type="container", rooms=frozenset())
        sol = normalize_empty_cluster(node=node)
        assert sol.node_id == "empty"
        assert sol.local_coordinates == {}
        assert sol.sub_boxes == ()
        assert sol.metadata == {"child_offsets": {}}

    def test_normalize_single_room_cluster(self) -> None:
        """Single room cluster returns origin coordinate and unit sub-box."""
        node = SolveTreeNode(node_id="single", node_type="leaf", rooms=frozenset({42}))
        sol = normalize_single_room_cluster(node=node, room=42, is_boundary_port=False)
        assert sol.local_coordinates == {42: Coord3D(0, 0, 0)}
        assert sol.sub_boxes == (BoundingBox3D(0, 0, 0, 0, 0, 0),)

        sol_port = normalize_single_room_cluster(node=node, room=42, is_boundary_port=True)
        assert sol_port.sub_boxes == ()

    def test_normalize_cluster_solution_translates_to_origin(self) -> None:
        """Normalizes solved coordinates and sub-boxes by minimum bounding origin."""
        child = SolveTreeNode(node_id="c_child", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(node_id="parent", node_type="container", rooms=frozenset({10}), children=(child,))

        extraction = SolvedClusterExtraction(
            raw_parent_coords={10: (10, 20, 30)},
            raw_child_offsets={"c_child": (15, 25, 35)},
            raw_sub_boxes=(BoundingBox3D(10, 15, 20, 25, 30, 35),),
            solver_status=SolverStatus.OPTIMAL,
        )
        c_sol = NodeSolution(
            node_id="c_child",
            local_coordinates={20: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 1),),
        )

        norm_sol = normalize_cluster_solution(
            extraction=extraction,
            node=parent,
            children=(child,),
            child_solutions={"c_child": c_sol},
        )

        assert norm_sol.local_coordinates[10] == Coord3D(0, 0, 0)
        assert norm_sol.metadata["child_offsets"]["c_child"] == Coord3D(5, 5, 5)
        assert norm_sol.sub_boxes[0] == BoundingBox3D(0, 5, 0, 5, 0, 5)
        assert norm_sol.metadata["status"] == SolverStatus.OPTIMAL

    def test_normalize_cluster_solution_child_without_bbox_and_empty_points(self) -> None:
        """Handles children with no bounding box and empty point collections."""
        child = SolveTreeNode(node_id="c_nobox", node_type="leaf", rooms=frozenset({30}))
        parent = SolveTreeNode(node_id="p_empty_pts", node_type="container", rooms=frozenset(), children=(child,))
        extraction = SolvedClusterExtraction(
            raw_parent_coords={},
            raw_child_offsets={"c_nobox": (2, 3, 4)},
            raw_sub_boxes=(),
            solver_status=SolverStatus.FEASIBLE,
        )
        c_sol = NodeSolution(
            node_id="c_nobox",
            local_coordinates={30: Coord3D(0, 0, 0)},
            sub_boxes=(),
        )
        norm_sol = normalize_cluster_solution(
            extraction=extraction,
            node=parent,
            children=(child,),
            child_solutions={"c_nobox": c_sol},
        )
        assert norm_sol.local_coordinates == {}
        assert norm_sol.metadata["child_offsets"]["c_nobox"] == Coord3D(0, 0, 0)

    def test_solve_and_extract_timeout_and_missing_child_offset(self) -> None:
        """Timeout status returns SolverStatus.TIMEOUT and missing child offset raises error."""
        child = SolveTreeNode(node_id="c_to", node_type="leaf", rooms=frozenset({1}))
        parent = SolveTreeNode(node_id="p_to", node_type="container", rooms=frozenset(), children=(child,))
        inv = inventory_cluster(node=parent, child_solutions={"c_to": NodeSolution("c_to", {}, ())}, graph=_make_graph([1], []))
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.options = {}
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.warning
        mock_results.solver.termination_condition = TerminationCondition.maxTimeLimit
        mock_solver.solve.return_value = mock_results

        opts = LayoutOptions(timeout_seconds=15)
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", return_value=None):
                with pytest.raises(LayoutSolverError, match="failed to retrieve offset values"):
                    solve_and_extract(formulation=formulation, solver_name="cbc", options=opts)
            assert mock_solver.options.get("seconds") == 15

            # When value succeeds
            with patch("pyomo.environ.value", side_effect=[0, 0, 0]):
                ext = solve_and_extract(formulation=formulation, solver_name="glpk", options=opts)
                assert ext.solver_status == SolverStatus.TIMEOUT
                assert mock_solver.options.get("tmlim") == 15
