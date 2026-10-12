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
    ClearanceConstraintBuilder,
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
    GatewayCutHandler,
    GatewayCutResult,
    InterChildBridgeHandler,
    InterChildBridgeResult,
    RoomChildAlignment,
    SplitPairHandler,
    SplitPairResult,
    derive_child_sub_boxes,
    get_child_boundary_ports,
)
from romutil.layout.milp.normalization import (
    normalize_empty_cluster,
    normalize_single_room_cluster,
)
from romutil.layout.milp.solver import (
    SolvedClusterExtraction,
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

    def test_derive_child_sub_boxes_leaf(self) -> None:
        """Leaf child decomposes non-convex interior cells into orthogonal sub-boxes."""
        child = SolveTreeNode(node_id="leaf_l", node_type="leaf", rooms=frozenset({1, 2, 3}))
        child_sol = NodeSolution(
            node_id="leaf_l",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(1, 0, 0),
                3: Coord3D(0, 1, 0),
            },
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset(),
        )
        assert len(boxes) == 2  # L-shape decomposes into 2 disjoint boxes
        assert sum(b.volume() for b in boxes) == 3

    def test_derive_child_sub_boxes_composite(self) -> None:
        """Composite child computes enclosing bounding box over interior cells."""
        grandchild = SolveTreeNode(node_id="gc", node_type="leaf", rooms=frozenset({10}))
        child = SolveTreeNode(
            node_id="comp",
            node_type="container",
            rooms=frozenset({1}),
            children=(grandchild,),
        )
        child_sol = NodeSolution(
            node_id="comp",
            local_coordinates={1: Coord3D(0, 0, 0)},
            metadata={
                "child_offsets": {"gc": Coord3D(5, 0, 0)},
                "bounding_box": BoundingBox3D(0, 5, 0, 0, 0, 0),
            },
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset(),
        )
        assert len(boxes) == 1
        assert boxes[0].min_x == 0
        assert boxes[0].max_x == 5

    def test_derive_child_sub_boxes_composite_recursive_cavity_carving(self) -> None:
        """Composite child aggregates constituent rooms and child sub-boxes, preserving concavities."""
        # Child has an internal room at (0, 0, 0), and a grandchild at offset (0, 4, 0)
        # The region between Y=1 and Y=3 is completely empty.
        grandchild = SolveTreeNode(node_id="gc", node_type="leaf", rooms=frozenset({10}))
        child = SolveTreeNode(
            node_id="comp_concave",
            node_type="container",
            rooms=frozenset({1}),
            children=(grandchild,),
        )
        child_sol = NodeSolution(
            node_id="comp_concave",
            local_coordinates={1: Coord3D(0, 0, 0)},
            metadata={
                "child_offsets": {"gc": Coord3D(0, 4, 0)},
                "child_sub_boxes": {
                    "gc": (BoundingBox3D(0, 0, 0, 0, 0, 0),),
                },
                "bounding_box": BoundingBox3D(0, 0, 0, 4, 0, 0),
            },
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset(),
        )
        # Because Y in [1..3] is void, the decomposition must yield 2 disjoint sub-boxes
        assert len(boxes) == 2
        box_coords = {(b.min_x, b.max_x, b.min_y, b.max_y, b.min_z, b.max_z) for b in boxes}
        assert (0, 0, 0, 0, 0, 0) in box_coords
        assert (0, 0, 4, 4, 0, 0) in box_coords

    def test_derive_child_sub_boxes_composite_excludes_boundary_ports(self) -> None:
        """Composite child strictly excludes boundary ports on constituent and descendant rooms."""
        grandchild = SolveTreeNode(node_id="gc", node_type="leaf", rooms=frozenset({20}))
        child = SolveTreeNode(
            node_id="comp_ports",
            node_type="container",
            rooms=frozenset({1, 2}),  # 1 is internal, 2 is boundary port
            children=(grandchild,),
        )
        child_sol = NodeSolution(
            node_id="comp_ports",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 1, 0)},
            metadata={
                "child_offsets": {"gc": Coord3D(0, 3, 0)},
                "child_sub_boxes": {
                    "gc": (BoundingBox3D(0, 0, 0, 0, 0, 0),),
                },
                "all_room_coordinates": {
                    1: Coord3D(0, 0, 0),
                    2: Coord3D(0, 1, 0),
                    20: Coord3D(0, 3, 0),
                },
            },
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset({2}),
        )
        assert len(boxes) == 2
        # Room 2 at (0, 1, 0) must NOT be contained within any sub-box
        for b in boxes:
            assert not (b.min_x <= 0 <= b.max_x and b.min_y <= 1 <= b.max_y and b.min_z <= 0 <= b.max_z)

    def test_derive_child_sub_boxes_explicit_sub_boxes_punches_boundary_port(self) -> None:
        """Explicit sub-boxes overlapping boundary ports are carved and re-decomposed."""
        child = SolveTreeNode(node_id="c_explicit", node_type="leaf", rooms=frozenset({1, 2}))
        # Explicit box covers X=[0..2], Y=[0..0], Z=[0..0]. Port 2 is at (1, 0, 0).
        child_sol = NodeSolution(
            node_id="c_explicit",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(1, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 2, 0, 0, 0, 0),),
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset({2}),
        )
        # Port 2 at (1, 0, 0) carved out -> remaining are (0, 0, 0) and (2, 0, 0) -> 2 disjoint boxes
        assert len(boxes) == 2
        box_coords = {(b.min_x, b.max_x, b.min_y, b.max_y, b.min_z, b.max_z) for b in boxes}
        assert (0, 0, 0, 0, 0, 0) in box_coords
        assert (2, 2, 0, 0, 0, 0) in box_coords

    def test_derive_child_sub_boxes_empty_when_all_ports(self) -> None:
        """Returns empty tuple when all rooms are boundary ports."""
        child = SolveTreeNode(node_id="c_all_ports", node_type="leaf", rooms=frozenset({1}))
        child_sol = NodeSolution(
            node_id="c_all_ports",
            local_coordinates={1: Coord3D(0, 0, 0)},
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset({1}),
        )
        assert boxes == ()

    def test_derive_child_sub_boxes_excludes_boundary_ports(self) -> None:
        """Boundary ports are strictly excluded when deriving child sub-boxes."""
        child = SolveTreeNode(
            node_id="sp_child",
            node_type="block",
            rooms=frozenset({1, 2, 3}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        child_sol = NodeSolution(
            node_id="sp_child",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(4, 0, 0),
                3: Coord3D(2, 0, 0),
            },
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset({1, 2}),
        )
        assert len(boxes) == 1
        assert boxes[0] == BoundingBox3D(2, 2, 0, 0, 0, 0)

    def test_inventory_cluster_derives_child_sub_boxes(self) -> None:
        """Stage 1 inventory derives child_sub_boxes for all child clusters."""
        child = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10, 20}))
        parent = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset({1}), children=(child,))
        c_sol = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0), 20: Coord3D(1, 0, 0)},
        )
        graph = _make_graph([1, 10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c1": c_sol}, graph=graph)
        assert "c1" in inv.child_sub_boxes
        assert len(inv.child_sub_boxes["c1"]) == 1
        assert inv.child_sub_boxes["c1"][0] == BoundingBox3D(0, 1, 0, 0, 0, 0)


class TestStage2Formulation:
    """Unit tests for Stage 2: Basic MILP Formulation."""

    def test_cluster_formulation_canonical_constructor(self) -> None:
        """ClusterFormulation can be constructed directly with inventory and options."""
        node = SolveTreeNode(node_id="canon_node", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        opts = LayoutOptions(timeout_seconds=10)

        formulation = ClusterFormulation(inventory=inv, options=opts)
        assert formulation.inventory is inv
        assert formulation.options is opts
        assert formulation.model is not None
        assert formulation.m_coord >= 1000
        assert formulation.m_big == 2 * formulation.m_coord

    def test_formulate_cluster_model_variables_and_anchors(self) -> None:
        """Formulation declares room variables, fixes anchor, and provides member accessors."""
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

    def test_formulation_child_point_and_sub_box_accessors(self) -> None:
        """Child point and sub-box accessors correctly resolve local offsets."""
        child = SolveTreeNode(node_id="c_sub", node_type="leaf", rooms=frozenset({5}))
        parent = SolveTreeNode(node_id="p_sub", node_type="container", rooms=frozenset({1}), children=(child,))
        sol = NodeSolution(
            node_id="c_sub",
            local_coordinates={5: Coord3D(2, 3, 4)},
            sub_boxes=(BoundingBox3D(1, 3, 2, 4, 3, 5),),
        )
        inv = inventory_cluster(node=parent, child_solutions={"c_sub": sol}, graph=_make_graph([1, 5], []))
        formulation = ClusterFormulation(inventory=inv)

        pt_child = formulation.get_point(("c_sub", 5))
        assert str(pt_child[0]) == "cx[c_sub] + 2"
        assert str(pt_child[1]) == "cy[c_sub] + 3"
        assert str(pt_child[2]) == "cz[c_sub] + 4"

        sub_box = formulation.get_child_sub_box(("c_sub", 0))
        assert str(sub_box[0]) == "cx[c_sub] + 1"
        assert str(sub_box[1]) == "cx[c_sub] + 3"
        assert str(sub_box[2]) == "cy[c_sub] + 2"
        assert str(sub_box[3]) == "cy[c_sub] + 4"
        assert str(sub_box[4]) == "cz[c_sub] + 3"
        assert str(sub_box[5]) == "cz[c_sub] + 5"

        # Also test with raw tuple sub-box fallback in child_sub_boxes
        inv.child_sub_boxes["c_sub"] = ((Coord3D(0, 0, 0), Coord3D(1, 1, 1)),)
        raw_box = formulation.get_child_sub_box(("c_sub", 0))
        assert str(raw_box[0]) == "cx[c_sub]"
        assert str(raw_box[1]) == "cx[c_sub] + 1"

    def test_formulation_corridor_boxes_all_directions(self) -> None:
        """Corridor interior and full boxes are evaluated correctly across all 6 directions."""
        node = SolveTreeNode(node_id="n_dirs", node_type="leaf", rooms=frozenset({1, 2}))
        inv = inventory_cluster(node=node, child_solutions={}, graph=_make_graph([1, 2], []))
        formulation = ClusterFormulation(inventory=inv)

        directions = [
            ((1, 0, 0), "east"),
            ((-1, 0, 0), "west"),
            ((0, 1, 0), "north"),
            ((0, -1, 0), "south"),
            ((0, 0, 1), "up"),
            ((0, 0, -1), "down"),
        ]

        for dir_vec, name in directions:
            ae = ActiveEdge(
                edge_id=f"e_{name}",
                src=1,
                dst=2,
                direction=dir_vec,
                min_length=3,
                associated_rooms=frozenset({1, 2}),
                associated_children=frozenset(),
            )
            int_box = formulation.get_corridor_interior_box(ae)
            full_box = formulation.get_corridor_full_box(ae)
            assert len(int_box) == 6
            assert len(full_box) == 6


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

    def test_formulation_apply_clearances_member_method(self) -> None:
        """Direct invocation of formulation.apply_clearances() attaches clearance constraints."""
        node = SolveTreeNode(node_id="p_member", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        assert not hasattr(formulation.model, "pr_sep_sum")
        formulation.apply_clearances()
        assert hasattr(formulation.model, "pr_sep_sum")

    def test_formulation_apply_clearance_constraints_alias(self) -> None:
        """Calling formulation.apply_clearance_constraints() delegates cleanly to apply_clearances."""
        node = SolveTreeNode(node_id="p_alias", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        formulation.apply_clearance_constraints()
        assert hasattr(formulation.model, "pr_sep_sum")

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

    def test_apply_clearances_child_child_separation(self) -> None:
        """Pairwise child-child sub-box separation is attached for non-shared children."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(node_id="p_cc", node_type="container", rooms=frozenset({1}), children=(c1, c2))
        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),),
        )
        graph = _make_graph([1, 10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c1": sol1, "c2": sol2}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        formulation.apply_clearances()
        assert hasattr(formulation.model, "cb_br")
        assert hasattr(formulation.model, "cb_sep_sum")

    def test_apply_clearances_corridor_child_and_crossing(self) -> None:
        """Corridor vs. sibling child and corridor crossing separation constraints are created."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        c_empty = SolveTreeNode(node_id="c_empty", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(
            node_id="p_cross",
            node_type="container",
            rooms=frozenset({1, 2, 3, 4}),
            children=(c1, c_empty),
        )
        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),),
        )
        sol_empty = NodeSolution(
            node_id="c_empty",
            local_coordinates={20: Coord3D(0, 0, 0)},
            sub_boxes=(),
        )
        # Create 2 orthogonal non-incident edges
        graph = _make_graph(
            [1, 2, 3, 4, 10, 20],
            [
                (1, 2, Direction.east, 3),
                (3, 4, Direction.north, 3),
            ],
        )
        inv = inventory_cluster(node=parent, child_solutions={"c1": sol1, "c_empty": sol_empty}, graph=graph)
        # Add a non-orthogonal edge to test direction filtering branch
        diag_edge = ActiveEdge(
            edge_id="e_diag",
            src=1,
            dst=3,
            direction=(1, 1, 0),
            min_length=1,
            associated_rooms=frozenset({1, 3}),
            associated_children=frozenset(),
        )
        import dataclasses
        empty_sub_boxes = dict(inv.child_sub_boxes)
        empty_sub_boxes["c_empty"] = ()
        inv = dataclasses.replace(
            inv,
            active_edges=inv.active_edges + (diag_edge,),
            child_sub_boxes=empty_sub_boxes,
        )
        formulation = ClusterFormulation(inventory=inv)

        formulation.apply_clearances()
        assert hasattr(formulation.model, "cws_br")  # Corridor vs sibling child
        assert hasattr(formulation.model, "cc_br")   # Corridor crossing

    def test_clearance_constraint_builder_direct(self) -> None:
        """ClearanceConstraintBuilder can be instantiated and executed directly."""
        node = SolveTreeNode(node_id="p_builder", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        builder = ClearanceConstraintBuilder(formulation)
        builder.build()
        assert hasattr(formulation.model, "pr_sep_sum")

    def test_individual_clearance_methods_on_formulation(self) -> None:
        """Individual clearance separation methods on ClusterFormulation execute without error."""
        node = SolveTreeNode(node_id="p_indiv", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        formulation.apply_parent_room_separation()
        formulation.apply_child_child_separation()
        formulation.apply_room_child_separation()
        formulation.apply_corridor_room_separation()
        formulation.apply_corridor_child_separation()
        formulation.apply_corridor_crossing_separation()

        assert hasattr(formulation.model, "pr_sep_sum")


class TestStage4Solver:
    """Unit tests for Stage 4: Solve and Extraction."""

    def test_solve_and_extract_raw_coordinates_and_offsets(self) -> None:
        """Solved extraction contains raw parent coordinates, child offsets, and solver status."""
        child = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset({1, 2}), children=(child,))
        graph = _make_graph([1, 2, 10], [(1, 2, Direction.east, 1)])
        c_sol = NodeSolution(node_id="c1", local_coordinates={10: Coord3D(0, 0, 0)})
        inv = inventory_cluster(node=parent, child_solutions={"c1": c_sol}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        ext = solve_and_extract(formulation=formulation, solver_name="cbc")
        assert 1 in ext.raw_parent_coords
        assert 2 in ext.raw_parent_coords
        assert "c1" in ext.raw_child_offsets
        assert ext.solver_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert not hasattr(ext, "raw_sub_boxes")

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
        """Single room cluster returns origin coordinate and empty sub_boxes."""
        node = SolveTreeNode(node_id="single", node_type="leaf", rooms=frozenset({42}))
        sol = normalize_single_room_cluster(node=node, room=42, is_boundary_port=False)
        assert sol.local_coordinates == {42: Coord3D(0, 0, 0)}
        assert sol.sub_boxes == ()

        sol_port = normalize_single_room_cluster(node=node, room=42, is_boundary_port=True)
        assert sol_port.sub_boxes == ()

    def test_normalize_cluster_solution_translates_to_origin(self) -> None:
        """Normalizes solved coordinates and child offsets by minimum bounding origin."""
        child = SolveTreeNode(node_id="c_child", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(node_id="parent", node_type="container", rooms=frozenset({10}), children=(child,))

        extraction = SolvedClusterExtraction(
            raw_parent_coords={10: (10, 20, 30)},
            raw_child_offsets={"c_child": (15, 25, 35)},
            solver_status=SolverStatus.OPTIMAL,
        )
        c_sol = NodeSolution(
            node_id="c_child",
            local_coordinates={20: Coord3D(0, 0, 0)},
            sub_boxes=(),
        )

        norm_sol = normalize_cluster_solution(
            extraction=extraction,
            node=parent,
            children=(child,),
            child_solutions={"c_child": c_sol},
        )

        assert norm_sol.local_coordinates[10] == Coord3D(0, 0, 0)
        assert norm_sol.metadata["child_offsets"]["c_child"] == Coord3D(5, 5, 5)
        assert norm_sol.sub_boxes == ()
        assert norm_sol.metadata["status"] == SolverStatus.OPTIMAL

    def test_normalize_cluster_solution_child_without_bbox_and_empty_points(self) -> None:
        """Handles children with no bounding box and empty point collections."""
        child = SolveTreeNode(node_id="c_nobox", node_type="leaf", rooms=frozenset({30}))
        parent = SolveTreeNode(node_id="p_empty_pts", node_type="container", rooms=frozenset(), children=(child,))
        extraction = SolvedClusterExtraction(
            raw_parent_coords={},
            raw_child_offsets={"c_nobox": (2, 3, 4)},
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
        assert norm_sol.sub_boxes == ()

    def test_normalize_cluster_solution_aggregates_child_sub_boxes_and_rooms(self) -> None:
        """Normalizes composite cluster, storing child sub-boxes and descendant room coordinates."""
        grandchild = SolveTreeNode(node_id="gc", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset({10}), children=(grandchild,))
        extraction = SolvedClusterExtraction(
            raw_parent_coords={10: (0, 0, 0)},
            raw_child_offsets={"gc": (2, 3, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        gc_sol = NodeSolution(
            node_id="gc",
            local_coordinates={20: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),),
        )
        norm_sol = normalize_cluster_solution(
            extraction=extraction,
            node=parent,
            children=(grandchild,),
            child_solutions={"gc": gc_sol},
            child_sub_boxes={"gc": (BoundingBox3D(0, 1, 0, 1, 0, 0),)},
        )
        assert "child_sub_boxes" in norm_sol.metadata
        assert "gc" in norm_sol.metadata["child_sub_boxes"]
        assert norm_sol.metadata["child_sub_boxes"]["gc"] == (BoundingBox3D(0, 1, 0, 1, 0, 0),)
        assert "all_room_coordinates" in norm_sol.metadata
        assert norm_sol.metadata["all_room_coordinates"][10] == Coord3D(0, 0, 0)
        assert norm_sol.metadata["all_room_coordinates"][20] == Coord3D(2, 3, 0)

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


class TestTypedChildHandlers:
    """Direct unit tests for typed child interface classifiers and handlers."""

    def test_split_pair_handler_grouping_and_alignments(self) -> None:
        """SplitPairHandler groups by canonical poles and aligns to parent rooms."""
        p0, p1 = 2, 5
        child1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({2, 5, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        child2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({2, 5, 30}),
            metadata={"cut_type": "split_pair", "poles": [5, 2]},
        )
        parent = SolveTreeNode(
            node_id="p_split",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({2, 5}),
            children=(child1, child2),
            metadata={"poles": (2, 5)},
        )
        c1_sol = NodeSolution("c1", {2: Coord3D(0, 0, 0), 5: Coord3D(4, 0, 0), 20: Coord3D(2, 0, 0)}, ())
        c2_sol = NodeSolution("c2", {2: Coord3D(0, 0, 0), 5: Coord3D(3, 0, 0), 30: Coord3D(1, 0, 0)}, ())
        graph = _make_graph([2, 5, 20, 30], [(2, 5, Direction.east, 3)])

        handler = SplitPairHandler(node=parent, child_solutions={"c1": c1_sol, "c2": c2_sol}, graph=graph)
        grouped = handler.group_split_children()
        assert (2, 5) in grouped
        assert set(c.node_id for c in grouped[(2, 5)]) == {"c1", "c2"}

        res = handler.process()
        assert len(res.room_alignments) == 2
        assert frozenset({"c1", "c2"}) in res.equated_children
        assert res.split_min_lengths[(2, 5)] == 4
        assert res.split_directions[(2, 5)] == (1, 0, 0)
        assert len(res.active_edges) == 1
        assert res.active_edges[0].edge_id == "sp_2_5"

    def test_split_pair_handler_transverse_anchor_selection(self) -> None:
        """Transverse exits select the transverse anchor pole over default pole."""
        p0, p1 = 1, 2
        child = SolveTreeNode(
            node_id="c_trans",
            node_type="leaf",
            rooms=frozenset({1, 2, 10}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="p_trans",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        c_sol = NodeSolution(
            "c_trans",
            {1: Coord3D(0, 0, 0), 2: Coord3D(4, 0, 0), 10: Coord3D(4, 1, 0)},
            (),
        )
        graph = _make_graph([1, 2, 10], [(2, 10, Direction.north, 1)])

        handler = SplitPairHandler(node=parent, child_solutions={"c_trans": c_sol}, graph=graph)
        res = handler.process()
        assert res.child_anchor_poles["c_trans"] == 2
        assert len(res.room_alignments) == 1
        assert res.room_alignments[0].room_vnum == 2

    def test_split_pair_handler_child_child_alignment(self) -> None:
        """When poles are not in parent room set, secondary children align to base child."""
        p0, p1 = 10, 20
        child1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        child2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="p_no_poles",
            node_type="container",
            rooms=frozenset({1}),
            children=(child1, child2),
        )
        c1_sol = NodeSolution("c1", {10: Coord3D(0, 0, 0), 20: Coord3D(3, 0, 0)}, ())
        c2_sol = NodeSolution("c2", {10: Coord3D(1, 0, 0), 20: Coord3D(4, 0, 0)}, ())
        graph = _make_graph([1, 10, 20], [])

        handler = SplitPairHandler(node=parent, child_solutions={"c1": c1_sol, "c2": c2_sol}, graph=graph)
        res = handler.process()
        assert len(res.room_alignments) == 0
        assert len(res.child_alignments) == 2
        assert res.child_alignments[0].base_child_id == "c1"
        assert res.child_alignments[0].target_child_id == "c1"
        assert res.child_alignments[1].base_child_id == "c1"
        assert res.child_alignments[1].target_child_id == "c2"
        assert frozenset({"c1", "c2"}) in res.equated_children

    def test_gateway_cut_handler_articulation_points_and_bridges(self) -> None:
        """GatewayCutHandler aligns articulation points and creates gw_ active edges."""
        c_art1 = SolveTreeNode(
            node_id="c_art1",
            node_type="leaf",
            rooms=frozenset({1, 11}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        c_art2 = SolveTreeNode(
            node_id="c_art2",
            node_type="leaf",
            rooms=frozenset({1, 12}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        c_brg_fwd = SolveTreeNode(
            node_id="c_brg_fwd",
            node_type="leaf",
            rooms=frozenset({21}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(2, 21), direction=Direction.south, weight=3)},
        )
        c_brg_rev = SolveTreeNode(
            node_id="c_brg_rev",
            node_type="leaf",
            rooms=frozenset({31}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(31, 3), direction=Direction.up, weight=2)},
        )
        c_door = SolveTreeNode(
            node_id="c_door",
            node_type="leaf",
            rooms=frozenset({41}),
        )
        parent = SolveTreeNode(
            node_id="p_gw",
            node_type="block",
            rooms=frozenset({1, 2, 3, 4}),
            children=(c_art1, c_art2, c_brg_fwd, c_brg_rev, c_door),
        )
        sols = {
            "c_art1": NodeSolution("c_art1", {1: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)}, ()),
            "c_art2": NodeSolution("c_art2", {1: Coord3D(0, 0, 0), 12: Coord3D(0, 1, 0)}, ()),
            "c_brg_fwd": NodeSolution("c_brg_fwd", {21: Coord3D(0, 0, 0)}, ()),
            "c_brg_rev": NodeSolution("c_brg_rev", {31: Coord3D(0, 0, 0)}, ()),
            "c_door": NodeSolution("c_door", {41: Coord3D(0, 0, 0)}, ()),
        }
        graph = _make_graph([1, 2, 3, 4, 11, 12, 21, 31, 41], [(4, 41, Direction.west, 1)])

        handler = GatewayCutHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process()

        assert len(res.room_alignments) == 2
        assert frozenset({"c_art1", "c_art2"}) in res.equated_children
        gw_edge_ids = {e.edge_id for e in res.active_edges}
        assert "gw_c_brg_fwd_2_21" in gw_edge_ids
        assert "gw_c_brg_rev_31_3" in gw_edge_ids
        assert "gw_c_door_4_41" in gw_edge_ids
        assert len(res.active_edges) == 3

    def test_inter_child_bridge_handler_edges(self) -> None:
        """InterChildBridgeHandler discovers edges connecting distinct children."""
        child_a = SolveTreeNode(node_id="ca", node_type="leaf", rooms=frozenset({10}))
        child_b = SolveTreeNode(
            node_id="cb",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(10, 20), direction=Direction.east, weight=2)},
        )
        child_c = SolveTreeNode(node_id="cc", node_type="leaf", rooms=frozenset({30}))
        parent = SolveTreeNode(
            node_id="p_icb",
            node_type="container",
            rooms=frozenset(),
            children=(child_a, child_b, child_c),
        )
        sols = {
            "ca": NodeSolution("ca", {10: Coord3D(0, 0, 0)}, ()),
            "cb": NodeSolution("cb", {20: Coord3D(0, 0, 0)}, ()),
            "cc": NodeSolution("cc", {30: Coord3D(0, 0, 0)}, ()),
        }
        graph = _make_graph([10, 20, 30], [(20, 30, Direction.north, 4)])

        handler = InterChildBridgeHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process(aligned_children={"ca"}, equated_children=set())

        assert "cb" in res.aligned_children
        assert "cc" in res.aligned_children
        edge_ids = {e.edge_id for e in res.active_edges}
        assert "ccb_ca_cb_10_20" in edge_ids
        assert "ccb_cb_cc_20_30" in edge_ids

    def test_inter_child_bridge_handler_single_child(self) -> None:
        """InterChildBridgeHandler returns empty result when children count is 1 or less."""
        child = SolveTreeNode(node_id="single", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p", node_type="container", rooms=frozenset(), children=(child,))
        handler = InterChildBridgeHandler(
            node=parent,
            child_solutions={"single": NodeSolution("single", {10: Coord3D(0, 0, 0)}, ())},
            graph=_make_graph([10], []),
        )
        res = handler.process()
        assert res.active_edges == ()
        assert res.aligned_children == frozenset()

    def test_cluster_inventory_from_node_factory(self) -> None:
        """ClusterInventory.from_node produces identical result to inventory_cluster."""
        node = SolveTreeNode(node_id="leaf", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])

        inv1 = inventory_cluster(node=node, child_solutions={}, graph=graph)
        inv2 = ClusterInventory.from_node(node=node, child_solutions={}, graph=graph)

        assert inv1 == inv2
        assert inv2.rooms == (1, 2)

    def test_split_pair_handler_inter_split_corridors(self) -> None:
        """SplitPairHandler builds active edges for corridors between sibling split children."""
        child_sp1 = SolveTreeNode(
            node_id="sp1",
            node_type="leaf",
            rooms=frozenset({1, 2, 10}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        child_sp2 = SolveTreeNode(
            node_id="sp2",
            node_type="leaf",
            rooms=frozenset({3, 4, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(3, 4))},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(child_sp1, child_sp2),
        )
        sols = {
            "sp1": NodeSolution("sp1", {1: Coord3D(0, 0, 0), 2: Coord3D(2, 0, 0), 10: Coord3D(1, 0, 0)}, ()),
            "sp2": NodeSolution("sp2", {3: Coord3D(0, 0, 0), 4: Coord3D(2, 0, 0), 20: Coord3D(1, 0, 0)}, ()),
        }
        # Inter-split edge from boundary port 3 of sp2 to interior room 10 of sp1
        graph = _make_graph([1, 2, 3, 4, 10, 20], [(3, 10, Direction.north, 2)])

        handler = SplitPairHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process()
        edge_ids = {e.edge_id for e in res.corridor_edges}
        assert "ext_sp2_sp1_3_10" in edge_ids
        ext_edge = next(e for e in res.corridor_edges if e.edge_id == "ext_sp2_sp1_3_10")
        assert ext_edge.allow_transverse_slack is True

    def test_inter_child_bridge_reverse_orientations(self) -> None:
        """InterChildBridgeHandler handles reverse orientations for bridge cut and graph edges."""
        child_a = SolveTreeNode(node_id="ca", node_type="leaf", rooms=frozenset({10}))
        # Bridge cut defined with bridge_edge=(20, 10) where 10 in ca and 20 in cb
        child_b = SolveTreeNode(
            node_id="cb",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(20, 10), direction=Direction.west, weight=1)},
        )
        child_c = SolveTreeNode(node_id="cc", node_type="leaf", rooms=frozenset({30}))
        parent = SolveTreeNode(
            node_id="p_rev",
            node_type="container",
            rooms=frozenset(),
            children=(child_a, child_b, child_c),
        )
        sols = {
            "ca": NodeSolution("ca", {10: Coord3D(0, 0, 0)}, ()),
            "cb": NodeSolution("cb", {20: Coord3D(0, 0, 0)}, ()),
            "cc": NodeSolution("cc", {30: Coord3D(0, 0, 0)}, ()),
        }
        # Graph edge where dst is in already aligned child cb and src is in unaligned child cc
        graph = _make_graph([10, 20, 30], [(30, 20, Direction.south, 3)])

        handler = InterChildBridgeHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process(aligned_children={"ca"}, equated_children=set())
        edge_ids = {e.edge_id for e in res.active_edges}
        assert "ccb_cb_ca_20_10" in edge_ids
        assert "ccb_cc_cb_30_20" in edge_ids
