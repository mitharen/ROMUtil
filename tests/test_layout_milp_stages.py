"""Unit tests for the modular 5-stage MILP cluster layout pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import pyomo.environ as pyo
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
    ClearanceViolation,
    ClusterFormulation,
    ClusterInventory,
    calculate_dynamic_bounds,
    detect_cluster_collisions,
    formulate_cluster_model,
    inventory_cluster,
    normalize_cluster_solution,
    solve_and_extract,
)
from romutil.layout.milp.cluster import solve_cluster
from romutil.layout.milp.inventory import (
    ChildChildAlignment,
    GatewayCutHandler,
    GatewayCutResult,
    InterChildBridgeHandler,
    InterChildBridgeResult,
    SplitPairHandler,
    SplitPairResult,
    _build_internal_room_edges,
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
from romutil.models import Direction, Exit, Room
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

        node_art_contract = SolveTreeNode(
            node_id="c_art_contract",
            node_type="leaf",
            rooms=frozenset({10, 11}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=10)},
        )
        assert get_child_boundary_ports(node_art_contract) == {10}

        node_art_meta_enum = SolveTreeNode(
            node_id="c_art_enum",
            node_type="leaf",
            rooms=frozenset({12, 13}),
            metadata={"cut_type": CutType.ARTICULATION_POINT, "anchor_vnum": 12},
        )
        assert get_child_boundary_ports(node_art_meta_enum) == {12}

        node_art_meta_str = SolveTreeNode(
            node_id="c_art_str",
            node_type="leaf",
            rooms=frozenset({14, 15}),
            metadata={"cut_type": "articulation_point", "anchor_vnum": "14"},
        )
        assert get_child_boundary_ports(node_art_meta_str) == {14}

        node_art_meta_anchor_only = SolveTreeNode(
            node_id="c_art_anchor_only",
            node_type="leaf",
            rooms=frozenset({16, 17}),
            metadata={"anchor_vnum": 16},
        )
        assert get_child_boundary_ports(node_art_meta_anchor_only) == {16}

        node_bridge_contract_with_anchor = SolveTreeNode(
            node_id="c_bridge_contract_anchor",
            node_type="leaf",
            rooms=frozenset({20, 21}),
            metadata={
                "cut_contract": BridgeCut(bridge_edge=(1, 20), direction=0),
                "anchor_vnum": 20,
            },
        )
        assert get_child_boundary_ports(node_bridge_contract_with_anchor) == {20}

        node_bridge_meta_enum = SolveTreeNode(
            node_id="c_bridge_enum",
            node_type="leaf",
            rooms=frozenset({22, 23}),
            metadata={"cut_type": CutType.BRIDGE, "anchor_vnum": 22},
        )
        assert get_child_boundary_ports(node_bridge_meta_enum) == {22}

        node_bridge_meta_str = SolveTreeNode(
            node_id="c_bridge_str",
            node_type="leaf",
            rooms=frozenset({24, 25}),
            metadata={"cut_type": "bridge", "anchor_vnum": "24"},
        )
        assert get_child_boundary_ports(node_bridge_meta_str) == {24}

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
            rooms=frozenset({20}),
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
            local_coordinates={20: Coord3D(1, 0, 0)},
            port_coordinates={2: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(1, 1, 0, 0, 0, 0),),
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

        assert any(e.edge_id == "gw_c_art_2_20" and e.src == 2 and e.dst == ("c_art", 20) for e in inv.active_edges)
        gw_edges = [e for e in inv.active_edges if e.edge_id.startswith("gw_")]
        assert len(gw_edges) == 2
        assert any(e.src == 1 and e.dst == ("c_brg", 30) and e.min_length == 2 for e in gw_edges)
        assert any(e.src == 2 and e.dst == ("c_art", 20) and e.min_length == 1 for e in gw_edges)

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
        """Composite child computes orthogonal keep-out sub-boxes over interior cells."""
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
                "all_room_coordinates": {1: Coord3D(0, 0, 0), 10: Coord3D(5, 0, 0)},
            },
        )
        boxes = derive_child_sub_boxes(
            child=child,
            child_solution=child_sol,
            boundary_ports=frozenset(),
        )
        assert len(boxes) == 2
        assert boxes[0].min_x == 0
        assert boxes[1].max_x == 5

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

    def test_derive_child_sub_boxes_articulation_block_multi_z_isolation(self) -> None:
        """1-cut articulation block decomposes into multi-Z sub-boxes preserving vacant upper levels."""
        # Catacomb child at z=0 spans [0..4] x [0..4] x [0..0]
        catacomb = SolveTreeNode(
            node_id="catacombs",
            node_type="leaf",
            rooms=frozenset(range(1, 26)),
        )
        catacomb_box = BoundingBox3D(0, 4, 0, 4, 0, 0)

        # Composite block has surface rooms at z=2:
        # Room 100 is the articulation anchor (boundary port) at (0, 0, 2).
        # Room 101 is an interior surface room at (1, 0, 2).
        block_child = SolveTreeNode(
            node_id="art_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({100, 101}),
            children=(catacomb,),
            metadata={
                "cut_contract": ArticulationPointCut(anchor_vnum=100),
                "anchor_vnum": 100,
            },
        )
        catacomb_rooms = {
            (i + 1): Coord3D(x, y, 0)
            for i, (x, y) in enumerate(
                (x, y) for x in range(5) for y in range(5)
            )
        }
        all_rooms = dict(catacomb_rooms)
        all_rooms[100] = Coord3D(0, 0, 2)
        all_rooms[101] = Coord3D(1, 0, 2)

        child_sol = NodeSolution(
            node_id="art_block",
            local_coordinates={100: Coord3D(0, 0, 2), 101: Coord3D(1, 0, 2)},
            metadata={
                "bounding_box": BoundingBox3D(0, 4, 0, 4, 0, 2),
                "child_offsets": {"catacombs": Coord3D(0, 0, 0)},
                "child_sub_boxes": {"catacombs": (catacomb_box,)},
                "all_room_coordinates": all_rooms,
            },
        )

        # Verify get_child_boundary_ports detects the articulation anchor
        ports = get_child_boundary_ports(block_child)
        assert ports == {100}

        # Derive sub-boxes using the extracted boundary ports
        boxes = derive_child_sub_boxes(block_child, child_sol, boundary_ports=frozenset(ports))

        # Must decompose into discrete sub-boxes rather than a monolithic bounding box
        assert len(boxes) == 2
        z_ranges = {(b.min_z, b.max_z) for b in boxes}
        assert z_ranges == {(0, 0), (2, 2)}

        # The underground catacomb box is confined strictly to z=0
        catacomb_derived = [b for b in boxes if b.min_z == 0][0]
        assert catacomb_derived == BoundingBox3D(0, 4, 0, 4, 0, 0)

        # The surface box is confined strictly to room 101 at z=2
        surface_derived = [b for b in boxes if b.min_z == 2][0]
        assert surface_derived == BoundingBox3D(1, 1, 0, 0, 2, 2)

        # Boundary anchor (0, 0, 2) is excluded and vacant surface cells (e.g. (2, 2, 2)) are unblocked
        assert not any(b.contains_point(Coord3D(0, 0, 2)) for b in boxes)
        assert not any(b.contains_point(Coord3D(2, 2, 2)) for b in boxes)
        assert not any(b.contains_point(Coord3D(4, 4, 2)) for b in boxes)

    def test_derive_child_sub_boxes_bridge_block_decomposition(self) -> None:
        """1-cut bridge block child decomposes occupied cells into discrete sub-boxes."""
        child_leaf = SolveTreeNode(
            node_id="bridge_leaf",
            node_type="leaf",
            rooms=frozenset({1, 2}),
        )
        block_bridge = SolveTreeNode(
            node_id="bridge_block",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({10, 20}),
            children=(child_leaf,),
            metadata={
                "cut_contract": BridgeCut(bridge_edge=(10, 5), direction=0),
                "anchor_vnum": 10,
            },
        )
        child_sol = NodeSolution(
            node_id="bridge_block",
            local_coordinates={10: Coord3D(0, 0, 0), 20: Coord3D(3, 0, 0)},
            metadata={
                "bounding_box": BoundingBox3D(0, 3, 0, 0, 0, 0),
                "child_offsets": {"bridge_leaf": Coord3D(5, 0, 0)},
                "child_sub_boxes": {"bridge_leaf": (BoundingBox3D(0, 1, 0, 0, 0, 0),)},
                "all_room_coordinates": {
                    10: Coord3D(0, 0, 0),
                    20: Coord3D(3, 0, 0),
                    1: Coord3D(5, 0, 0),
                    2: Coord3D(6, 0, 0),
                },
            },
        )
        ports = get_child_boundary_ports(block_bridge)
        assert ports == {10}
        boxes = derive_child_sub_boxes(block_bridge, child_sol, boundary_ports=frozenset(ports))
        assert len(boxes) == 2
        # Room 10 at (0, 0, 0) must be excluded
        assert not any(b.contains_point(Coord3D(0, 0, 0)) for b in boxes)
        # Room 20 at (3, 0, 0) and leaf at (5..6, 0, 0) are kept
        assert any(b.contains_point(Coord3D(3, 0, 0)) for b in boxes)
        assert any(b.contains_point(Coord3D(5, 0, 0)) for b in boxes)

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

    def test_get_child_boundary_ports_metadata_parsing(self) -> None:
        """Parse child boundary ports from metadata with varied pole configurations."""
        node_no_poles = SolveTreeNode(
            node_id="c_no_poles",
            node_type="leaf",
            rooms=frozenset({1, 2}),
            metadata={"cut_type": CutType.SPLIT_PAIR},
        )
        assert get_child_boundary_ports(node_no_poles) == set()

        node_single_pole = SolveTreeNode(
            node_id="c_single_pole",
            node_type="leaf",
            rooms=frozenset({1, 2}),
            metadata={"cut_type": "split_pair", "poles": [10]},
        )
        assert get_child_boundary_ports(node_single_pole) == set()

        node_three_poles = SolveTreeNode(
            node_id="c_three_poles",
            node_type="leaf",
            rooms=frozenset({1, 2, 3}),
            metadata={"poles": [10, 20, 30]},
        )
        assert get_child_boundary_ports(node_three_poles) == set()

        node_none_poles = SolveTreeNode(
            node_id="c_none_poles",
            node_type="leaf",
            rooms=frozenset({1, 2}),
            metadata={"poles": None},
        )
        assert get_child_boundary_ports(node_none_poles) == set()

        mock_child = MagicMock(spec=SolveTreeNode)
        mock_child.cut_contract = None
        mock_child.cut_type = CutType.SPLIT_PAIR
        mock_child.metadata = {"poles": ("6", "7")}
        assert get_child_boundary_ports(mock_child) == {6, 7}

        mock_art_none = MagicMock(spec=SolveTreeNode)
        mock_art_none.cut_contract = None
        mock_art_none.cut_type = CutType.ARTICULATION_POINT
        mock_art_none.metadata = {"anchor_vnum": None}
        assert get_child_boundary_ports(mock_art_none) == set()

        mock_art_child = MagicMock(spec=SolveTreeNode)
        mock_art_child.cut_contract = None
        mock_art_child.cut_type = CutType.ARTICULATION_POINT
        mock_art_child.metadata = {"anchor_vnum": "42"}
        assert get_child_boundary_ports(mock_art_child) == {42}

        mock_bridge_contract = MagicMock(spec=SolveTreeNode)
        mock_contract = MagicMock(spec=BridgeCut)
        mock_contract.anchor_vnum = 99
        mock_bridge_contract.cut_contract = mock_contract
        mock_bridge_contract.metadata = {}
        assert get_child_boundary_ports(mock_bridge_contract) == {99}

    def test_derive_child_sub_boxes_fallbacks_and_empty_coordinates(self) -> None:
        """Derive sub-boxes handling None boundary ports, bounding box fallbacks, and empty cells."""
        child_sp = SolveTreeNode(
            node_id="c_sp_none",
            node_type="leaf",
            rooms=frozenset({1, 2, 3}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        sol_sp = NodeSolution(
            node_id="c_sp_none",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(2, 0, 0), 3: Coord3D(1, 0, 0)},
        )
        boxes_sp = derive_child_sub_boxes(child=child_sp, child_solution=sol_sp, boundary_ports=None)
        assert len(boxes_sp) == 1
        assert boxes_sp[0] == BoundingBox3D(1, 1, 0, 0, 0, 0)

        grandchild = SolveTreeNode(node_id="gc", node_type="leaf", rooms=frozenset({10}))
        comp_child = SolveTreeNode(
            node_id="comp_no_boxes",
            node_type="container",
            rooms=frozenset({1}),
            children=(grandchild,),
        )
        sol_comp_no_bbox = NodeSolution(node_id="comp_no_boxes", local_coordinates={})
        assert derive_child_sub_boxes(comp_child, sol_comp_no_bbox) == ()

        block_child = SolveTreeNode(
            node_id="block_child",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1}),
            children=(grandchild,),
        )
        sol_block_no_bbox = NodeSolution(
            node_id="block_child",
            local_coordinates={},
            metadata={"child_sub_boxes": {}},
        )
        assert derive_child_sub_boxes(block_child, sol_block_no_bbox) == ()

        gc1 = SolveTreeNode(node_id="gc1", node_type="leaf", rooms=frozenset({10}))
        gc2 = SolveTreeNode(node_id="gc2", node_type="leaf", rooms=frozenset({20}))
        gc3 = SolveTreeNode(node_id="gc3", node_type="leaf", rooms=frozenset({30}))
        comp_partial = SolveTreeNode(
            node_id="comp_partial",
            node_type="container",
            rooms=frozenset({1}),
            children=(gc1, gc2, gc3),
        )
        sol_comp_partial = NodeSolution(
            node_id="comp_partial",
            local_coordinates={1: Coord3D(0, 0, 0)},
            metadata={
                "child_offsets": {"gc1": Coord3D(3, 0, 0)},
                "child_sub_boxes": {"gc2": (BoundingBox3D(0, 0, 0, 0, 0, 0),)},
            },
        )
        boxes_partial = derive_child_sub_boxes(comp_partial, sol_comp_partial)
        assert len(boxes_partial) == 1
        assert boxes_partial[0] == BoundingBox3D(0, 0, 0, 0, 0, 0)

        comp_empty_cells = SolveTreeNode(
            node_id="comp_empty",
            node_type="container",
            rooms=frozenset({1}),
            children=(gc1,),
        )
        sol_empty_cells_no_bbox = NodeSolution(
            node_id="comp_empty",
            local_coordinates={},
            metadata={"child_sub_boxes": {}},
        )
        assert derive_child_sub_boxes(
            comp_empty_cells, sol_empty_cells_no_bbox, boundary_ports=frozenset({1})
        ) == ()

        child_carved = SolveTreeNode(node_id="c_carved", node_type="leaf", rooms=frozenset({1}))
        sol_carved = NodeSolution(
            node_id="c_carved",
            local_coordinates={1: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )
        assert derive_child_sub_boxes(child_carved, sol_carved, boundary_ports=frozenset({1})) == ()

    def test_split_pair_handler_grouping_and_metadata_poles(self) -> None:
        """SplitPairHandler groups children from metadata and container-level poles."""
        child_meta = SolveTreeNode(
            node_id="c_meta",
            node_type="leaf",
            rooms=frozenset({1, 2, 5}),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": (1, 2)},
        )
        child_invalid = SolveTreeNode(
            node_id="c_inv",
            node_type="leaf",
            rooms=frozenset({3, 4}),
            metadata={"cut_type": CutType.SPLIT_PAIR, "poles": [3]},
        )
        child_container_bound = SolveTreeNode(
            node_id="c_cb",
            node_type="leaf",
            rooms=frozenset({10, 20}),
        )
        mock_uncontracted = MagicMock(spec=SolveTreeNode)
        mock_uncontracted.node_id = "c_uncontracted"
        mock_uncontracted.cut_contract = None
        mock_uncontracted.cut_type = CutType.SPLIT_PAIR
        mock_uncontracted.metadata = {"poles": (50, 60)}
        mock_uncontracted.rooms = frozenset({50, 60})
        mock_uncontracted.children = ()

        parent = SolveTreeNode(
            node_id="parent_cont",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2, 10, 20}),
            children=(child_meta, child_invalid, child_container_bound, mock_uncontracted),
            metadata={"poles": (10, 20)},
        )
        sols = {
            "c_meta": NodeSolution("c_meta", {1: Coord3D(0, 0, 0), 2: Coord3D(2, 0, 0), 5: Coord3D(1, 0, 0)}, ()),
            "c_inv": NodeSolution("c_inv", {3: Coord3D(0, 0, 0), 4: Coord3D(1, 0, 0)}, ()),
            "c_cb": NodeSolution("c_cb", {10: Coord3D(0, 0, 0), 20: Coord3D(3, 0, 0)}, ()),
            "c_uncontracted": NodeSolution("c_uncontracted", {50: Coord3D(0, 0, 0), 60: Coord3D(1, 0, 0)}, ()),
        }
        graph = _make_graph([1, 2, 3, 4, 5, 10, 20, 50, 60], [])
        handler = SplitPairHandler(node=parent, child_solutions=sols, graph=graph)
        grouped = handler.group_split_children()

        assert (1, 2) in grouped
        assert child_meta in grouped[(1, 2)]
        assert (10, 20) in grouped
        assert child_container_bound in grouped[(10, 20)]
        assert (3, 4) not in grouped
        assert (50, 60) in grouped

    def test_split_pair_handler_corridor_coupling_and_edge_cases(self) -> None:
        """SplitPairHandler emits corridor active edges connecting parent rooms to split children."""
        child_p0 = SolveTreeNode(
            node_id="c_p0",
            node_type="leaf",
            rooms=frozenset({1, 2, 10}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        parent_p0 = SolveTreeNode(
            node_id="parent_p0",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(child_p0,),
        )
        sol_p0 = NodeSolution(
            node_id="c_p0",
            local_coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(4, 0, 0),
                10: Coord3D(0, 1, 0),
                99: Coord3D(0, 2, 0),
            },
            sub_boxes=(),
        )
        room_defs = {v: Room(vnum=v, name="", description="", exits=()) for v in [1, 2, 10, 99]}
        edges = [
            LayoutEdge(src=1, dst=10, direction=int(Direction.north), weight=1, cut=False),
            LayoutEdge(src=2, dst=99, direction=999, weight=1, cut=False),
            LayoutEdge(src=99, dst=1, direction=999, weight=1, cut=False),
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(edges))

        handler = SplitPairHandler(node=parent_p0, child_solutions={"c_p0": sol_p0}, graph=graph)
        res = handler.process()
        assert any(e.edge_id == "ext_c_p0_1_10" for e in res.active_edges)

    def test_split_pair_handler_align_children_branches(self) -> None:
        """SplitPairHandler exercises align_children branches: empty groups and external poles."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(10, 20))},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(10, 20))},
        )
        parent_no_poles = SolveTreeNode(
            node_id="parent_np",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1}),
            children=(c1, c2),
        )
        sols = {
            "c1": NodeSolution("c1", {10: Coord3D(0, 0, 0), 20: Coord3D(2, 1, 0)}, ()),
            "c2": NodeSolution("c2", {10: Coord3D(0, 0, 0), 20: Coord3D(2, 1, 0)}, ()),
        }
        graph = _make_graph([1, 10, 20], [(10, 20, Direction.east, 3)])
        handler = SplitPairHandler(node=parent_no_poles, child_solutions=sols, graph=graph)

        groups = {(10, 20): [c1, c2], (30, 40): []}
        (
            child_aligns,
            equated,
            gw_anchors,
            aligned,
        ) = handler.align_children(split_children_by_poles=groups, room_anchored_children=None)

        assert len(child_aligns) == 2
        assert frozenset({"c1", "c2"}) in equated

    def test_split_pair_handler_corridor_active_edges_edge_cases(self) -> None:
        """SplitPairHandler exercises cut edges, reverse parent-child edges, and self-loops in corridors."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({1, 2, 10}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({3, 4, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(3, 4))},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type="container",
            rooms=frozenset({1, 2, 5, 6}),
            children=(c1, c2),
        )
        sols = {
            "c1": NodeSolution("c1", {1: Coord3D(0, 0, 0), 2: Coord3D(2, 0, 0), 10: Coord3D(1, 0, 0)}, ()),
            "c2": NodeSolution("c2", {3: Coord3D(0, 0, 0), 4: Coord3D(2, 0, 0), 20: Coord3D(1, 0, 0)}, ()),
        }
        room_defs = {v: Room(vnum=v, name="", description="", exits=()) for v in [1, 2, 3, 4, 5, 6, 10, 20]}
        edges = [
            LayoutEdge(src=5, dst=10, direction=int(Direction.east), weight=1, cut=True),
            LayoutEdge(src=5, dst=10, direction=999, weight=1, cut=False),
            LayoutEdge(src=5, dst=6, direction=int(Direction.east), weight=1, cut=False),
            LayoutEdge(src=10, dst=1, direction=int(Direction.east), weight=1, cut=False),
            LayoutEdge(src=10, dst=5, direction=int(Direction.south), weight=2, cut=False),
            LayoutEdge(src=10, dst=10, direction=int(Direction.north), weight=1, cut=False),
            LayoutEdge(src=10, dst=3, direction=int(Direction.west), weight=1, cut=False),
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(edges))

        handler = SplitPairHandler(node=parent, child_solutions=sols, graph=graph)
        corridors, gw, aligned = handler.build_corridor_active_edges(
            split_children_by_poles={(1, 2): [c1], (3, 4): [c2]},
            equated_children=set(),
        )
        edge_ids = {e.edge_id for e in corridors}
        assert "ext_c1_10_5" in edge_ids
        assert "ext_c1_c2_10_3" in edge_ids

        # Equated children skip inter-split corridor construction
        corridors_eq, _, _ = handler.build_corridor_active_edges(
            split_children_by_poles={(1, 2): [c1], (3, 4): [c2]},
            equated_children={frozenset({"c1", "c2"})},
        )
        assert "ext_c1_c2_10_3" not in {e.edge_id for e in corridors_eq}

    def test_split_pair_handler_direct_sp_edges_and_process_defaults(self) -> None:
        """SplitPairHandler processes split pairs without synthetic edges, producing corridor active edges."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({1, 2, 3}),
            metadata={"cut_contract": SplitPairCut(poles=(1, 2))},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({1, 2, 10}),
            children=(c1,),
        )
        sols = {
            "c1": NodeSolution("c1", {1: Coord3D(0, 0, 0), 2: Coord3D(2, 0, 0), 3: Coord3D(1, 0, 0)}, ()),
        }
        graph = _make_graph([1, 2, 3, 10], [(10, 3, Direction.east, 2)])
        handler = SplitPairHandler(node=parent, child_solutions=sols, graph=graph)

        res = handler.process(room_anchored_children=None)
        assert res.active_edges == res.corridor_edges
        assert len(res.active_edges) == 1
        assert res.active_edges[0].edge_id == "ext_c1_10_3"

    def test_split_pair_handler_boundary_ports_empty_container(self) -> None:
        """SplitPairHandler emits sp_port and sp_span active edges when poles reside in boundary_ports of an empty container."""
        child = SolveTreeNode(
            node_id="c_split",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(10, 20))},
        )
        container = SolveTreeNode(
            node_id="container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(child,),
        )
        sol = NodeSolution("c_split", {10: Coord3D(0, 0, 0), 20: Coord3D(0, 2, 0)}, ())
        graph = _make_graph([10, 20], [(10, 20, Direction.north, 2)])
        handler = SplitPairHandler(
            node=container,
            child_solutions={"c_split": sol},
            graph=graph,
            boundary_ports=frozenset({10, 20}),
        )

        res = handler.process()
        edge_ids = {e.edge_id for e in res.active_edges}
        assert "sp_port_c_split_10" in edge_ids
        assert "sp_span_c_split_10_20" in edge_ids

        port_edge = next(e for e in res.active_edges if e.edge_id == "sp_port_c_split_10")
        assert port_edge.src == 10
        assert port_edge.dst == ("c_split", 10)
        assert port_edge.direction == (0, 0, 0)
        assert port_edge.min_length == 0

        span_edge = next(e for e in res.active_edges if e.edge_id == "sp_span_c_split_10_20")
        assert span_edge.src == 10
        assert span_edge.dst == 20
        assert span_edge.direction == (0, 1, 0)
        assert span_edge.min_length == 2

    def test_split_pair_handler_inherited_boundary_port(self) -> None:
        """SplitPairHandler couples split-pair children when one pole is an owned room and the other is an inherited boundary port."""
        child = SolveTreeNode(
            node_id="c_split",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(10, 20))},
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type="container",
            rooms=frozenset({20}),
            children=(child,),
        )
        sol = NodeSolution("c_split", {10: Coord3D(0, 0, 0), 20: Coord3D(3, 0, 0)}, ())
        graph = _make_graph([10, 20], [(10, 20, Direction.east, 3)])
        handler = SplitPairHandler(
            node=parent,
            child_solutions={"c_split": sol},
            graph=graph,
            boundary_ports=frozenset({10}),
        )

        res = handler.process()
        edge_ids = {e.edge_id for e in res.active_edges}
        assert "sp_port_c_split_10" in edge_ids
        assert "sp_span_c_split_10_20" in edge_ids

    def test_split_pair_handler_boundary_port_corridor_exit(self) -> None:
        """SplitPairHandler creates corridor active edges connecting inherited boundary ports to split child rooms."""
        child = SolveTreeNode(
            node_id="sc",
            node_type="leaf",
            rooms=frozenset({10, 20, 30}),
            metadata={"cut_contract": SplitPairCut(poles=(10, 20))},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type="container",
            rooms=frozenset({10, 20}),
            children=(child,),
        )
        sol = NodeSolution("sc", {10: Coord3D(0, 0, 0), 20: Coord3D(2, 0, 0), 30: Coord3D(0, 1, 0)}, ())
        graph = _make_graph([5, 10, 20, 30], [(5, 30, Direction.north, 1)])
        handler = SplitPairHandler(
            node=parent,
            child_solutions={"sc": sol},
            graph=graph,
            boundary_ports=frozenset({5}),
        )

        res = handler.process()
        assert any(e.edge_id == "ext_sc_5_30" for e in res.active_edges)

    def test_inventory_cluster_passes_boundary_ports_to_split_pair_handler(self) -> None:
        """inventory_cluster forwards boundary_ports down to SplitPairHandler to produce split active edges."""
        child = SolveTreeNode(
            node_id="c_split",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(10, 20))},
        )
        container = SolveTreeNode(
            node_id="container",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(child,),
        )
        sol = NodeSolution("c_split", {10: Coord3D(0, 0, 0), 20: Coord3D(0, 2, 0)}, ())
        graph = _make_graph([10, 20], [(10, 20, Direction.north, 2)])
        inv = inventory_cluster(
            node=container,
            child_solutions={"c_split": sol},
            graph=graph,
            boundary_ports=frozenset({10, 20}),
        )

        edge_ids = {e.edge_id for e in inv.active_edges}
        assert "sp_port_c_split_10" in edge_ids
        assert "sp_span_c_split_10_20" in edge_ids

    def test_gateway_cut_handler_edge_cases_and_error_handling(self) -> None:
        """GatewayCutHandler exercises metadata fallback, multiple anchors, reverse doorway edges, and invalid directions."""
        c_art1 = SolveTreeNode(
            node_id="c_art1",
            node_type="leaf",
            rooms=frozenset({1, 10}),
            metadata={"anchor_vnum": 1},
        )
        c_art2 = SolveTreeNode(
            node_id="c_art2",
            node_type="leaf",
            rooms=frozenset({1, 20}),
            metadata={"cut_type": CutType.ARTICULATION_POINT, "anchor_vnum": "1"},
        )
        c_art_none = SolveTreeNode(
            node_id="c_art_none",
            node_type="leaf",
            rooms=frozenset({1, 30}),
            metadata={"cut_type": CutType.ARTICULATION_POINT, "anchor_vnum": None},
        )
        mock_uncontracted_art = MagicMock(spec=SolveTreeNode)
        mock_uncontracted_art.node_id = "c_uncontracted_art"
        mock_uncontracted_art.cut_contract = None
        mock_uncontracted_art.cut_type = CutType.ARTICULATION_POINT
        mock_uncontracted_art.metadata = {"anchor_vnum": 1}
        mock_uncontracted_art.rooms = frozenset({1, 70})
        mock_uncontracted_art.children = ()

        c_brg_fwd = SolveTreeNode(
            node_id="c_brg_fwd",
            node_type="leaf",
            rooms=frozenset({40}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(1, 40), direction=999, weight=1)},
        )
        c_brg_rev = SolveTreeNode(
            node_id="c_brg_rev",
            node_type="leaf",
            rooms=frozenset({50}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(50, 1), direction=999, weight=1)},
        )
        c_graph_rev = SolveTreeNode(
            node_id="c_graph_rev",
            node_type="leaf",
            rooms=frozenset({60}),
        )
        parent = SolveTreeNode(
            node_id="p_gw",
            node_type="container",
            rooms=frozenset({1}),
            children=(c_art1, c_art2, c_art_none, mock_uncontracted_art, c_brg_fwd, c_brg_rev, c_graph_rev),
        )
        sols = {
            "c_art1": NodeSolution("c_art1", {1: Coord3D(0, 0, 0), 10: Coord3D(1, 0, 0)}, ()),
            "c_art2": NodeSolution("c_art2", {1: Coord3D(0, 0, 0), 20: Coord3D(0, 1, 0)}, ()),
            "c_art_none": NodeSolution("c_art_none", {1: Coord3D(0, 0, 0), 30: Coord3D(0, 0, 1)}, ()),
            "c_uncontracted_art": NodeSolution("c_uncontracted_art", {1: Coord3D(0, 0, 0), 70: Coord3D(1, 0, 0)}, ()),
            "c_brg_fwd": NodeSolution("c_brg_fwd", {40: Coord3D(0, 0, 0)}, ()),
            "c_brg_rev": NodeSolution("c_brg_rev", {50: Coord3D(0, 0, 0)}, ()),
            "c_graph_rev": NodeSolution("c_graph_rev", {60: Coord3D(0, 0, 0)}, ()),
        }
        room_defs = {v: Room(vnum=v, name="", description="", exits=()) for v in [1, 10, 20, 30, 40, 50, 60, 70]}
        edges = [
            LayoutEdge(src=1, dst=60, direction=int(Direction.north), weight=1, cut=True),
            LayoutEdge(src=1, dst=40, direction=999, weight=1, cut=False),
            LayoutEdge(src=60, dst=1, direction=999, weight=1, cut=False),
            LayoutEdge(src=60, dst=1, direction=int(Direction.south), weight=3, cut=False),
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(edges))

        handler = GatewayCutHandler(node=parent, child_solutions=sols, graph=graph)
        # Direct call to process_articulation_points with room_anchored_children=None
        art_edges, _, _, _ = handler.process_articulation_points(aligned_children=set(), room_anchored_children=None)
        assert len(art_edges) >= 1

        res = handler.process(aligned_children=None, room_anchored_children=None)
        assert frozenset({"c_art1", "c_art2"}) not in res.equated_children
        edge_ids = {e.edge_id for e in res.active_edges}
        assert "gw_c_graph_rev_60_1" in edge_ids

    def test_inter_child_bridge_handler_filtering_and_error_handling(self) -> None:
        """InterChildBridgeHandler exercises equated filtering, unaligned target skipping, and invalid directions."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(10, 20), direction=999, weight=1)},
        )
        c3 = SolveTreeNode(
            node_id="c3",
            node_type="leaf",
            rooms=frozenset({30}),
            metadata={"cut_contract": BridgeCut(bridge_edge=(30, 10), direction=999, weight=1)},
        )
        c4 = SolveTreeNode(node_id="c4", node_type="leaf", rooms=frozenset({40}))
        parent = SolveTreeNode(
            node_id="p_icb",
            node_type="container",
            rooms=frozenset(),
            children=(c1, c2, c3, c4),
        )
        sols = {
            "c1": NodeSolution("c1", {10: Coord3D(0, 0, 0)}, ()),
            "c2": NodeSolution("c2", {20: Coord3D(0, 0, 0)}, ()),
            "c3": NodeSolution("c3", {30: Coord3D(0, 0, 0)}, ()),
            "c4": NodeSolution("c4", {40: Coord3D(0, 0, 0)}, ()),
        }
        room_defs = {v: Room(vnum=v, name="", description="", exits=()) for v in [10, 20, 30, 40]}
        edges = [
            LayoutEdge(src=10, dst=20, direction=int(Direction.east), weight=1, cut=True),
            LayoutEdge(src=10, dst=40, direction=999, weight=1, cut=False),
            LayoutEdge(src=40, dst=10, direction=999, weight=1, cut=False),
            LayoutEdge(src=10, dst=40, direction=int(Direction.east), weight=2, cut=False),
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(edges))

        handler_eq = InterChildBridgeHandler(node=parent, child_solutions=sols, graph=graph)
        res_eq = handler_eq.process(
            aligned_children={"c1"},
            equated_children={frozenset({"c1", "c4"}), frozenset({"c1", "c2"}), frozenset({"c1", "c3"})},
        )
        assert len(res_eq.active_edges) == 0

        handler = InterChildBridgeHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process(aligned_children={"c1"}, equated_children=set())
        assert len(res.active_edges) == 1
        assert res.active_edges[0].edge_id == "ccb_c1_c4_10_40"
        assert "c4" in res.aligned_children

    def test_build_internal_room_edges_invalid_directions_and_empty_rooms(self) -> None:
        """Internal room edge builder handles empty rooms, self-loops, and invalid directions."""
        assert _build_internal_room_edges(
            graph=_make_graph([], []),
            rooms=(),
        ) == []

        room_defs = {v: Room(vnum=v, name="", description="", exits=()) for v in [1, 2, 3]}
        edges = [
            LayoutEdge(src=1, dst=1, direction=int(Direction.east), weight=1, cut=False),
            LayoutEdge(src=1, dst=2, direction=999, weight=1, cut=False),
            LayoutEdge(src=1, dst=2, direction=int(Direction.east), weight=1, cut=False),
            LayoutEdge(src=2, dst=3, direction=999, weight=1, cut=False),
            LayoutEdge(src=1, dst=3, direction=int(Direction.north), weight=1, cut=False),
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(edges))
        active = _build_internal_room_edges(
            graph=graph,
            rooms=(1, 2, 3),
        )
        assert len(active) == 2
        edge_ids = {e.edge_id for e in active}
        assert edge_ids == {"int_1_2", "int_1_3"}

    def test_build_internal_room_edges_deduplicates_bidirectional_pairs(self) -> None:
        """Stage 1 inventory deduplicates reciprocal internal room edges into a single ActiveEdge."""
        room_defs = {v: Room(vnum=v, name="", description="", exits=()) for v in [1, 2]}
        edges = [
            LayoutEdge(src=1, dst=2, direction=int(Direction.east), weight=1, cut=False),
            LayoutEdge(src=2, dst=1, direction=int(Direction.west), weight=1, cut=False),
        ]
        graph = LayoutGraph(rooms=room_defs, edges=tuple(edges))
        active = _build_internal_room_edges(
            graph=graph,
            rooms=(1, 2),
        )
        assert len(active) == 1
        assert active[0].edge_id == "int_1_2"
        assert active[0].associated_rooms == frozenset({1, 2})

    def test_inventory_cluster_missing_child_solution_and_empty_sub_boxes(self) -> None:
        """Stage 1 inventory assigns empty sub-boxes tuple when child cluster lacks a solution."""
        child_solved = SolveTreeNode(node_id="c_solved", node_type="leaf", rooms=frozenset({1}))
        child_unsolved = SolveTreeNode(node_id="c_unsolved", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(
            node_id="p",
            node_type="container",
            rooms=frozenset({10}),
            children=(child_solved, child_unsolved),
        )
        sol_solved = NodeSolution("c_solved", {1: Coord3D(0, 0, 0)}, ())
        graph = _make_graph([1, 2, 10], [])

        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_solved": sol_solved},
            graph=graph,
        )
        assert inv.child_sub_boxes["c_unsolved"] == ()
        assert "c_solved" in inv.child_sub_boxes


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
        assert formulation.m_coord == 50
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

        # Objective exists and redundant bbox variables are eliminated
        assert hasattr(model, "obj")
        assert not hasattr(model, "x_min")
        assert not hasattr(model, "x_max")
        assert not hasattr(model, "bbox_bounds")

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

    def test_dynamic_bounds_small_leaf_hits_floor(self) -> None:
        """Small leaf clusters with few rooms and edges resolve to the baseline floor of 50."""
        node = SolveTreeNode(node_id="leaf_3", node_type="leaf", rooms=frozenset({1, 2, 3}))
        graph = _make_graph([1, 2, 3], [(1, 2, Direction.east, 1), (2, 3, Direction.north, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)

        m_coord = calculate_dynamic_bounds(inv)
        assert m_coord == 50

        formulation = ClusterFormulation(inventory=inv)
        assert formulation.m_coord == 50
        assert formulation.m_big == 100

    def test_dynamic_bounds_empty_inventory_floor(self) -> None:
        """Empty inventory with no rooms, children, or edges safely hits the baseline floor."""
        node = SolveTreeNode(node_id="empty_node", node_type="leaf", rooms=frozenset())
        inv = ClusterInventory(
            node=node,
            rooms=(),
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )

        assert calculate_dynamic_bounds(inv) == 50
        formulation = ClusterFormulation(inventory=inv)
        assert formulation.m_coord == 50
        assert formulation.m_big == 100

    def test_dynamic_bounds_moderate_leaf_proportional_scaling(self) -> None:
        """Moderate leaf clusters scale proportionally to room count and active edge lengths."""
        rooms = list(range(1, 11))
        edges = [(i, i + 1, Direction.east, 2) for i in range(1, 10)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="leaf_10", node_type="leaf", rooms=frozenset(rooms))
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)

        # 10 rooms -> room_scale = 100
        # 9 edges with min_length 2 -> edge_min_sum = 18 -> edge_scale = 36
        # child_spans = 0
        # total_scale = 136
        m_coord = calculate_dynamic_bounds(inv)
        assert m_coord == 136

        formulation = ClusterFormulation(inventory=inv)
        assert formulation.m_coord == 136
        assert formulation.m_big == 272

    def test_dynamic_bounds_composite_with_child_solutions(self) -> None:
        """Composite clusters derive bounds from child solution bounding boxes and port spans."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(
            node_id="p_comp",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 10, 0, 10, 0, 1),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 5, 0, 8, 0, 2),),
        )

        graph = _make_graph([1, 2, 10, 20], [(1, 2, Direction.east, 3)])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )

        m_coord = calculate_dynamic_bounds(inv)
        assert m_coord > 50
        formulation = ClusterFormulation(inventory=inv)
        assert formulation.m_coord == m_coord
        assert formulation.m_big == 2 * m_coord

    def test_dynamic_bounds_composite_with_keepout_sub_boxes_only(self) -> None:
        """Composite clusters without pre-solved solutions inspect keep-out sub-boxes directly."""
        node = SolveTreeNode(node_id="p_keepout", node_type="container", rooms=frozenset({1}))
        inv = ClusterInventory(
            node=node,
            rooms=(1,),
            children=(SolveTreeNode(node_id="c_unsolv", node_type="leaf", rooms=frozenset({2})),),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            child_sub_boxes={
                "c_unsolv": (
                    BoundingBox3D(0, 20, 0, 20, 0, 5),
                    (Coord3D(0, 0, 0), Coord3D(25, 25, 5)),
                ),
            },
        )

        # Enclosing box dimensions: (26 + 26 + 6) = 58 span, max_offset 25 -> max(58, 50) = 58.
        # Room scale: 1 * 10 = 10.
        # Total scale = 68.
        m_coord = calculate_dynamic_bounds(inv)
        assert m_coord == 68
        assert ClusterFormulation(inventory=inv).m_big == 136

    def test_dynamic_bounds_unresolved_child_fallback(self) -> None:
        """Unresolved child nodes lacking both solutions and sub-boxes receive a safe fallback span."""
        child = SolveTreeNode(node_id="c_raw", node_type="leaf", rooms=frozenset({5}))
        parent = SolveTreeNode(node_id="p_raw", node_type="container", rooms=frozenset({1}), children=(child,))
        inv = ClusterInventory(
            node=parent,
            rooms=(1,),
            children=(child,),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )

        # Room scale: 10, Child fallback: 20 -> total scale = 30 -> baseline floor 50.
        assert calculate_dynamic_bounds(inv) == 50

    def test_dynamic_bounds_large_edge_min_length(self) -> None:
        """Active edges with large min_length ensure coordinate bound exceeds edge length with headroom."""
        node = SolveTreeNode(node_id="leaf_long", node_type="leaf", rooms=frozenset({1, 2}))
        ae = ActiveEdge(
            edge_id="e_long",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=80,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
        )
        inv = ClusterInventory(
            node=node,
            rooms=(1, 2),
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(ae,),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )

        # 2 rooms = 20, edge min_length 80 -> edge_scale = 160 -> total scale = 180.
        m_coord = calculate_dynamic_bounds(inv)
        assert m_coord == 180
        assert m_coord >= 2 * ae.min_length

    def test_dynamic_bounds_explicit_override(self) -> None:
        """Explicit coordinate bound overrides dynamic calculation in constructor and factory."""
        node = SolveTreeNode(node_id="leaf_ovr", node_type="leaf", rooms=frozenset({1, 2}))
        inv = inventory_cluster(node=node, child_solutions={}, graph=_make_graph([1, 2], []))

        formulation = ClusterFormulation(inventory=inv, m_coord=500)
        assert formulation.m_coord == 500
        assert formulation.m_big == 1000

        factory_formulation = formulate_cluster_model(inventory=inv, m_coord=750)
        assert factory_formulation.m_coord == 750
        assert factory_formulation.m_big == 1500

    def test_dynamic_bounds_mathematical_soundness(self) -> None:
        """Disjunctive separation Big-M relation holds strictly for all extreme coordinate values."""
        node = SolveTreeNode(node_id="leaf_sound", node_type="leaf", rooms=frozenset({1, 2}))
        inv = inventory_cluster(node=node, child_solutions={}, graph=_make_graph([1, 2], []))
        formulation = ClusterFormulation(inventory=inv)

        m_coord = formulation.m_coord
        m_big = formulation.m_big

        # Any coordinates min_xa, max_xb in [-m_coord, m_coord]
        # Worst-case separation when br = 0: min_xa = -m_coord, max_xb = m_coord
        min_xa_worst = -m_coord
        max_xb_worst = m_coord
        diff = min_xa_worst - max_xb_worst  # -2 * m_coord
        rhs = 1 - 2 * m_big  # 1 - 4 * m_coord

        # Constraint is min_xa - max_xb >= 1 - 2 * m_big (i.e. diff - rhs >= 0)
        slack = diff - rhs
        assert slack > 0
        assert slack == 2 * m_coord - 1

    def test_quadratic_transverse_penalty_weight_calculation(self) -> None:
        """Transverse slack penalty scales quadratically based on inventory active edge minimum length sum."""
        node = SolveTreeNode(node_id="leaf_w", node_type="leaf", rooms=frozenset({1, 2}))

        # 1. Empty active edges: falls back to baseline floor M=50, W=10 * 50^2 = 25000
        inv_empty = ClusterInventory(
            node=node,
            rooms=(1, 2),
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )
        formulation_empty = ClusterFormulation(inventory=inv_empty)
        assert formulation_empty.m_scale == 50
        assert formulation_empty.w_transverse == 25000.0

        # 2. Sum of min_lengths < 50: still clamps to floor M=50, W=25000
        ae1 = ActiveEdge(
            edge_id="e1",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=30,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
        )
        inv_small = ClusterInventory(
            node=node,
            rooms=(1, 2),
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(ae1,),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )
        formulation_small = ClusterFormulation(inventory=inv_small)
        assert formulation_small.m_scale == 50
        assert formulation_small.w_transverse == 25000.0

        # 3. Sum of min_lengths > 50: M = sum(min_lengths), W = 10 * M^2
        ae2 = ActiveEdge(
            edge_id="e2",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=70,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
        )
        inv_large = ClusterInventory(
            node=node,
            rooms=(1, 2),
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(ae2,),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )
        formulation_large = ClusterFormulation(inventory=inv_large)
        assert formulation_large.m_scale == 70
        assert formulation_large.w_transverse == 10.0 * (70**2)
        assert formulation_large.w_transverse == 49000.0

    def test_quadratic_transverse_penalty_explicit_override(self) -> None:
        """Explicit w_transverse parameter overrides quadratic calculation in constructor and factory."""
        node = SolveTreeNode(node_id="leaf_w_ovr", node_type="leaf", rooms=frozenset({1, 2}))
        inv = inventory_cluster(node=node, child_solutions={}, graph=_make_graph([1, 2], []))

        formulation = ClusterFormulation(inventory=inv, w_transverse=1234.0)
        assert formulation.w_transverse == 1234.0

        factory_formulation = formulate_cluster_model(inventory=inv, w_transverse=5678.0)
        assert factory_formulation.w_transverse == 5678.0

    def test_piecewise_linear_convex_stretch_buckets(self) -> None:
        """Convex piecewise stretch decomposes edge elongation into three progressive linear buckets."""
        node = SolveTreeNode(node_id="leaf_stretch", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)
        model = formulation.model

        assert hasattr(model, "s_stretch_1")
        assert hasattr(model, "s_stretch_2")
        assert hasattr(model, "s_stretch_3")
        assert hasattr(model, "stretch_decompose")

        # 1. Delta L = 0 (min_length = 1, L = 1): all buckets zero
        model.L[0].fix(1)
        solve_and_extract(formulation, options=LayoutOptions())
        assert pyo.value(model.s_stretch_1[0]) == pytest.approx(0.0)
        assert pyo.value(model.s_stretch_2[0]) == pytest.approx(0.0)
        assert pyo.value(model.s_stretch_3[0]) == pytest.approx(0.0)

        # 2. Delta L = 2 (L = 3): mild stretch fills Bucket 1 (capacity 2, weight 10)
        model.L[0].fix(3)
        solve_and_extract(formulation, options=LayoutOptions())
        assert pyo.value(model.s_stretch_1[0]) == pytest.approx(2.0)
        assert pyo.value(model.s_stretch_2[0]) == pytest.approx(0.0)
        assert pyo.value(model.s_stretch_3[0]) == pytest.approx(0.0)

        # 3. Delta L = 5 (L = 6): spills over into Bucket 2 (capacity 4, weight 30)
        model.L[0].fix(6)
        solve_and_extract(formulation, options=LayoutOptions())
        assert pyo.value(model.s_stretch_1[0]) == pytest.approx(2.0)
        assert pyo.value(model.s_stretch_2[0]) == pytest.approx(3.0)
        assert pyo.value(model.s_stretch_3[0]) == pytest.approx(0.0)

        # 4. Delta L = 8 (L = 9): spills over into Bucket 3 (unbounded, weight 100)
        model.L[0].fix(9)
        solve_and_extract(formulation, options=LayoutOptions())
        assert pyo.value(model.s_stretch_1[0]) == pytest.approx(2.0)
        assert pyo.value(model.s_stretch_2[0]) == pytest.approx(4.0)
        assert pyo.value(model.s_stretch_3[0]) == pytest.approx(2.0)

    def test_piecewise_stretch_empty_active_edges(self) -> None:
        """Formulation with no active edges creates empty indexed stretch variables without error."""
        node = SolveTreeNode(node_id="leaf_no_edges", node_type="leaf", rooms=frozenset({1}))
        inv = inventory_cluster(node=node, child_solutions={}, graph=_make_graph([1], []))
        formulation = ClusterFormulation(inventory=inv)
        assert len(formulation.model.s_stretch_1) == 0
        assert len(formulation.model.s_stretch_2) == 0
        assert len(formulation.model.s_stretch_3) == 0


class TestStage3Clearance:
    """Unit tests for Stage 3: Geometric Clearance & Separation Constraints."""

    def test_apply_clearance_constraints_parent_rooms(self) -> None:
        """Room-room non-overlap separation constraints are attached with prefix 'parent_room'."""
        node = SolveTreeNode(node_id="p_rooms", node_type="block", rooms=frozenset({1, 2, 3}))
        graph = _make_graph([1, 2, 3], [(1, 2, Direction.east, 1), (2, 3, Direction.north, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        formulation.apply_clearances()
        model = formulation.model

        # Parent-room separation variables and constraints
        assert hasattr(model, "parent_room_br")
        assert hasattr(model, "parent_room_sep_sum")
        assert len(model.parent_room_sep_sum) == 3  # (1,2), (1,3), (2,3)

    def test_formulation_apply_clearances_member_method(self) -> None:
        """Direct invocation of formulation.apply_clearances() attaches clearance constraints."""
        node = SolveTreeNode(node_id="p_member", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        assert not hasattr(formulation.model, "parent_room_sep_sum")
        formulation.apply_clearances()
        assert hasattr(formulation.model, "parent_room_sep_sum")

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

        formulation.apply_clearances()
        model = formulation.model

        assert hasattr(model, "room_child_br")  # Room vs child separation
        assert hasattr(model, "corridor_room_br")  # Corridor vs room separation

    def test_apply_clearances_child_child_separation(self) -> None:
        """Pairwise child-child sub-box separation is attached for child clusters."""
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
        assert hasattr(formulation.model, "child_child_br")
        assert hasattr(formulation.model, "child_child_sep_sum")

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
        assert hasattr(formulation.model, "corridor_child_br")  # Corridor vs sibling child
        assert hasattr(formulation.model, "corridor_crossing_br")   # Corridor crossing

    def test_clearance_constraint_builder_direct(self) -> None:
        """ClearanceConstraintBuilder can be instantiated and executed directly."""
        node = SolveTreeNode(node_id="p_builder", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        builder = ClearanceConstraintBuilder(formulation)
        builder.build()
        assert hasattr(formulation.model, "parent_room_sep_sum")

    def test_individual_clearance_builder_methods(self) -> None:
        """Individual clearance separation methods on ClearanceConstraintBuilder execute without error."""
        node = SolveTreeNode(node_id="p_indiv", node_type="block", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [(1, 2, Direction.east, 1)])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = ClusterFormulation(inventory=inv)

        builder = formulation.clearance_builder
        builder.add_parent_room_separation()
        builder.add_child_child_separation()
        builder.add_room_child_separation()
        builder.add_corridor_room_separation()
        builder.add_corridor_child_separation()
        builder.add_corridor_crossing_separation()

        assert hasattr(formulation.model, "parent_room_sep_sum")

    def test_add_corridor_filtering_branches(self) -> None:
        """add_corridor_room_separation and add_corridor_child_separation handle skipped branches."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p_flt", node_type="container", rooms=frozenset({1, 2, 3}), children=(c1,))
        sol1 = NodeSolution("c1", {10: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),))
        graph = _make_graph([1, 2, 3, 10], [])

        import dataclasses
        inv = inventory_cluster(node=parent, child_solutions={"c1": sol1}, graph=graph)
        inv = dataclasses.replace(
            inv,
            active_edges=(
                ActiveEdge("sp_virtual", 1, 2, (1, 0, 0), 2, frozenset({1, 2}), frozenset({"c1"})),
                ActiveEdge("e_diag", 1, 2, (1, 1, 0), 2, frozenset({1, 2}), frozenset()),
                ActiveEdge("e_ortho", 1, 2, (1, 0, 0), 2, frozenset({1, 2}), frozenset({"c1"})),
            ),
        )
        formulation = ClusterFormulation(inventory=inv)
        builder = formulation.clearance_builder
        builder.add_corridor_room_separation()
        builder.add_corridor_child_separation()
        builder.add_corridor_crossing_separation()


class TestStage3DelayedCutsAndCollisionDetector:
    """Unit tests for Stage 3 geometric collision detection and delayed cutting-plane injection."""

    def test_detect_and_inject_room_room_collision(self) -> None:
        """Room-room collision detected at identical coordinates and separated by injected cut."""
        node = SolveTreeNode(node_id="p_rr", node_type="leaf", rooms=frozenset({1, 2}))
        graph = _make_graph([1, 2], [])
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Extraction with identical coordinates for room 1 and 2
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (0, 0, 0)},
            raw_child_offsets={},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        assert len(violations) == 1
        assert violations[0].category == "parent_room"
        assert violations[0].item_a == 1
        assert violations[0].item_b == 2

        # Inject cut
        builder = formulation.clearance_builder
        assert not hasattr(formulation.model, "parent_room_br")
        cuts_added = builder.inject_clearance_cuts(violations)
        assert cuts_added == 1
        assert hasattr(formulation.model, "parent_room_br")

        # Deduplication check: injecting the same violation again adds 0 cuts
        cuts_added_again = builder.inject_clearance_cuts(violations)
        assert cuts_added_again == 0

    def test_detect_and_inject_room_child_collision(self) -> None:
        """Room penetrating child sub-box is detected and separated."""
        child = SolveTreeNode(node_id="c_box", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p_rc", node_type="container", rooms=frozenset({1}), children=(child,))
        sol = NodeSolution(
            node_id="c_box",
            local_coordinates={10: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(0, 2, 0, 2, 0, 0),),
        )
        graph = _make_graph([1, 10], [])
        inv = inventory_cluster(node=parent, child_solutions={"c_box": sol}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Room 1 at (1, 1, 0) penetrates child box at offset (0, 0, 0)
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (1, 1, 0)},
            raw_child_offsets={"c_box": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        assert len(violations) == 1
        assert violations[0].category == "room_child"
        assert violations[0].item_a == 1
        assert violations[0].item_b == ("c_box", 0)

        cuts_added = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_added == 1
        assert hasattr(formulation.model, "room_child_br")

    def test_detect_and_inject_child_child_collision(self) -> None:
        """Overlapping child sub-boxes are detected and separated."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}))
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({20}))
        parent = SolveTreeNode(node_id="p_cc", node_type="container", rooms=frozenset(), children=(c1, c2))
        sol1 = NodeSolution("c1", {10: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 2, 0, 2, 0, 0),))
        sol2 = NodeSolution("c2", {20: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 2, 0, 2, 0, 0),))
        graph = _make_graph([10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c1": sol1, "c2": sol2}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Offsets overlap at (0, 0, 0) and (1, 1, 0)
        extraction = SolvedClusterExtraction(
            raw_parent_coords={},
            raw_child_offsets={"c1": (0, 0, 0), "c2": (1, 1, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        assert any(v.category == "child_child" for v in violations)

        cuts_added = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_added >= 1
        assert hasattr(formulation.model, "child_child_br")

    def test_detect_and_inject_corridor_room_collision(self) -> None:
        """Corridor interior penetrating third-party room is detected and separated."""
        rooms = [1, 2, 3]
        edges = [(1, 2, Direction.east, 4)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="p_cr", node_type="leaf", rooms=frozenset(rooms))
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Room 1 at (0, 0, 0), Room 2 at (4, 0, 0), Room 3 placed right on the corridor interior at (2, 0, 0)
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (4, 0, 0), 3: (2, 0, 0)},
            raw_child_offsets={},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        assert any(v.category == "corridor_room" and v.item_b == 3 for v in violations)

        cuts_added = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_added >= 1
        assert hasattr(formulation.model, "corridor_room_br")

    def test_detect_and_inject_corridor_child_collision(self) -> None:
        """Corridor interior penetrating sibling child sub-box is detected and separated."""
        child = SolveTreeNode(node_id="c_sib", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p_cchild", node_type="container", rooms=frozenset({1, 2}), children=(child,))
        sol = NodeSolution("c_sib", {10: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 2, 0, 2, 0, 0),))
        graph = _make_graph([1, 2, 10], [(1, 2, Direction.east, 4)])
        inv = inventory_cluster(node=parent, child_solutions={"c_sib": sol}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Corridor 1->2 runs (0, 1, 0) -> (4, 1, 0). Child placed at (1, 0, 0) so box spans (1..3, 0..2, 0..0)
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 1, 0), 2: (4, 1, 0)},
            raw_child_offsets={"c_sib": (1, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        assert any(v.category == "corridor_child" for v in violations)

        cuts_added = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_added >= 1
        assert hasattr(formulation.model, "corridor_child_br")

    def test_detect_and_inject_corridor_corridor_crossing(self) -> None:
        """Crossing orthogonal corridor segments in 3D are detected and separated."""
        rooms = [1, 2, 3, 4]
        edges = [
            (1, 2, Direction.east, 4),
            (3, 4, Direction.north, 4),
        ]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="p_crossing", node_type="leaf", rooms=frozenset(rooms))
        inv = inventory_cluster(node=node, child_solutions={}, graph=graph)
        formulation = formulate_cluster_model(inventory=inv)

        # Seg 1: (0, 2, 0) -> (4, 2, 0), Seg 2: (2, 0, 0) -> (2, 4, 0). They cross at (2, 2, 0).
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 2, 0), 2: (4, 2, 0), 3: (2, 0, 0), 4: (2, 4, 0)},
            raw_child_offsets={},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        assert any(v.category == "corridor_crossing" for v in violations)

        cuts_added = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_added >= 1
        assert hasattr(formulation.model, "corridor_crossing_br")

    def test_solve_cluster_iterative_clean_single_iteration(self) -> None:
        """A simple non-colliding cluster solves in a single iteration without any Big-M binary cuts."""
        rooms = [1, 2, 3]
        edges = [(1, 2, Direction.east, 1), (2, 3, Direction.east, 1)]
        graph = _make_graph(rooms, edges)
        node = SolveTreeNode(node_id="p_clean", node_type="leaf", rooms=frozenset(rooms))

        sol = solve_cluster(node, {}, graph)
        assert sol.local_coordinates[1] == Coord3D(0, 0, 0)
        assert sol.local_coordinates[2] == Coord3D(1, 0, 0)
        assert sol.local_coordinates[3] == Coord3D(2, 0, 0)

    def test_solve_cluster_max_iterations_exceeded_error(self) -> None:
        """Iterative solver raises LayoutSolverError when collisions cannot be resolved in max_iterations."""
        rooms = [1, 2]
        graph = _make_graph(rooms, [])
        node = SolveTreeNode(node_id="p_loop", node_type="leaf", rooms=frozenset(rooms))

        mock_violation = ClearanceViolation(category="parent_room", item_a=1, item_b=2)
        with patch(
            "romutil.layout.milp.cluster.detect_cluster_collisions",
            return_value=[mock_violation],
        ), patch.object(
            ClearanceConstraintBuilder, "inject_clearance_cuts", return_value=1
        ):
            with pytest.raises(
                LayoutSolverError,
                match=r"failed to resolve all geometric collisions within \d+ iterations",
            ):
                solve_cluster(node, {}, graph)

    def test_solve_cluster_unresolvable_cuts_error(self) -> None:
        """Iterative solver raises LayoutSolverError when collisions exist but 0 cuts can be added."""
        rooms = [1, 2]
        graph = _make_graph(rooms, [])
        node = SolveTreeNode(node_id="p_nocut", node_type="leaf", rooms=frozenset(rooms))

        mock_violation = ClearanceViolation(category="parent_room", item_a=1, item_b=2)
        with patch(
            "romutil.layout.milp.cluster.detect_cluster_collisions",
            return_value=[mock_violation],
        ):
            with patch.object(ClearanceConstraintBuilder, "inject_clearance_cuts", return_value=0):
                with pytest.raises(LayoutSolverError, match="encountered unresolvable geometric collisions"):
                    solve_cluster(node, {}, graph)

    def test_extract_entity_coord_port_and_child_and_keyerror(self) -> None:
        """_extract_entity_coord resolves boundary port, child coordinate, and raises KeyError on unknown."""
        from romutil.layout.milp.clearance import _extract_entity_coord

        c_sol = NodeSolution("c1", local_coordinates={10: Coord3D(2, 3, 4)}, sub_boxes=())
        node = SolveTreeNode(node_id="p", node_type="leaf", rooms=frozenset({1}))
        inv = ClusterInventory(
            node=node,
            rooms=(1,),
            children=(),
            child_solutions={"c1": c_sol},
            boundary_ports=frozenset({2}),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )

        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0)},
            raw_child_offsets={"c1": (10, 20, 30)},
            solver_status=SolverStatus.OPTIMAL,
            raw_port_coords={2: (5, 6, 7)},
        )

        # Boundary port
        assert _extract_entity_coord(2, ext, inv) == (5, 6, 7)
        # Child endpoint
        assert _extract_entity_coord(("c1", 10), ext, inv) == (12, 23, 34)
        # Unknown room
        with pytest.raises(KeyError, match="not found in solved extraction"):
            _extract_entity_coord(999, ext, inv)

    def test_detect_corridor_child_negative_directions_and_l1_skip(self) -> None:
        """Corridor-child collision detection handles negative X, Y, Z directions and skips unit-length corridors."""
        child = SolveTreeNode(node_id="c_neg", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p_neg", node_type="container", rooms=frozenset({1, 2, 3, 4}), children=(child,))
        sol = NodeSolution("c_neg", {10: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 2, 0, 2, 0, 2),))

        # West direction: src=(4, 1, 1), dst=(0, 1, 1) -> dx = -1
        # South direction: src=(1, 4, 1), dst=(1, 0, 1) -> dy = -1
        # Down direction: src=(1, 1, 4), dst=(1, 1, 0) -> dz = -1
        # Unit length edge: src=(1, 1, 0), dst=(2, 1, 0) -> L=1
        inv = ClusterInventory(
            node=parent,
            rooms=(1, 2, 3, 4),
            children=(child,),
            child_solutions={"c_neg": sol},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("e_west", 1, 2, (-1, 0, 0), 4, frozenset({1, 2}), frozenset()),
                ActiveEdge("e_south", 1, 3, (0, -1, 0), 4, frozenset({1, 3}), frozenset()),
                ActiveEdge("e_down", 1, 4, (0, 0, -1), 4, frozenset({1, 4}), frozenset()),
                ActiveEdge("e_unit", 1, 2, (1, 0, 0), 1, frozenset({1, 2}), frozenset()),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            child_sub_boxes={"c_neg": (BoundingBox3D(0, 2, 0, 2, 0, 2),)},
        )
        formulation = ClusterFormulation(inventory=inv)

        ext = SolvedClusterExtraction(
            raw_parent_coords={
                1: (4, 4, 4),
                2: (0, 4, 4),
                3: (4, 0, 4),
                4: (4, 4, 0),
            },
            raw_child_offsets={"c_neg": (2, 2, 2)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, ext)
        # Should not crash on any negative direction calculations
        assert isinstance(violations, list)

    def test_detect_corridor_crossing_filtering_branches(self) -> None:
        """Parallel corridors and incident corridors are skipped in corridor crossing detection."""
        rooms = [1, 2, 3, 4]
        inv = ClusterInventory(
            node=SolveTreeNode(node_id="p_flt", node_type="leaf", rooms=frozenset(rooms)),
            rooms=(1, 2, 3, 4),
            children=(),
            child_solutions={},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("e_parallel1", 1, 2, (1, 0, 0), 2, frozenset({1, 2}), frozenset()),
                ActiveEdge("e_parallel2", 3, 4, (1, 0, 0), 2, frozenset({3, 4}), frozenset()),
                ActiveEdge("e_incident", 2, 3, (0, 1, 0), 2, frozenset({2, 3}), frozenset()),
                ActiveEdge("sp_virtual", 1, 4, (0, 1, 0), 2, frozenset({1, 4}), frozenset()),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )
        formulation = ClusterFormulation(inventory=inv)

        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (2, 0, 0), 3: (0, 2, 0), 4: (2, 2, 0)},
            raw_child_offsets={},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, ext)
        crossing_viols = [v for v in violations if v.category == "corridor_crossing"]
        assert len(crossing_viols) == 0

    def test_inject_clearance_cuts_empty_violations(self) -> None:
        """inject_clearance_cuts returns 0 immediately when violations list is empty."""
        node = SolveTreeNode(node_id="p_empty", node_type="leaf", rooms=frozenset({1}))
        inv = inventory_cluster(node=node, child_solutions={}, graph=_make_graph([1], []))
        formulation = ClusterFormulation(inventory=inv)
        assert formulation.clearance_builder.inject_clearance_cuts([]) == 0

    def test_detect_cluster_collisions_missing_coordinates_and_keyerror(self) -> None:
        """detect_cluster_collisions gracefully handles missing extraction coordinates and unknown endpoints."""
        child = SolveTreeNode(node_id="c_miss", node_type="leaf", rooms=frozenset({10}))
        parent = SolveTreeNode(node_id="p_miss", node_type="container", rooms=frozenset({1, 2}), children=(child,))
        sol = NodeSolution("c_miss", {10: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),))
        base_graph = _make_graph([1, 2, 10, 99], [(1, 2, Direction.east, 2)])
        room_defs = dict(base_graph.rooms)
        room_defs[1] = Room(vnum=1, name="", description="", exits=(Exit(direction=999, dst_vnum=99),))
        graph = LayoutGraph(rooms=room_defs, edges=base_graph.edges)

        import dataclasses
        inv = inventory_cluster(node=parent, child_solutions={"c_miss": sol}, graph=graph)
        inv = dataclasses.replace(
            inv,
            boundary_ports=frozenset({1, 999}),  # 999 not in graph.rooms
        )
        formulation = ClusterFormulation(inventory=inv)

        # Extraction missing room 2 and child c_miss: _extract_entity_coord(2, ext, inv) will raise KeyError
        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0)},
            raw_child_offsets={},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations = detect_cluster_collisions(formulation, ext)
        assert isinstance(violations, list)

    def test_extract_entity_coord_with_port_metadata(self) -> None:
        """_extract_entity_coord uses port_coordinates metadata when available."""
        from romutil.layout.milp.clearance import _extract_entity_coord

        c_sol = NodeSolution(
            "c_meta",
            local_coordinates={},
            sub_boxes=(),
            port_coordinates={10: Coord3D(5, 5, 5)},
            metadata={"port_coordinates": {20: Coord3D(1, 2, 3)}},
        )
        node = SolveTreeNode(node_id="p", node_type="leaf", rooms=frozenset({1}))
        inv = ClusterInventory(
            node=node,
            rooms=(1,),
            children=(),
            child_solutions={"c_meta": c_sol},
            boundary_ports=frozenset(),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
        )
        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0)},
            raw_child_offsets={"c_meta": (1, 1, 1)},
            solver_status=SolverStatus.OPTIMAL,
        )
        assert _extract_entity_coord(("c_meta", 10), ext, inv) == (6, 6, 6)
        assert _extract_entity_coord(("c_meta", 20), ext, inv) == (2, 3, 4)


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

    def test_solve_and_extract_max_solutions_and_stagnation_options(self) -> None:
        """max_solutions sets solver option and Pyomo aborted/other status yields FEASIBLE."""
        child = SolveTreeNode(node_id="c_ms", node_type="leaf", rooms=frozenset({1}))
        parent = SolveTreeNode(node_id="p_ms", node_type="container", rooms=frozenset(), children=(child,))
        inv = inventory_cluster(node=parent, child_solutions={"c_ms": NodeSolution("c_ms", {}, ())}, graph=_make_graph([1], []))
        formulation = formulate_cluster_model(inventory=inv)

        mock_solver = MagicMock()
        mock_solver.available.return_value = True
        mock_solver.options = {}
        mock_results = MagicMock()
        mock_results.solver.status = PyomoSolverStatus.aborted
        mock_results.solver.termination_condition = TerminationCondition.other
        mock_solver.solve.return_value = mock_results

        opts = LayoutOptions(timeout_seconds=20)
        with patch("pyomo.environ.SolverFactory", return_value=mock_solver):
            with patch("pyomo.environ.value", side_effect=[1, 2, 3]):
                ext = solve_and_extract(
                    formulation=formulation,
                    solver_name="cbc",
                    options=opts,
                    max_solutions=1,
                    solver_options={"heuristics": "on", "ratioGap": 0.05},
                )
                assert ext.solver_status == SolverStatus.FEASIBLE
                assert ext.raw_child_offsets["c_ms"] == (1, 2, 3)
                assert mock_solver.options.get("maxSolutions") == 1
                assert mock_solver.options.get("heuristics") == "on"
                assert mock_solver.options.get("ratioGap") == 0.05
                assert mock_solver.options.get("seconds") == 20


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
        assert res.active_edges == res.corridor_edges
        assert len(res.active_edges) == 4
        assert {"c1", "c2"}.issubset(res.aligned_children)

    def test_split_pair_handler_corridor_exit_coupling(self) -> None:
        """Split pairs with exit edges connecting to parent rooms emit corridor active edges."""
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
        assert any(e.edge_id == "ext_c_trans_2_10" for e in res.active_edges)

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
        assert len(res.child_alignments) == 2
        assert res.child_alignments[0].base_child_id == "c1"
        assert res.child_alignments[0].target_child_id == "c2"
        assert res.child_alignments[0].base_coord == Coord3D(0, 0, 0)
        assert res.child_alignments[0].target_coord == Coord3D(1, 0, 0)
        assert res.child_alignments[1].base_child_id == "c1"
        assert res.child_alignments[1].target_child_id == "c2"
        assert res.child_alignments[1].base_coord == Coord3D(3, 0, 0)
        assert res.child_alignments[1].target_coord == Coord3D(4, 0, 0)
        assert frozenset({"c1", "c2"}) in res.equated_children

    def test_gateway_cut_handler_articulation_points_and_bridges(self) -> None:
        """GatewayCutHandler aligns articulation points and creates gw_ active edges."""
        c_art1 = SolveTreeNode(
            node_id="c_art1",
            node_type="leaf",
            rooms=frozenset({11}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        c_art2 = SolveTreeNode(
            node_id="c_art2",
            node_type="leaf",
            rooms=frozenset({12}),
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
            "c_art1": NodeSolution("c_art1", {11: Coord3D(1, 0, 0)}, port_coordinates={1: Coord3D(0, 0, 0)}),
            "c_art2": NodeSolution("c_art2", {12: Coord3D(0, 1, 0)}, port_coordinates={1: Coord3D(0, 0, 0)}),
            "c_brg_fwd": NodeSolution("c_brg_fwd", {21: Coord3D(0, 0, 0)}, ()),
            "c_brg_rev": NodeSolution("c_brg_rev", {31: Coord3D(0, 0, 0)}, ()),
            "c_door": NodeSolution("c_door", {41: Coord3D(0, 0, 0)}, ()),
        }
        graph = _make_graph([1, 2, 3, 4, 11, 12, 21, 31, 41], [(4, 41, Direction.west, 1)])

        handler = GatewayCutHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process()

        assert frozenset({"c_art1", "c_art2"}) not in res.equated_children
        gw_edge_ids = {e.edge_id for e in res.active_edges}
        assert "gw_c_art1_1_11" in gw_edge_ids
        assert "gw_c_art2_1_12" in gw_edge_ids
        assert "gw_c_brg_fwd_2_21" in gw_edge_ids
        assert "gw_c_brg_rev_31_3" in gw_edge_ids
        assert "gw_c_door_4_41" in gw_edge_ids
        assert len(res.active_edges) == 5

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

    def test_inter_child_bridge_handler_already_aligned_with_boundary_ports(self) -> None:
        """InterChildBridgeHandler creates active edges between already-aligned children for designated severed corridors."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"boundary_ports": (100,), "severed_bridge_edges": ((100, 200),)},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"boundary_ports": (200,)},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )
        sols = {
            "c1": NodeSolution(
                node_id="c1",
                local_coordinates={10: Coord3D(0, 0, 0)},
                port_coordinates={100: Coord3D(1, 0, 0)},
            ),
            "c2": NodeSolution(
                node_id="c2",
                local_coordinates={20: Coord3D(0, 0, 0)},
                port_coordinates={200: Coord3D(-1, 0, 0)},
            ),
        }
        graph = _make_graph([1, 2, 10, 20, 100, 200], [(100, 200, Direction.east, 3)])
        handler = InterChildBridgeHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process(aligned_children={"c1", "c2"}, equated_children=set())
        assert len(res.active_edges) == 1
        edge = res.active_edges[0]
        assert edge.src == ("c1", 100)
        assert edge.dst == ("c2", 200)
        assert edge.allow_transverse_slack is True
        assert edge.min_length == 3

        # Uncut graph exits between already-aligned children are discovered as active edges
        parent_no_severed = SolveTreeNode(
            node_id="p_none",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(
                SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({10}), metadata={"boundary_ports": (100,)}),
                SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({20}), metadata={"boundary_ports": (200,)}),
            ),
        )
        handler_no_severed = InterChildBridgeHandler(node=parent_no_severed, child_solutions=sols, graph=graph)
        res_no_severed = handler_no_severed.process(aligned_children={"c1", "c2"}, equated_children=set())
        assert len(res_no_severed.active_edges) == 1
        edge_no_sev = res_no_severed.active_edges[0]
        assert edge_no_sev.src == ("c1", 100)
        assert edge_no_sev.dst == ("c2", 200)
        assert edge_no_sev.allow_transverse_slack is True

    def test_split_pair_handler_active_edge_sp_port_and_span(self) -> None:
        """Split pairs with poles in parent room set emit zero-length port weld and rigid span active edges."""
        p0, p1 = 10, 20
        child_x = SolveTreeNode(
            node_id="cx",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent_x = SolveTreeNode(
            node_id="px",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({10, 20, 30}),
            children=(child_x,),
        )
        sol_x = NodeSolution("cx", {10: Coord3D(0, 0, 0), 20: Coord3D(6, 0, 0)}, ())
        graph = _make_graph([10, 20, 30], [(10, 20, Direction.east, 1)])

        handler = SplitPairHandler(node=parent_x, child_solutions={"cx": sol_x}, graph=graph)
        res = handler.process()

        assert "cx" in res.aligned_children
        assert res.gateway_anchors["cx"] == frozenset({10, 20})

        port_edge = next(e for e in res.active_edges if e.edge_id == "sp_port_cx_10")
        assert port_edge.src == 10
        assert port_edge.dst == ("cx", 10)
        assert port_edge.direction == (0, 0, 0)
        assert port_edge.min_length == 0
        assert port_edge.associated_rooms == frozenset({10})
        assert port_edge.associated_children == frozenset({"cx"})

        span_edge = next(e for e in res.active_edges if e.edge_id == "sp_span_cx_10_20")
        assert span_edge.src == 10
        assert span_edge.dst == 20
        assert span_edge.direction == (1, 0, 0)
        assert span_edge.min_length == 6
        assert span_edge.associated_rooms == frozenset({10, 20})
        assert span_edge.associated_children == frozenset({"cx"})
        assert span_edge.allow_transverse_slack is True

    def test_split_pair_handler_active_edge_span_directions(self) -> None:
        """Split pairs derive primary span direction and minimum length across axes."""
        p0, p1 = 10, 20
        child_y = SolveTreeNode(
            node_id="cy",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        child_z = SolveTreeNode(
            node_id="cz",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({10, 20}),
            children=(child_y, child_z),
        )
        sol_y = NodeSolution("cy", {10: Coord3D(0, 0, 0), 20: Coord3D(0, -5, 0)}, ())
        sol_z = NodeSolution("cz", {10: Coord3D(0, 0, 0), 20: Coord3D(0, 0, 3)}, ())
        graph = _make_graph([10, 20], [])

        handler = SplitPairHandler(node=parent, child_solutions={"cy": sol_y, "cz": sol_z}, graph=graph)
        res = handler.process()

        span_y = next(e for e in res.active_edges if e.edge_id == "sp_span_cy_10_20")
        assert span_y.direction == (0, -1, 0)
        assert span_y.min_length == 5

        span_z = next(e for e in res.active_edges if e.edge_id == "sp_span_cz_10_20")
        assert span_z.direction == (0, 0, 1)
        assert span_z.min_length == 3

    def test_split_pair_handler_active_edge_missing_solutions_and_ports(self) -> None:
        """Split pairs with missing solutions or missing pole coordinates skip active edge generation safely."""
        p0, p1 = 10, 20
        child_no_sol = SolveTreeNode(
            node_id="c_no_sol",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        child_missing_pole = SolveTreeNode(
            node_id="c_miss",
            node_type="leaf",
            rooms=frozenset({10, 20}),
            metadata={"cut_contract": SplitPairCut(poles=(p0, p1))},
        )
        parent = SolveTreeNode(
            node_id="p",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset({10, 20}),
            children=(child_no_sol, child_missing_pole),
        )
        # child_missing_pole only has pole 10, missing pole 20
        sols = {"c_miss": NodeSolution("c_miss", {10: Coord3D(0, 0, 0)}, ())}
        graph = _make_graph([10, 20], [])

        handler = SplitPairHandler(node=parent, child_solutions=sols, graph=graph)
        res = handler.process()

        assert len(res.active_edges) == 0
        assert "c_no_sol" not in res.aligned_children
        assert "c_miss" not in res.aligned_children
