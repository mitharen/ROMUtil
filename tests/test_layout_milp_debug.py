"""Tests for MILP cluster layout debug visualization."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import LayoutEdge, LayoutGraph, NodeSolution, SolveTreeNode, TreeNodeType
from romutil.layout.milp.cluster import HierarchicalClusterSolver, solve_cluster
from romutil.layout.milp.debug import (
    _classify_edge,
    _direction_label,
    _get_child_bounding_box,
    _resolve_endpoint_coord,
    render_cluster_solve_svg,
)
from romutil.layout.milp.inventory import ActiveEdge, ClusterInventory
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, LayoutOptions, SolverStatus


def _create_simple_cluster_inventory(
    rooms: tuple[int, ...] = (100, 101),
    active_edges: tuple[ActiveEdge, ...] = (),
    boundary_ports: frozenset[int] = frozenset(),
    children: tuple[SolveTreeNode, ...] = (),
    child_solutions: dict[str, NodeSolution] | None = None,
    child_sub_boxes: dict[str, tuple[BoundingBox3D, ...]] | None = None,
    graph: LayoutGraph | None = None,
) -> ClusterInventory:
    """Create a test ClusterInventory instance."""
    node = SolveTreeNode(
        node_id="test_cluster",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset(rooms),
        children=children,
    )
    return ClusterInventory(
        node=node,
        rooms=rooms,
        children=children,
        child_solutions=child_solutions or {},
        boundary_ports=boundary_ports,
        active_edges=active_edges,
        room_alignments=(),
        child_alignments=(),
        gateway_anchors={},
        aligned_children=frozenset(),
        equated_children=frozenset(),
        child_sub_boxes=child_sub_boxes or {},
        graph=graph,
    )


def test_resolve_endpoint_coord_parent_room():
    extraction = SolvedClusterExtraction(
        raw_parent_coords={101: (2, 3, 0)},
        raw_child_offsets={},
        solver_status=SolverStatus.OPTIMAL,
    )
    inventory = _create_simple_cluster_inventory(rooms=(101,))
    pt = _resolve_endpoint_coord(101, extraction, inventory)
    assert pt == (2, 3, 0)


def test_resolve_endpoint_coord_child_room():
    child_node = SolveTreeNode(node_id="c1", node_type=TreeNodeType.BLOCK, rooms=frozenset({200}))
    child_sol = NodeSolution(
        node_id="c1",
        local_coordinates={200: Coord3D(1, 1, 0)},
        sub_boxes=(),
    )
    extraction = SolvedClusterExtraction(
        raw_parent_coords={},
        raw_child_offsets={"c1": (5, 5, 0)},
        solver_status=SolverStatus.OPTIMAL,
    )
    inventory = _create_simple_cluster_inventory(
        rooms=(),
        children=(child_node,),
        child_solutions={"c1": child_sol},
    )
    # Via tuple endpoint ref
    pt1 = _resolve_endpoint_coord(("c1", 200), extraction, inventory)
    assert pt1 == (6, 6, 0)

    # Via int endpoint ref searching child solutions
    pt2 = _resolve_endpoint_coord(200, extraction, inventory)
    assert pt2 == (6, 6, 0)


def test_resolve_endpoint_coord_missing_raises_key_error():
    extraction = SolvedClusterExtraction(
        raw_parent_coords={101: (0, 0, 0)},
        raw_child_offsets={},
        solver_status=SolverStatus.OPTIMAL,
    )
    inventory = _create_simple_cluster_inventory(rooms=(101,))
    with pytest.raises(KeyError):
        _resolve_endpoint_coord(999, extraction, inventory)


def test_direction_label():
    assert _direction_label((0, 1, 0)) == "N"
    assert _direction_label((0, -1, 0)) == "S"
    assert _direction_label((1, 0, 0)) == "E"
    assert _direction_label((-1, 0, 0)) == "W"
    assert _direction_label((0, 0, 1)) == "U"
    assert _direction_label((0, 0, -1)) == "D"
    assert _direction_label((2, 3, 4)) == "(2,3,4)"


def test_classify_edge():
    e_corridor = ActiveEdge(
        edge_id="corridor_1",
        src=1,
        dst=2,
        direction=(1, 0, 0),
        min_length=1,
        associated_rooms=frozenset({1, 2}),
        associated_children=frozenset(),
    )
    e_split = ActiveEdge(
        edge_id="split_pair_2",
        src=1,
        dst=2,
        direction=(1, 0, 0),
        min_length=1,
        associated_rooms=frozenset({1, 2}),
        associated_children=frozenset(),
    )
    e_gateway = ActiveEdge(
        edge_id="edge_gw",
        src=("child_1", 1),
        dst=2,
        direction=(1, 0, 0),
        min_length=1,
        associated_rooms=frozenset({2}),
        associated_children=frozenset({"child_1"}),
    )
    e_room_room = ActiveEdge(
        edge_id="r2r_edge",
        src=1,
        dst=2,
        direction=(1, 0, 0),
        min_length=1,
        associated_rooms=frozenset({1, 2}),
        associated_children=frozenset(),
    )
    assert _classify_edge(e_corridor) == "corridor"
    assert _classify_edge(e_split) == "split_pair"
    assert _classify_edge(e_gateway) == "gateway"
    assert _classify_edge(e_room_room) == "room_room"


def test_get_child_bounding_box():
    inventory = _create_simple_cluster_inventory()
    assert _get_child_bounding_box("missing", inventory) is None

    # Child with explicit bounding_box
    sol_bbox = NodeSolution(
        node_id="c_box",
        local_coordinates={},
        sub_boxes=(),
        metadata={"bounding_box": BoundingBox3D(0, 2, 0, 3, 0, 0)},
    )
    inv_bbox = _create_simple_cluster_inventory(child_solutions={"c_box": sol_bbox})
    bbox1 = _get_child_bounding_box("c_box", inv_bbox)
    assert bbox1 == BoundingBox3D(0, 2, 0, 3, 0, 0)

    # Child deriving bbox from local coordinates
    sol_coords = NodeSolution(
        node_id="c_coords",
        local_coordinates={
            1: Coord3D(1, 2, 0),
            2: Coord3D(4, 5, 1),
        },
        sub_boxes=(),
    )
    inv_coords = _create_simple_cluster_inventory(child_solutions={"c_coords": sol_coords})
    bbox2 = _get_child_bounding_box("c_coords", inv_coords)
    assert bbox2 == BoundingBox3D(1, 4, 2, 5, 0, 1)


def test_render_cluster_solve_svg_basic(tmp_path: Path):
    edge = ActiveEdge(
        edge_id="edge_100_101",
        src=100,
        dst=101,
        direction=(1, 0, 0),
        min_length=1,
        associated_rooms=frozenset({100, 101}),
        associated_children=frozenset(),
        allow_transverse_slack=True,
    )
    room100 = Room(vnum=100, name="Temple Entrance", description="", exits=())
    room101 = Room(vnum=101, name="Altar Room", description="", exits=())
    layout_graph = LayoutGraph(rooms={100: room100, 101: room101}, edges=())

    inventory = _create_simple_cluster_inventory(
        rooms=(100, 101),
        active_edges=(edge,),
        boundary_ports=frozenset({100}),
        graph=layout_graph,
    )
    extraction = SolvedClusterExtraction(
        raw_parent_coords={100: (0, 0, 0), 101: (2, 0, 0)},
        raw_child_offsets={},
        solver_status=SolverStatus.OPTIMAL,
    )

    out_file = tmp_path / "test_nested" / "cluster_basic.svg"
    res_path = render_cluster_solve_svg(
        extraction=extraction,
        inventory=inventory,
        output_path=out_file,
        title="Custom Test Cluster",
    )

    assert res_path == out_file
    assert out_file.is_file()

    # Parse XML to verify SVG integrity
    tree = ET.parse(out_file)
    root = tree.getroot()
    assert root.tag.endswith("svg")

    svg_content = out_file.read_text(encoding="utf-8")
    assert "Custom Test Cluster" in svg_content
    assert "Temple Entrance" in svg_content
    assert "Altar Room" in svg_content
    assert "100" in svg_content
    assert "101" in svg_content
    assert "edge_100_101" in svg_content
    assert "slack" in svg_content.lower()


def test_render_cluster_solve_svg_with_children_and_sub_boxes(tmp_path: Path):
    child_node = SolveTreeNode(
        node_id="child_sub",
        node_type=TreeNodeType.SERIES_LOOP,
        rooms=frozenset({201, 202}),
    )
    child_sol = NodeSolution(
        node_id="child_sub",
        local_coordinates={201: Coord3D(0, 0, 0), 202: Coord3D(1, 0, 0)},
        sub_boxes=(BoundingBox3D(0, 1, 0, 1, 0, 0),),
        metadata={"bounding_box": BoundingBox3D(0, 1, 0, 1, 0, 0)},
    )
    sub_box = BoundingBox3D(0, 1, 0, 1, 0, 0)

    edge = ActiveEdge(
        edge_id="bridge_edge",
        src=100,
        dst=("child_sub", 201),
        direction=(0, 1, 0),
        min_length=2,
        associated_rooms=frozenset({100}),
        associated_children=frozenset({"child_sub"}),
    )

    inventory = _create_simple_cluster_inventory(
        rooms=(100,),
        children=(child_node,),
        child_solutions={"child_sub": child_sol},
        child_sub_boxes={"child_sub": (sub_box,)},
        active_edges=(edge,),
        boundary_ports=frozenset({100, 201}),
    )
    extraction = SolvedClusterExtraction(
        raw_parent_coords={100: (0, 0, 0)},
        raw_child_offsets={"child_sub": (0, 2, 0)},
        solver_status=SolverStatus.OPTIMAL,
    )

    out_file = tmp_path / "cluster_with_child.svg"
    render_cluster_solve_svg(extraction, inventory, out_file)

    assert out_file.is_file()
    svg_content = out_file.read_text(encoding="utf-8")
    assert "child_sub" in svg_content
    assert "Keep-Out Sub-box" in svg_content
    assert "bridge_edge" in svg_content
    assert "201" in svg_content


def test_render_cluster_solve_svg_3d_elevations(tmp_path: Path):
    child_node = SolveTreeNode(
        node_id="child_tower",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset({301}),
    )
    child_sol = NodeSolution(
        node_id="child_tower",
        local_coordinates={301: Coord3D(0, 0, 0)},
        sub_boxes=(),
        metadata={"bounding_box": BoundingBox3D(0, 1, 0, 1, 0, 2)},
    )

    inventory = _create_simple_cluster_inventory(
        rooms=(100, 101),
        children=(child_node,),
        child_solutions={"child_tower": child_sol},
    )
    extraction = SolvedClusterExtraction(
        raw_parent_coords={100: (0, 0, 0), 101: (0, 0, 1)},
        raw_child_offsets={"child_tower": (3, 0, 0)},
        solver_status=SolverStatus.OPTIMAL,
    )

    out_file = tmp_path / "cluster_3d.svg"
    render_cluster_solve_svg(extraction, inventory, out_file)

    assert out_file.is_file()
    svg_content = out_file.read_text(encoding="utf-8")
    assert "3D Prism" in svg_content
    assert "(0,0,1)" in svg_content


def test_solve_cluster_with_debug_solve_dir(tmp_path: Path):
    r1 = Room(
        vnum=501,
        name="Room A",
        description="",
        exits=(Exit(direction=Direction.east, dst_vnum=502),),
    )
    r2 = Room(
        vnum=502,
        name="Room B",
        description="",
        exits=(Exit(direction=Direction.west, dst_vnum=501),),
    )
    graph = LayoutGraph(
        rooms={501: r1, 502: r2},
        edges=(
            LayoutEdge(src=501, dst=502, direction=Direction.east.value),
            LayoutEdge(src=502, dst=501, direction=Direction.west.value),
        ),
    )
    node = SolveTreeNode(
        node_id="test_two_room",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset({501, 502}),
    )

    debug_dir = tmp_path / "debug_out"
    options = LayoutOptions(debug_solve_dir=debug_dir)

    solution = solve_cluster(
        node=node,
        graph=graph,
        options=options,
    )
    assert 501 in solution.local_coordinates
    assert 502 in solution.local_coordinates

    svg_file = debug_dir / "cluster_test_two_room.svg"
    assert svg_file.is_file()
    assert "Room A" in svg_file.read_text(encoding="utf-8")

    # Test via HierarchicalClusterSolver with direct debug_solve_dir parameter
    direct_debug_dir = tmp_path / "direct_debug"
    cluster_solver = HierarchicalClusterSolver()
    solution2 = cluster_solver.solve(
        node=node,
        graph=graph,
        debug_solve_dir=direct_debug_dir,
    )
    assert 501 in solution2.local_coordinates
    direct_svg = direct_debug_dir / "cluster_test_two_room.svg"
    assert direct_svg.is_file()
