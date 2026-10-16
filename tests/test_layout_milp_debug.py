"""Tests for MILP cluster layout debug visualization."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
)
from romutil.layout.milp.cluster import HierarchicalClusterSolver, solve_cluster
from romutil.layout.milp.debug import (
    _classify_edge,
    _direction_label,
    _get_child_bounding_box,
    _resolve_endpoint_coord,
    render_cluster_solve_svg,
    render_debug_solve_tree_index,
)
from romutil.layout.milp.inventory import ActiveEdge, ClusterInventory
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.layout.milp.tree import HierarchicalTreeSolver
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

    # Two-tier legend items
    assert "Parent Room" in svg_content
    assert "Parent Port" in svg_content
    assert "Child Room" in svg_content
    assert "Child Port" in svg_content
    assert "Child Macro-Box (Envelope)" in svg_content
    assert "Keep-Out Sub-Box (Obstacle)" in svg_content
    assert "Corridor" in svg_content
    assert "Split-Pair" in svg_content
    assert "Gateway/Bridge" in svg_content
    assert "Room-to-Room" in svg_content
    assert "Transverse Slack" in svg_content


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
    assert "202" in svg_content
    assert "cluster_child_sub.svg" in svg_content
    assert "Click to view child cluster child_sub" in svg_content
    assert "index.html" in svg_content
    assert "Index" in svg_content

    # Child cluster tag badge placement
    assert "Child: child_sub" in svg_content

    # Child room circle sizing & typography
    # 201 is boundary port (r=15), 202 is normal room (r=13)
    assert 'r="15"' in svg_content
    assert 'r="13"' in svg_content
    assert 'font-size="11px"' in svg_content


def test_render_cluster_solve_svg_legend_and_child_tag_placement(tmp_path: Path):
    child_node = SolveTreeNode(
        node_id="comp_0",
        node_type=TreeNodeType.BLOCK,
        rooms=frozenset({500, 501}),
    )
    child_sol = NodeSolution(
        node_id="comp_0",
        local_coordinates={500: Coord3D(0, 0, 0), 501: Coord3D(1, 0, 0)},
        sub_boxes=(),
        metadata={"bounding_box": BoundingBox3D(0, 1, 0, 1, 0, 0)},
    )
    inventory = _create_simple_cluster_inventory(
        rooms=(400,),
        children=(child_node,),
        child_solutions={"comp_0": child_sol},
        boundary_ports=frozenset({500}),
    )
    extraction = SolvedClusterExtraction(
        raw_parent_coords={400: (0, 3, 0)},
        raw_child_offsets={"comp_0": (0, 0, 0)},
        solver_status=SolverStatus.OPTIMAL,
    )
    out_file = tmp_path / "cluster_legend_tag_test.svg"
    render_cluster_solve_svg(extraction, inventory, out_file)

    assert out_file.is_file()
    tree = ET.parse(out_file)
    root = tree.getroot()

    # Find the legend card rect and verify its dimensions
    legend_rect = None
    for g in root.iter():
        if g.attrib.get("id") == "legend":
            for elem in g:
                if elem.tag.endswith("rect") and elem.attrib.get("y") == "96":
                    legend_rect = elem
                    break
    assert legend_rect is not None
    assert legend_rect.attrib.get("height") == "74"

    # Verify child badge rect is placed inside top-left corner of the macro-box
    child_group = None
    for g in root.iter():
        if g.attrib.get("id") == "child-comp_0":
            child_group = g
            break
    assert child_group is not None

    # Find macro-box polygon and tag rect
    polygon_points_str = None
    tag_rect_y = None
    for elem in child_group.iter():
        if elem.tag.endswith("polygon") and "points" in elem.attrib:
            polygon_points_str = elem.attrib["points"]
        if elem.tag.endswith("rect") and elem.attrib.get("fill") == "#1d4ed8":
            tag_rect_y = float(elem.attrib["y"])

    assert polygon_points_str is not None
    assert tag_rect_y is not None
    # First point in polygon is p_nw0 (top-left)
    p_nw_y = float(polygon_points_str.split()[0].split(",")[1])
    # Tag y should be strictly inside the macro-box (tag_y >= p_nw_y), not above it
    assert tag_rect_y >= p_nw_y


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


def test_render_cluster_solve_svg_with_parent_breadcrumb(tmp_path: Path):
    node = SolveTreeNode(
        node_id="child_cluster_7",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset({101}),
        metadata={"parent_id": "root_spqr_0"},
    )
    inventory = ClusterInventory(
        node=node,
        rooms=(101,),
        children=(),
        child_solutions={},
        boundary_ports=frozenset(),
        active_edges=(),
        child_alignments=(),
        gateway_anchors={},
        aligned_children=frozenset(),
        equated_children=frozenset(),
        child_sub_boxes={},
        graph=None,
    )
    extraction = SolvedClusterExtraction(
        raw_parent_coords={101: (0, 0, 0)},
        raw_child_offsets={},
        solver_status=SolverStatus.OPTIMAL,
    )
    out_file = tmp_path / "cluster_child_7.svg"
    render_cluster_solve_svg(extraction, inventory, out_file)
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "cluster_root_spqr_0.svg" in content
    assert "↑ root_spqr_0" in content


def test_render_debug_solve_tree_index_complete_hierarchy(tmp_path: Path):
    leaf_a = SolveTreeNode(
        node_id="leaf_a",
        node_type=TreeNodeType.BLOCK,
        rooms=frozenset({10, 11}),
    )
    leaf_b = SolveTreeNode(
        node_id="leaf_b",
        node_type="leaf",
        rooms=frozenset({12}),
    )
    series_node = SolveTreeNode(
        node_id="series_1",
        node_type=TreeNodeType.SERIES_LOOP,
        rooms=frozenset({10, 11, 12}),
        children=(leaf_a, leaf_b),
        metadata={"cut_type": "articulation_point", "anchor_vnum": 10},
    )
    split_node = SolveTreeNode(
        node_id="split_pair_2",
        node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
        rooms=frozenset({20, 21}),
        children=(),
        metadata={"cut_type": "split_pair", "poles": (20, 21)},
    )
    root_node = SolveTreeNode(
        node_id="root_rigid",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset({1, 2}),
        children=(series_node, split_node),
    )
    comp_node = SolveTreeNode(
        node_id="comp_isolated",
        node_type=TreeNodeType.BLOCK,
        rooms=frozenset({99}),
        children=(),
    )
    tree = SolveTree(
        root=root_node,
        components=(comp_node,),
    )

    out_dir = tmp_path / "debug_solves" / "thalos"
    index_file = render_debug_solve_tree_index(tree, out_dir, area_name="thalos")

    assert index_file == out_dir / "index.html"
    assert index_file.is_file()
    html_text = index_file.read_text(encoding="utf-8")

    # Metrics
    assert "SolveTree Explorer: Thalos" in html_text
    assert "root_rigid" in html_text
    assert "OPTIMAL" in html_text
    assert "Nodes:" in html_text
    assert "Depth:" in html_text
    assert "Rooms:" in html_text

    # Node IDs and direct SVG links
    assert "cluster_root_rigid.svg" in html_text
    assert "cluster_series_1.svg" in html_text
    assert "cluster_split_pair_2.svg" in html_text
    assert "cluster_leaf_a.svg" in html_text
    assert "cluster_leaf_b.svg" in html_text
    assert "cluster_comp_isolated.svg" in html_text

    # Badges
    assert "RIGID_CLUSTER" in html_text
    assert "SERIES_LOOP" in html_text
    assert "SPLIT_PAIR_CONTAINER" in html_text
    assert "ARTICULATION_POINT" in html_text
    assert "SPLIT_PAIR" in html_text
    assert "BLOCK" in html_text

    # Disconnected components section
    assert "Disconnected Components" in html_text

    # Area map breadcrumb link
    assert "../../thalos.html" in html_text

    # Iframe preview element
    assert '<iframe id="svg-frame" src="cluster_root_rigid.svg"' in html_text

    # Tree hierarchy CSS styling and indentation rules
    assert ".tree-children {" in html_text
    assert "margin-left: 14px;" in html_text
    assert "border-left: 1px dashed #cbd5e1;" in html_text
    assert ".tree-scroll > details.tree-item" in html_text
    assert "details.tree-item:first-of-type" not in html_text


def test_render_debug_solve_tree_index_hierarchy_indentation_css(tmp_path: Path):
    """Verify hierarchical tree indentation CSS rules and DOM nesting in solve tree explorer."""
    leaf_child = SolveTreeNode(
        node_id="child_leaf",
        node_type=TreeNodeType.BLOCK,
        rooms=frozenset({101}),
    )
    branch_child = SolveTreeNode(
        node_id="child_branch",
        node_type=TreeNodeType.SERIES_LOOP,
        rooms=frozenset({102}),
        children=(leaf_child,),
    )
    root = SolveTreeNode(
        node_id="root_cluster",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset({100, 101, 102}),
        children=(branch_child,),
    )
    tree = SolveTree(root=root, components=())

    out_dir = tmp_path / "tree_test"
    index_file = render_debug_solve_tree_index(tree, out_dir, area_name="test_area")
    html_text = index_file.read_text(encoding="utf-8")

    # CSS indentation and alignment rules
    assert ".tree-children {" in html_text
    assert "margin-left: 14px;" in html_text
    assert "padding-left: 6px;" in html_text
    assert "border-left: 1px dashed #cbd5e1;" in html_text

    # Root item targeting without unwanted outer indent
    assert ".tree-scroll > details.tree-item" in html_text
    assert ".tree-scroll > .tree-item" in html_text

    # Negative check: no first-of-type selector collapsing first child nodes
    assert "details.tree-item:first-of-type" not in html_text

    # Caret and leaf alignment
    assert ".caret {" in html_text
    assert ".leaf-dot {" in html_text
    assert "flex-shrink: 0;" in html_text

    # DOM structure: root details contains tree-children, which contains child_branch details,
    # which contains tree-children, which contains child_leaf leaf div.
    assert '<details class="tree-item" open data-node-id="root_cluster"' in html_text
    assert '<div class="tree-children">' in html_text
    assert '<details class="tree-item" open data-node-id="child_branch"' in html_text
    assert '<div class="tree-item leaf" data-node-id="child_leaf"' in html_text



def test_render_debug_solve_tree_index_inferred_area_and_leaf_root(tmp_path: Path):
    root = SolveTreeNode(
        node_id="single_leaf",
        node_type=TreeNodeType.BLOCK,
        rooms=frozenset({100}),
    )
    tree = SolveTree(root=root, components=())
    solve_dir = tmp_path / "midgaard"
    index_file = render_debug_solve_tree_index(tree, solve_dir)

    assert index_file.is_file()
    html_text = index_file.read_text(encoding="utf-8")
    assert "Midgaard" in html_text
    assert "single_leaf" in html_text
    assert "cluster_single_leaf.svg" in html_text


def test_solve_tree_generates_debug_index(tmp_path: Path):
    r1 = Room(
        vnum=10,
        name="Hall",
        description="",
        exits=(Exit(direction=Direction.east, dst_vnum=11),),
    )
    r2 = Room(
        vnum=11,
        name="Chamber",
        description="",
        exits=(Exit(direction=Direction.west, dst_vnum=10),),
    )
    graph = LayoutGraph(
        rooms={10: r1, 11: r2},
        edges=(LayoutEdge(src=10, dst=11, direction=Direction.east.value),),
    )
    root = SolveTreeNode(
        node_id="tree_root",
        node_type=TreeNodeType.RIGID_CLUSTER,
        rooms=frozenset({10, 11}),
        children=(),
    )
    tree = SolveTree(root=root, graph=graph)

    debug_dir = tmp_path / "solve_inspect"
    options = LayoutOptions(debug_solve_dir=debug_dir)

    solver = HierarchicalTreeSolver()
    solution = solver.solve_tree(tree, options=options)

    assert "tree_root" in solution.node_solutions
    index_file = debug_dir / "index.html"
    assert index_file.is_file()
    html_text = index_file.read_text(encoding="utf-8")
    assert "tree_root" in html_text
    assert "cluster_tree_root.svg" in html_text
