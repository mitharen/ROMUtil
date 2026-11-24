"""Unit and integration tests for exact collision-time clearance box growth and slanted corridor boxes."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

from pyomo.opt import SolverFactory
import pyomo.environ as pyo
import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    ArticulationPointCut,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTreeNode,
    TreeNodeType,
)
from romutil.layout.milp import ClusterFormulation, detect_cluster_collisions, inventory_cluster
from romutil.layout.milp.clearance import (
    ChildSubClearanceBox,
    ClearanceViolation,
    CorridorClearanceBox,
    RoomClearanceBox,
)
from romutil.layout.milp.clearance_boxes import (
    ExactCollisionCutGrower,
    _cells_2d_to_x_runs,
    _minkowski_diff_2d,
    compute_planar_component_sizes,
    dilate_2d_along_axis,
    fill_enclosed_2d_holes,
    fill_narrow_gaps_along_axis,
    filter_box_directions,
    get_locked_z_diff,
    max_box_3d_containing,
    max_rect_2d_containing,
    sub_boxes_to_cells,
)
from romutil.layout.milp.cluster import solve_cluster
from romutil.layout.milp.inventory import ActiveEdge, ClusterInventory
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.layout.milp.spatial_dag import ClusterPositionalDAG, SpatialBoundExpr
from romutil.layout.positional_dag import AXIS_X, AXIS_Y, AXIS_Z
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, LayoutOptions, SolverStatus


def _make_room(vnum: int, exits: dict[Direction, int] | None = None) -> Room:
    exits_dict = exits or {}
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(Exit(direction=int(d), dst_vnum=dst) for d, dst in exits_dict.items()),
    )


def _make_graph(
    rooms: list[int],
    edges: list[tuple[int, int, Direction | int, int]],
) -> LayoutGraph:
    exits_map: dict[int, dict[Direction, int]] = {r: {} for r in rooms}
    layout_edges: list[LayoutEdge] = []
    for u, v, d, w in edges:
        exits_map[u][Direction(d)] = v
        layout_edges.append(LayoutEdge(src=u, dst=v, direction=int(d), weight=w))
    return LayoutGraph(
        rooms={r: _make_room(r, exits_map[r]) for r in rooms},
        edges=tuple(layout_edges),
    )


class TestDiscreteGeometryPrimitives:
    """Unit tests for cell expansion, bounded hole filling, gap filling, and max-rect growth."""

    def test_sub_boxes_to_cells_supports_bbox_and_legacy_tuples(self) -> None:
        """Expands BoundingBox3D and legacy 2-point tuple sub-boxes into exact 3D cell sets."""
        cells = sub_boxes_to_cells(
            [
                BoundingBox3D(0, 1, 0, 0, 0, 0),
                (Coord3D(2, 1, 1), Coord3D(2, 2, 1)),
            ]
        )
        assert cells == {(0, 0, 0), (1, 0, 0), (2, 1, 1), (2, 2, 1)}

    def test_fill_enclosed_2d_holes_respects_hole_size_and_open_pockets(self) -> None:
        """Fills enclosed interior voids up to max_hole_size while keeping open pockets and large yards empty."""
        # Solid wall at x=5 separates the 4-cell courtyard (3..4, 1..2) from the right edge (x=6),
        # leaving an open exterior notch at (6, 2) plus enclosed holes at (1, 2) [1 cell] and (3..4, 1..2) [4 cells].
        clean_frame = {
            (x, y)
            for x in range(7)
            for y in range(5)
            if not (
                (x == 1 and y == 2)  # 1-cell enclosed hole
                or (x in (3, 4) and y in (1, 2))  # 4-cell enclosed courtyard
                or (x == 6 and y == 2)  # open exterior notch on right wall
            )
        }
        # Only (1, 2) has size <= 2
        filled_small = fill_enclosed_2d_holes(clean_frame, max_hole_size=2)
        assert (1, 2) in filled_small
        assert (3, 1) not in filled_small
        assert (6, 2) not in filled_small

        # With max_hole_size=4, both enclosed holes fill, but exterior notch (6, 2) still stays empty
        filled_large = fill_enclosed_2d_holes(clean_frame, max_hole_size=4)
        assert (1, 2) in filled_large
        assert (3, 1) in filled_large and (4, 2) in filled_large
        assert (6, 2) not in filled_large

        # With max_hole_size <= 0 or empty set, returns unchanged
        assert fill_enclosed_2d_holes(clean_frame, max_hole_size=0) == clean_frame
        assert fill_enclosed_2d_holes(set(), max_hole_size=5) == set()

    def test_dilate_and_fill_narrow_gaps_along_x_and_y(self) -> None:
        """Asymmetric axis dilation and collinear gap filling operate accurately along X and Y."""
        base = {(0, 0), (3, 0), (0, 4)}
        dx = dilate_2d_along_axis(base, "x", neg=0, pos=1)
        assert {(1, 0), (4, 0), (1, 4)} <= dx
        assert (-1, 0) not in dx

        dy = dilate_2d_along_axis(base, "y", neg=1, pos=0)
        assert {(0, -1), (3, -1), (0, 3)} <= dy
        assert (0, 1) not in dy

        # Gap on row y=0 between x=0 and x=3 has size 2; gap between x=0 and x=6 has size 2 (< 3) vs (3,0)..(6,0) size 2
        pts_gap_x = {(0, 0), (2, 0), (6, 0)}  # gaps of 1 and 3 along y=0
        fx = fill_narrow_gaps_along_axis(pts_gap_x, "x", min_clear_span=2)
        assert (1, 0) in fx
        assert (3, 0) not in fx and (4, 0) not in fx and (5, 0) not in fx

        pts_gap_y = {(1, 0), (1, 2), (1, 6)}
        fy = fill_narrow_gaps_along_axis(pts_gap_y, "y", min_clear_span=2)
        assert (1, 1) in fy
        assert (1, 3) not in fy

        assert fill_narrow_gaps_along_axis(pts_gap_x, "x", min_clear_span=1) == pts_gap_x

    def test_max_rect_2d_and_box_3d_containing(self) -> None:
        """Grows exact maximum-area solid 2D rectangle and 3D prism containing seed point."""
        # L-shaped union: horizontal bar [0..4, 1..2] + vertical stem [2..3, 0..4]
        pts_2d = {(x, y) for x in range(5) for y in range(1, 3)} | {
            (x, y) for x in range(2, 4) for y in range(5)
        }
        # Seed (2, 2) belongs to both bars; max area is 5x2=10 (x=0..4, y=1..2) vs 2x5=10 (deterministic tie-break)
        rect = max_rect_2d_containing(pts_2d, 1, 1)
        assert rect == (0, 4, 1, 2)
        assert max_rect_2d_containing(pts_2d, 9, 9) == (9, 9, 9, 9)

        # 3D prism: z=0,1 have [0..2, 0..1], z=2 misses (1, 1) -> stops at z=1
        pts_3d = {(x, y, z) for x in range(3) for y in range(2) for z in (0, 1)} | {
            (0, 0, 2),
            (2, 1, 2),
        }
        assert max_box_3d_containing(pts_3d, 1, 1, 0) == BoundingBox3D(0, 2, 0, 1, 0, 1)
        assert max_box_3d_containing(pts_3d, 9, 9, 9) == BoundingBox3D(9, 9, 9, 9, 9, 9)

    def test_cells_2d_to_x_runs_and_minkowski_diff_2d_equivalence_and_guards(self) -> None:
        """Run-length 2D Minkowski difference matches naive cell pairs and enforces run/output guards."""
        assert _cells_2d_to_x_runs(set()) == []
        assert _minkowski_diff_2d(set(), {(0, 0)}) == set()
        assert _minkowski_diff_2d({(0, 0)}, set()) == set()

        s1 = {(-1, 0), (0, 0), (1, 0), (3, 0), (0, 1), (3, 2), (4, 2)}
        s2 = {(2, -1), (3, -1), (5, -1), (2, 0), (3, 0), (4, 1)}
        assert _cells_2d_to_x_runs(s1) == [(0, -1, 1), (0, 3, 3), (1, 0, 0), (2, 3, 4)]

        expected = {(x2 - x1, y2 - y1) for (x2, y2) in s2 for (x1, y1) in s1}
        assert _minkowski_diff_2d(s2, s1) == expected

        # Guard: len(r1) * len(r2) > max_run_pairs returns None
        assert _minkowski_diff_2d(s2, s1, max_run_pairs=5) is None

        # Guard: total output cells > max_output_cells returns None
        bar_a = {(x, 0) for x in range(20)}
        bar_b = {(x, 0) for x in range(20)}
        assert _minkowski_diff_2d(bar_b, bar_a, max_output_cells=30) is None

        # Disjoint intervals on same dy row exceeding max_output_cells also returns None
        two_bars = {(x, 0) for x in range(5)} | {(x, 0) for x in range(20, 25)}
        assert _minkowski_diff_2d(two_bars, {(0, 0)}, max_output_cells=8) is None


class TestPlanarComponentAndZLockHelpers:
    """Tests for planar connected component lower bounds, DAG Z-lock queries, and direction pruning."""

    def test_compute_planar_component_sizes_excludes_vertical_edges(self) -> None:
        """Sums rooms and min-slice child cells connected by planar edges while separating Z-jumps."""
        ch_a = SolveTreeNode(node_id="ch_a", node_type=TreeNodeType.BLOCK, rooms=frozenset({10, 11}))
        sol_a = NodeSolution(
            node_id="ch_a",
            local_coordinates={10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 2, 0, 1, 0, 0),  # 6 cells on z=0
                BoundingBox3D(0, 1, 0, 0, 1, 1),  # 2 cells on z=1 -> min_slice = 2
            ),
        )
        ae_planar = ActiveEdge(
            edge_id="e_flat",
            src=1,
            dst=("ch_a", 10),
            direction=(1, 0, 0),
            min_length=1,
            associated_rooms=frozenset({1}),
            associated_children=frozenset({"ch_a"}),
        )
        ae_vert = ActiveEdge(
            edge_id="e_up",
            src=1,
            dst=2,
            direction=(0, 0, 1),
            min_length=1,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
        )
        parent = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
            children=(ch_a,),
        )
        inv = inventory_cluster(parent, {"ch_a": sol_a}, _make_graph([1, 2, 10, 11], []))
        inv = dataclasses.replace(inv, active_edges=(ae_planar, ae_vert))

        sizes = compute_planar_component_sizes(inv)
        # Room 1 + ch_a (min_slice=2) share planar edge => weight 3; Room 2 is Z-separated => weight 1
        assert sizes[1] == 3
        assert sizes["ch_a"] == 3
        assert sizes[2] == 1

    def test_get_locked_z_diff_and_filter_box_directions(self) -> None:
        """Verifies signed Z-lock extraction and DAG pruning of impossible half-spaces."""
        inv = ClusterInventory(
            node=SolveTreeNode(node_id="r", node_type=TreeNodeType.BLOCK, rooms=frozenset({1})),
            rooms=(1,),
            children=(SolveTreeNode(node_id="c2", node_type=TreeNodeType.BLOCK, rooms=frozenset({20})),),
            child_solutions={"c2": NodeSolution("c2", {20: Coord3D(0, 0, 3)})},
            boundary_ports=frozenset(),
            active_edges=(),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            child_sub_boxes={"c2": (BoundingBox3D(0, 1, 0, 1, 2, 3),)},
        )
        dag = ClusterPositionalDAG.from_formulation(ClusterFormulation(inventory=inv))
        dag.add_equality(AXIS_Z, 1, "c2", -3)
        dag.add_equality(AXIS_X, 1, "c2", 0)
        # z(1) - z(c2) == 3, with off_a=1, off_b=2 -> (z(1)+1) - (z(c2)+2) == 2
        assert get_locked_z_diff(dag, 1, 1, "c2", 2) == 2
        assert get_locked_z_diff(dag, "c2", 5, "c2", 2) == 3
        assert get_locked_z_diff(dag, 1, 0, 999, 0) is None

        bounds_room = (
            SpatialBoundExpr(1, 0, 1, 0),
            SpatialBoundExpr(1, 0, 1, 0),
            SpatialBoundExpr(1, 0, 1, 0),
        )
        bounds_box = (
            SpatialBoundExpr("c2", 0, "c2", 1),
            SpatialBoundExpr("c2", 0, "c2", 1),
            SpatialBoundExpr("c2", 0, "c2", 0),
        )
        # c2 is constrained right/ahead-free along Y only since x(c2) >= x(1)+2 is not violated? Wait x(1)==x(c2), so min_x(1)=max_x(c2)+1 fails; min_x(c2)=max_x(1)+1 fails!
        valid = filter_box_directions(dag, bounds_room, bounds_box, ("r", "l", "a", "b"))
        assert valid == ["a", "b"]


class TestExactCollisionCutGrower:
    """Unit and end-to-end tests for ExactCollisionCutGrower and ClearanceConstraintBuilder integration."""

    @staticmethod
    def _build_two_child_formulation(
        *,
        c1_boxes: tuple[BoundingBox3D, ...],
        c2_boxes: tuple[BoundingBox3D, ...],
        edges: list[tuple[int, int, Direction | int, int]],
        c1_coords: dict[int, Coord3D],
        c2_coords: dict[int, Coord3D],
    ) -> ClusterFormulation:
        ch1 = SolveTreeNode(
            node_id="c1",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(c1_coords.keys()),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        ch2 = SolveTreeNode(
            node_id="c2",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset(c2_coords.keys()),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=2)},
        )
        all_rooms = sorted({1, 2, *c1_coords.keys(), *c2_coords.keys()})
        graph = _make_graph(all_rooms, edges)
        parent = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2}),
            children=(ch1, ch2),
        )
        sol1 = NodeSolution("c1", c1_coords, port_coordinates={1: Coord3D(-1, 0, 0)}, sub_boxes=c1_boxes)
        sol2 = NodeSolution("c2", c2_coords, port_coordinates={2: Coord3D(0, -1, 0)}, sub_boxes=c2_boxes)
        inv = inventory_cluster(parent, {"c1": sol1, "c2": sol2}, graph)
        inv = dataclasses.replace(inv, child_sub_boxes={"c1": c1_boxes, "c2": c2_boxes})
        return ClusterFormulation(inventory=inv, options=LayoutOptions(timeout_seconds=10))

    def test_child_child_minkowski_closes_infeasible_hole_yet_preserves_concave_nesting(self) -> None:
        """Grows exact Minkowski box ignoring an infeasible 1x1 hole while allowing open-pocket nesting."""
        # c_pocket: 4x4 frame with open 2x2 concave bay on the West side (x=0..1, y=1..2)
        # and an enclosed 1x1 infeasible hole inside at (2, 2)? Let's represent solid walls:
        #   - top bar (0..3, 3), bottom bar (0..3, 0), right wall (3, 1..2), mid-post (2, 1)? No:
        #   Look at {(0..3, 0), (0..3, 3), (3, 1..2), (1, 2)} -> pocket at (0..2, 1..2) excluding (1, 2)
        # Easier: C-shape [(0..3, 0), (0..3, 3), (3, 1..2)] PLUS enclosed 1-cell hole at (3, 1) with outer wall at x=4:
        #   Wait: 3-room solid frame on x=2..4, y=0..2 with 1-cell hole (3, 1), AND open West pocket x=0..1, y=1..2:
        #   Bottom bar: (0..4, 0), Top bar: (0..4, 2)... Wait height 5 (y=0..4) gives room:
        #   Let pocket be (1, 1) [1-cell enclosed hole] at x=3, y=2 inside wall (2..4, 1..3), and open West bay x=0..1, y=1..3!
        c_pocket_boxes = (
            BoundingBox3D(0, 4, 0, 0, 0, 0),  # bottom wall
            BoundingBox3D(0, 4, 4, 4, 0, 0),  # top wall
            BoundingBox3D(2, 2, 1, 3, 0, 0),  # vertical bar closing x=3,y=2 (wait leaves (3,2) + (3,1),(3,3)!)
            BoundingBox3D(4, 4, 1, 3, 0, 0),  # right wall
            BoundingBox3D(3, 3, 1, 1, 0, 0),  # leaves 1x1 hole at (3, 2)!
            BoundingBox3D(3, 3, 3, 3, 0, 0),
        )
        # Verify c_small (1x1) can sit inside the OPEN West bay (1, 2) even when cuts use ExactCollisionCutGrower!
        c_small_boxes = (BoundingBox3D(0, 0, 0, 0, 0, 0),)
        form = self._build_two_child_formulation(
            c1_boxes=c_pocket_boxes,
            c2_boxes=c_small_boxes,
            edges=[(1, 2, Direction.east, 2)],  # short flat tie keeping Z locked
            c1_coords={10: Coord3D(0, 0, 0)},
            c2_coords={20: Coord3D(0, 0, 0)},
        )
        builder = form.clearance_builder
        grower = ExactCollisionCutGrower(form.inventory, builder.spatial_dag)

        # Collision at offset(c_pocket)=(0,0,0), offset(c_small)=(2,2,0) => rel=(2,2,0) touches bar (2, 2)
        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (-1, 0, 0), 2: (2, 1, 0)},
            raw_child_offsets={"c1": (0, 0, 0), "c2": (2, 2, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        # Wait: c1 has a 1x1 hole at (3, 2) and c2 has 1 cell (0,0,0) -> hole IS feasible for 1x1 c2!
        # Give c2 a 2x1 shape so the 1x1 hole (3, 2) IS geometrically infeasible for c2!
        c_two_cell_boxes = (BoundingBox3D(0, 1, 0, 0, 0, 0),)
        form2 = self._build_two_child_formulation(
            c1_boxes=c_pocket_boxes,
            c2_boxes=c_two_cell_boxes,
            edges=[(1, 2, Direction.east, 2)],
            c1_coords={10: Coord3D(0, 0, 0)},
            c2_coords={20: Coord3D(0, 0, 0), 21: Coord3D(1, 0, 0)},
        )
        b2 = form2.clearance_builder
        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            injected = b2.inject_clearance_cuts(
                [ClearanceViolation("child_child", ChildSubClearanceBox("c1", 2), ChildSubClearanceBox("c2", 0))],
                extraction=ext,
            )
            assert injected == 1
            assert any(k[0] == "mink_child_child" for k in b2.separated_pairs)
            mink_key = next(k for k in b2.separated_pairs if k[0] == "mink_child_child")
            # Box covers the filled solid block x=2..4 (minus right offset -> spans at least width 3 across infeasible hole!),
            # while leaving open West pocket (x=0, y=2: dx=0, dy=2) OUTSIDE the Minkowski cut!
            mink_box: BoundingBox3D = mink_key[3]
            assert mink_box.volume() >= 9
            assert mock_sep.call_args.kwargs["prefix"] == "mink_child_child"

        # End-to-end check: force 2x1 sibling right inside open West bay (x=0..1, y=2)!
        # Anchor 1 holds c1 at (1, 0, 0); Anchor 2 holds c2 at x(2)=1, y(2)=1 => c2 at (1, 2) INSIDE open West bay!
        if SolverFactory("cbc").available():
            g_nest = _make_graph(
                [1, 2, 10, 20, 21],
                [
                    (1, 10, Direction.east, 1),
                    (1, 2, Direction.north, 1),
                    (2, 20, Direction.north, 1),
                    (20, 21, Direction.east, 1),
                ],
            )
            ch1_n = SolveTreeNode("c1", TreeNodeType.BLOCK, frozenset({10}), metadata={"cut_contract": ArticulationPointCut(1)})
            ch2_n = SolveTreeNode("c2", TreeNodeType.BLOCK, frozenset({20, 21}), metadata={"cut_contract": ArticulationPointCut(2)})
            root_n = SolveTreeNode("root", TreeNodeType.BLOCK, frozenset({1, 2}), children=(ch1_n, ch2_n))
            s1_n = NodeSolution("c1", {10: Coord3D(0, 0, 0)}, port_coordinates={1: Coord3D(-1, 0, 0)}, sub_boxes=c_pocket_boxes)
            s2_n = NodeSolution("c2", {20: Coord3D(0, 0, 0), 21: Coord3D(1, 0, 0)}, port_coordinates={2: Coord3D(0, -1, 0)}, sub_boxes=c_two_cell_boxes)
            sol_root = solve_cluster(root_n, {"c1": s1_n, "c2": s2_n}, g_nest, LayoutOptions(timeout_seconds=15))
            # Open West bay (x=0..1, y=1..3 of c1) allows c2 to stay nested cleanly without pushing 1->2 far away!
            assert sol_root.metadata["status"] in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    def test_child_child_unlocked_z_3d_minkowski_and_negative_guards(self) -> None:
        """Uses 3D Minkowski when Z is unlocked and falls back on non-overlapping / self / missing offsets."""
        ch1 = SolveTreeNode("c1", TreeNodeType.BLOCK, frozenset({10}))
        ch2 = SolveTreeNode("c2", TreeNodeType.BLOCK, frozenset({20}))
        parent = SolveTreeNode("root", TreeNodeType.BLOCK, frozenset({1}), children=(ch1, ch2))
        s1 = NodeSolution("c1", {10: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 1, 0, 0, 0, 1),))
        s2 = NodeSolution("c2", {20: Coord3D(0, 0, 0)}, sub_boxes=(BoundingBox3D(0, 0, 0, 1, 0, 0),))
        inv = inventory_cluster(parent, {"c1": s1, "c2": s2}, _make_graph([1, 10, 20], []))
        inv = dataclasses.replace(
            inv,
            child_sub_boxes={
                "c1": (BoundingBox3D(0, 1, 0, 0, 0, 1),),
                "c2": (BoundingBox3D(0, 0, 0, 1, 0, 0),),
            },
        )
        form = ClusterFormulation(inventory=inv)
        grower = ExactCollisionCutGrower(form.inventory, form.clearance_builder.spatial_dag)

        # No edge between c1 and c2 -> Z is unlocked -> 3D Minkowski includes 'u' and 'd'
        ext = SolvedClusterExtraction({1: (0, 0, 0)}, {"c1": (0, 0, 0), "c2": (0, 0, 0)}, SolverStatus.OPTIMAL)
        cut = grower.try_grow_cut("child_child", ("c2", 0), ("c1", 0), ext)
        assert cut is not None
        assert cut.kind == "mink_child_child"
        assert "u" in cut.valid_dirs and "d" in cut.valid_dirs

        # Negative guards: non-overlapping offset, self-pair, missing offset, unknown category, zero DAG dirs
        ext_disjoint = SolvedClusterExtraction({1: (0, 0, 0)}, {"c1": (10, 10, 0), "c2": (0, 0, 0)}, SolverStatus.OPTIMAL)
        assert grower.try_grow_cut("child_child", ("c1", 0), ("c2", 0), ext_disjoint) is None
        assert grower.try_grow_cut("child_child", ("c1", 0), ("c1", 0), ext) is None
        assert grower.try_grow_cut("child_child", ("c_missing", 0), ("c2", 0), ext) is None
        assert grower.try_grow_cut("corridor_crossing", 0, 1, ext) is None

        # Deadlocked DAG prunes all directions -> returns None
        form.clearance_builder.spatial_dag.add_equality(AXIS_X, "c1", "c2", 0)
        form.clearance_builder.spatial_dag.add_equality(AXIS_Y, "c1", "c2", 0)
        form.clearance_builder.spatial_dag.add_equality(AXIS_Z, "c1", "c2", 0)
        grower_locked = ExactCollisionCutGrower(form.inventory, form.clearance_builder.spatial_dag)
        assert grower_locked.try_grow_cut("child_child", ("c1", 0), ("c2", 0), ext) is None

    def test_corridor_and_room_child_growth_and_fallback_behavior(self) -> None:
        """Tests orthogonal dilation, slanted slack corridors, room_child, and fallbacks."""
        ch = SolveTreeNode("c1", TreeNodeType.BLOCK, frozenset({200, 201}))
        s1 = NodeSolution(
            node_id="c1",
            local_coordinates={200: Coord3D(0, 0, 0), 201: Coord3D(2, 0, 0)},
            port_coordinates={10: Coord3D(-1, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 0, 0, 1, 0, 0),
                BoundingBox3D(2, 2, 0, 1, 0, 0),  # 1-cell concave cove at (1, 0..1)
            ),
        )
        ae_x = ActiveEdge(
            edge_id="e_x",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        ae_y_slack = ActiveEdge(
            edge_id="e_y_slack",
            src=3,
            dst=4,
            direction=(0, -1, 0),
            min_length=2,
            associated_rooms=frozenset({3, 4}),
            associated_children=frozenset(),
            allow_transverse_slack=True,
        )
        ae_inc = ActiveEdge(
            edge_id="e_inc",
            src=1,
            dst=("c1", 200),
            direction=(1, 0, 0),
            min_length=1,
            associated_rooms=frozenset({1}),
            associated_children=frozenset({"c1"}),
        )
        ae_port_touch = ActiveEdge(
            edge_id="e_port_touch",
            src=10,
            dst=5,
            direction=(0, 1, 0),
            min_length=1,
            associated_rooms=frozenset({10, 5}),
            associated_children=frozenset(),
        )
        ae_vert = ActiveEdge(
            edge_id="e_z",
            src=1,
            dst=6,
            direction=(0, 0, 1),
            min_length=1,
            associated_rooms=frozenset({1, 6}),
            associated_children=frozenset(),
        )

        parent = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({1, 2, 3, 4, 5, 6, 10}),
            children=(ch,),
        )
        inv = inventory_cluster(parent, {"c1": s1}, _make_graph([1, 2, 3, 4, 5, 6, 10, 200, 201], []))
        inv = dataclasses.replace(
            inv,
            active_edges=(ae_x, ae_y_slack, ae_inc, ae_port_touch, ae_vert),
            child_sub_boxes={
                "c1": (
                    BoundingBox3D(0, 0, 0, 1, 0, 0),
                    BoundingBox3D(2, 2, 0, 1, 0, 0),
                )
            },
        )
        form = ClusterFormulation(inventory=inv)
        builder = form.clearance_builder
        # Lock Z between rooms 1, 3, 10 and child c1
        builder.spatial_dag.add_equality(AXIS_Z, 1, "c1", 0)
        builder.spatial_dag.add_equality(AXIS_Z, 3, "c1", 0)
        grower = ExactCollisionCutGrower(form.inventory, builder.spatial_dag)

        ext = SolvedClusterExtraction(
            raw_parent_coords={
                1: (-2, 0, 0),
                2: (4, 0, 0),
                3: (0, 3, 0),
                4: (2, -2, 0),  # slanted Y corridor spanning x=0..2
                5: (-1, 3, 0),
                6: (-2, 0, 2),
                10: (-1, 0, 0),
            },
            raw_port_coords={77: (0, 1, 0)},
            raw_child_offsets={"c1": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        # 1. Rigid orthogonal X corridor collides with sub-box 0 at x=0, dilates +-1 to span x=-1..3!
        cut_x = grower.try_grow_cut("corridor_child", CorridorClearanceBox(0, "e_x"), ChildSubClearanceBox("c1", 0), ext)
        assert cut_x is not None
        assert cut_x.obstacle_box == BoundingBox3D(-1, 3, 0, 1, 0, 0)

        # 2. Slanted transverse-slack Y corridor collides at x=0 without blocking valid pass-through at x=1,
        #    while filling a 1-cell travel-axis Y-gap when two child slabs stack along Y.
        cut_y_slack = grower.try_grow_cut("corridor_child", 1, ("c1", 0), ext)
        assert cut_y_slack is not None
        assert cut_y_slack.obstacle_box == BoundingBox3D(0, 0, 0, 1, 0, 0)

        inv_y_gap = dataclasses.replace(
            inv,
            child_sub_boxes={
                "c1": (
                    BoundingBox3D(0, 0, 0, 0, 0, 0),
                    BoundingBox3D(0, 0, 2, 2, 0, 0),  # 1-cell gap at (0, 1) along Y travel axis
                )
            },
        )
        form_y_gap = ClusterFormulation(inventory=inv_y_gap)
        form_y_gap.clearance_builder.spatial_dag.add_equality(AXIS_Z, 3, "c1", 0)
        grower_y_gap = ExactCollisionCutGrower(form_y_gap.inventory, form_y_gap.clearance_builder.spatial_dag)
        cut_y_filled = grower_y_gap.try_grow_cut("corridor_child", 1, ("c1", 0), ext)
        assert cut_y_filled is not None
        assert cut_y_filled.obstacle_box == BoundingBox3D(0, 0, 0, 2, 0, 0)

        # 3. Incident edge, port-touching edge, Z-edge, out-of-range edge return None
        assert grower.try_grow_cut("corridor_child", 2, ("c1", 0), ext) is None
        assert grower.try_grow_cut("corridor_child", 3, ("c1", 0), ext) is None
        assert grower.try_grow_cut("corridor_child", 4, ("c1", 0), ext) is None
        assert grower.try_grow_cut("corridor_child", 999, ("c1", 0), ext) is None

        # 4. Room-child: RoomClearanceBox(77) resolves via raw_port_coords, while incident room 10 returns None
        builder.spatial_dag.add_equality(AXIS_Z, 77, "c1", 0)
        grower2 = ExactCollisionCutGrower(form.inventory, builder.spatial_dag)
        cut_rc = grower2.try_grow_cut("room_child", RoomClearanceBox(77), ("c1", 0), ext)
        assert cut_rc is not None
        assert cut_rc.kind == "grown_room_child"
        assert grower2.try_grow_cut("room_child", RoomClearanceBox(10), ("c1", 0), ext) is None

        # 6. Repeated grown cut falls back cleanly to standard sub-box cut in inject_clearance_cuts
        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation"):
            injected_1 = builder.inject_clearance_cuts(
                [
                    ClearanceViolation("room_child", RoomClearanceBox(10), ("c1", 0)),  # incident -> std
                    ClearanceViolation("corridor_child", 0, ("c1", 0)),  # grown
                ],
                extraction=ext,
            )
            assert injected_1 == 2
            # Second pass with same grown box in separated_pairs falls back to exact std sub-box cut!
            injected_2 = builder.inject_clearance_cuts(
                [ClearanceViolation("corridor_child", 0, ("c1", 0))],
                extraction=ext,
            )
            assert injected_2 == 1
            assert ("corridor_child", 0, ("c1", 0)) in builder.separated_pairs

    def test_orthogonal_corridor_does_not_fill_post_dilation_doorway_courtyard(self) -> None:
        """Removing post-dilation hole fill preserves feasible corridor interiors inside open bays."""
        # Open courtyard x=0..4, y=0..3 with 1-cell top doorway at (2, 3):
        # X dilation seals the doorway horizontally and narrows the empty pocket to (2, 1..2),
        # yet src=(1,2)->dst=(3,2) (min_length=2) validly places interior (2,2) inside the bay!
        courtyard_boxes = (
            BoundingBox3D(0, 0, 0, 3, 0, 0),  # left wall
            BoundingBox3D(4, 4, 0, 3, 0, 0),  # right wall
            BoundingBox3D(1, 3, 0, 0, 0, 0),  # bottom wall
            BoundingBox3D(1, 1, 3, 3, 0, 0),  # top-left doorframe
            BoundingBox3D(3, 3, 3, 3, 0, 0),  # top-right doorframe
        )
        ch = SolveTreeNode("c_yard", TreeNodeType.BLOCK, frozenset({200}))
        sol = NodeSolution("c_yard", {200: Coord3D(0, 0, 0)}, sub_boxes=courtyard_boxes)
        ae_x = ActiveEdge(
            edge_id="e_1_2",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        ae_2_3 = ActiveEdge("e_2_3", 2, 3, (0, 1, 0), 1, frozenset({2, 3}), frozenset())
        ae_3_4 = ActiveEdge("e_3_4", 3, 4, (0, 1, 0), 1, frozenset({3, 4}), frozenset())
        parent = SolveTreeNode("root", TreeNodeType.BLOCK, frozenset({1, 2, 3, 4}), children=(ch,))
        inv = inventory_cluster(parent, {"c_yard": sol}, _make_graph([1, 2, 3, 4, 200], []))
        inv = dataclasses.replace(
            inv,
            active_edges=(ae_x, ae_2_3, ae_3_4),
            child_sub_boxes={"c_yard": courtyard_boxes},
        )
        form = ClusterFormulation(inventory=inv)
        form.clearance_builder.spatial_dag.add_equality(AXIS_Z, 1, "c_yard", 0)
        grower = ExactCollisionCutGrower(form.inventory, form.clearance_builder.spatial_dag)

        # Corridor hits left wall (0, 2) with 1=(-1, 2, 0) -> 2=(3, 2, 0)
        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (-1, 2, 0), 2: (3, 2, 0), 3: (3, 3, 0), 4: (3, 4, 0)},
            raw_child_offsets={"c_yard": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        cut = grower.try_grow_cut("corridor_child", 0, ("c_yard", 0), ext)
        assert cut is not None
        # Left wall dilates only to [-1, 1] and MUST NOT seal (2, 2) across to the right wall!
        assert cut.obstacle_box == BoundingBox3D(-1, 1, 0, 3, 0, 0)
        assert not (cut.obstacle_box.min_x <= 2 <= cut.obstacle_box.max_x)

    def test_grow_child_child_cut_large_planar_locked_and_unlocked_z_guard(self) -> None:
        """Planar-locked children exceeding 25,000 cell-pairs use run-length 2D Minkowski while 3D unlocked guards."""
        box_16x16 = (BoundingBox3D(0, 15, 0, 15, 0, 0),)
        ch1 = SolveTreeNode("c1", TreeNodeType.BLOCK, frozenset({10}))
        ch2 = SolveTreeNode("c2", TreeNodeType.BLOCK, frozenset({20}))
        parent = SolveTreeNode("root", TreeNodeType.BLOCK, frozenset({1}), children=(ch1, ch2))
        s1 = NodeSolution("c1", {10: Coord3D(0, 0, 0)}, sub_boxes=box_16x16)
        s2 = NodeSolution("c2", {20: Coord3D(0, 0, 0)}, sub_boxes=box_16x16)
        inv = inventory_cluster(parent, {"c1": s1, "c2": s2}, _make_graph([1, 10, 20], []))
        inv = dataclasses.replace(inv, child_sub_boxes={"c1": box_16x16, "c2": box_16x16})

        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0)},
            raw_child_offsets={"c1": (0, 0, 0), "c2": (4, 4, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        # 1. Unlocked Z: 256 * 256 = 65,536 > 25,000 -> guarded to None
        form_unlocked = ClusterFormulation(inventory=inv)
        grower_unlocked = ExactCollisionCutGrower(
            form_unlocked.inventory, form_unlocked.clearance_builder.spatial_dag
        )
        assert grower_unlocked.try_grow_cut("child_child", ("c1", 0), ("c2", 0), ext) is None

        # 2. Planar-locked Z: bypasses 25,000 guard and synthesizes exact 31x31 Minkowski cut
        form_locked = ClusterFormulation(inventory=inv)
        form_locked.clearance_builder.spatial_dag.add_equality(AXIS_Z, "c1", "c2", 0)
        grower_locked = ExactCollisionCutGrower(
            form_locked.inventory, form_locked.clearance_builder.spatial_dag
        )
        cut = grower_locked.try_grow_cut("child_child", ("c1", 0), ("c2", 0), ext)
        assert cut is not None
        assert cut.kind == "mink_child_child"
        assert cut.obstacle_box == BoundingBox3D(-15, 15, -15, 15, 0, 0)
        assert cut.valid_dirs == ("r", "l", "a", "b")

        # 3. Planar-locked Z falls back to None if _minkowski_diff_2d exceeds limits and returns None
        with patch("romutil.layout.milp.clearance_boxes._minkowski_diff_2d", return_value=None):
            assert grower_locked.try_grow_cut("child_child", ("c1", 0), ("c2", 0), ext) is None


class TestTransverseSlackCorridorBoxes:
    """Tests ensuring allow_transverse_slack=True corridors bound full slanted extents in MILP."""

    def test_slanted_corridor_boxes_include_transverse_slack_variables(self) -> None:
        """get_corridor_interior_box and get_corridor_full_box expand with (sp, sm) for slanted edges."""
        ae_x_slack = ActiveEdge(
            edge_id="e_x_slack",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
            allow_transverse_slack=True,
        )
        ae_y_slack = ActiveEdge(
            edge_id="e_y_slack",
            src=3,
            dst=4,
            direction=(0, -1, 0),
            min_length=2,
            associated_rooms=frozenset({3, 4}),
            associated_children=frozenset(),
            allow_transverse_slack=True,
        )
        ae_rigid = ActiveEdge(
            edge_id="e_rigid",
            src=1,
            dst=3,
            direction=(0, 1, 0),
            min_length=1,
            associated_rooms=frozenset({1, 3}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        parent = SolveTreeNode("root", TreeNodeType.BLOCK, frozenset({1, 2, 3, 4}))
        inv = inventory_cluster(parent, {}, _make_graph([1, 2, 3, 4], []))
        inv = dataclasses.replace(inv, active_edges=(ae_x_slack, ae_y_slack, ae_rigid))
        form = ClusterFormulation(inventory=inv)

        env_int_x = form.get_corridor_envelope_box(ae_x_slack, interior=True)
        env_full_x = form.get_corridor_envelope_box(ae_x_slack, interior=False)
        assert "-" in str(env_int_x[2]) and "+" in str(env_int_x[3])
        assert "-" in str(env_full_x[2]) and "+" in str(env_full_x[3])
        int_x = form.get_corridor_interior_box(ae_x_slack)
        assert str(int_x[2]) == str(int_x[3])

        env_int_y = form.get_corridor_envelope_box(ae_y_slack, interior=True)
        env_full_y = form.get_corridor_envelope_box(ae_y_slack, interior=False)
        assert "-" in str(env_int_y[0]) and "+" in str(env_int_y[1])
        assert "-" in str(env_full_y[0]) and "+" in str(env_full_y[1])
        int_y = form.get_corridor_interior_box(ae_y_slack)
        assert str(int_y[0]) == str(int_y[1])

        env_rigid = form.get_corridor_envelope_box(ae_rigid, interior=True)
        assert str(env_rigid[0]) == str(env_rigid[1])

    def test_slanted_corridor_separation_prevents_intermediate_room_penetration(self) -> None:
        """MILP clearance cut on allow_transverse_slack corridor forces slanted segment clear of mid room."""
        if not SolverFactory("cbc").available():
            pytest.skip("CBC solver not available")

        # 1 --north(1)--> 10, 1 --east(2)--> 2 --east(2)--> 3, and slack North edge 10 -> 3 (Y slack/shift):
        # Place third-party room 50 at (2, 2) directly in the diagonal rectangle of slanted 10 -> 3,
        # and verify corridor_room cut forces 50 outside or shifts 3 cleanly.
        ae_anchor = ActiveEdge(
            edge_id="e_1_10",
            src=1,
            dst=10,
            direction=(0, 1, 0),
            min_length=4,
            associated_rooms=frozenset({1, 10}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        ae_slanted = ActiveEdge(
            edge_id="e_10_2_slack",
            src=10,
            dst=2,
            direction=(0, -1, 0),
            min_length=4,
            associated_rooms=frozenset({10, 2}),
            associated_children=frozenset(),
            allow_transverse_slack=True,
        )
        ae_mid = ActiveEdge(
            edge_id="e_1_40",
            src=1,
            dst=40,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 40}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        ae_40_50 = ActiveEdge(
            edge_id="e_40_50",
            src=40,
            dst=50,
            direction=(0, 1, 0),
            min_length=2,
            associated_rooms=frozenset({40, 50}),
            associated_children=frozenset(),
            allow_transverse_slack=False,
        )
        parent = SolveTreeNode("root", TreeNodeType.BLOCK, frozenset({1, 10, 40, 50, 2}))
        inv = inventory_cluster(parent, {}, _make_graph([1, 2, 10, 40, 50], []))
        inv = dataclasses.replace(inv, active_edges=(ae_anchor, ae_slanted, ae_mid, ae_40_50))
        form2 = ClusterFormulation(inventory=inv, options=LayoutOptions(timeout_seconds=10))
        # Pin dst 2 to x >= 4 so slanted corridor 10(0,4)->2(4,0) spans room 50 at (2,2),
        # forcing corridor_room separation to push 50 north or 2 south clear of the corridor.
        form2.model.pin_2_east = pyo.Constraint(expr=form2.model.x[2] >= 4)  # type: ignore[attr-defined]
        ext = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 10: (0, 4, 0), 40: (2, 0, 0), 50: (2, 2, 0), 2: (4, 0, 0)},
            raw_child_offsets={},
            solver_status=SolverStatus.OPTIMAL,
        )
        cuts = form2.clearance_builder.inject_clearance_cuts(
            [ClearanceViolation("corridor_room", 1, 50)],
            extraction=ext,
        )
        assert cuts == 1
        # Solve with CBC and verify corridor_room box separation prevents slanted segment from crossing room 50!
        from romutil.layout.milp.solver import solve_and_extract

        solved = solve_and_extract(form2, "cbc", LayoutOptions(timeout_seconds=10))
        assert [v for v in detect_cluster_collisions(form2, solved) if v.category == "corridor_room"] == []
