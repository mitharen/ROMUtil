"""Comprehensive unit tests for 3D layout clearance primitives and invariants."""

from __future__ import annotations

import pytest

from romutil.layout import (
    BoundingBox3D,
    Segment3D,
    point_in_segment_interior,
    segment_box_penetration,
    segment_room_penetration,
    segments_intersect_3d,
)
from romutil.types import Coord3D


class TestBoundingBox3D:
    """Tests for BoundingBox3D data structure and operations."""

    def test_construction_and_validation(self) -> None:
        box = BoundingBox3D(min_x=-5, max_x=10, min_y=0, max_y=20, min_z=-2, max_z=3)
        assert box.min_x == -5
        assert box.max_x == 10
        assert box.min_y == 0
        assert box.max_y == 20
        assert box.min_z == -2
        assert box.max_z == 3

        # Invalid bounds where min > max
        with pytest.raises(ValueError, match="Invalid bounding box boundaries"):
            BoundingBox3D(min_x=5, max_x=2, min_y=0, max_y=1, min_z=0, max_z=1)

        with pytest.raises(ValueError, match="Invalid bounding box boundaries"):
            BoundingBox3D(min_x=0, max_x=1, min_y=5, max_y=2, min_z=0, max_z=1)

        with pytest.raises(ValueError, match="Invalid bounding box boundaries"):
            BoundingBox3D(min_x=0, max_x=1, min_y=0, max_y=1, min_z=5, max_z=2)

    def test_single_point_box(self) -> None:
        box = BoundingBox3D(min_x=3, max_x=3, min_y=-1, max_y=-1, min_z=0, max_z=0)
        assert box.dimensions() == (1, 1, 1)
        assert box.volume() == 1
        assert box.contains_point(Coord3D(3, -1, 0))
        assert not box.contains_point(Coord3D(4, -1, 0))

    def test_volume_and_dimensions(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=3, min_y=-2, max_y=2, min_z=1, max_z=3)
        # width = 3 - 0 + 1 = 4
        # height = 2 - (-2) + 1 = 5
        # depth = 3 - 1 + 1 = 3
        assert box.dimensions() == (4, 5, 3)
        assert box.volume() == 4 * 5 * 3 == 60

    def test_contains_point(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        # Inside
        assert box.contains_point(Coord3D(5, 5, 5))
        # On corners and faces
        assert box.contains_point(Coord3D(0, 0, 0))
        assert box.contains_point(Coord3D(10, 10, 10))
        assert box.contains_point(Coord3D(0, 5, 5))
        assert box.contains_point(Coord3D(10, 5, 5))
        assert box.contains_point(Coord3D(5, 0, 5))
        assert box.contains_point(Coord3D(5, 10, 5))
        assert box.contains_point(Coord3D(5, 5, 0))
        assert box.contains_point(Coord3D(5, 5, 10))
        # Outside
        assert not box.contains_point(Coord3D(-1, 5, 5))
        assert not box.contains_point(Coord3D(11, 5, 5))
        assert not box.contains_point(Coord3D(5, -1, 5))
        assert not box.contains_point(Coord3D(5, 11, 5))
        assert not box.contains_point(Coord3D(5, 5, -1))
        assert not box.contains_point(Coord3D(5, 5, 11))

    def test_expand(self) -> None:
        box = BoundingBox3D(min_x=1, max_x=3, min_y=2, max_y=4, min_z=5, max_z=7)
        expanded = box.expand(2)
        assert expanded == BoundingBox3D(min_x=-1, max_x=5, min_y=0, max_y=6, min_z=3, max_z=9)

        # Default expand amount is 1
        expanded_default = box.expand()
        assert expanded_default == BoundingBox3D(min_x=0, max_x=4, min_y=1, max_y=5, min_z=4, max_z=8)

    def test_intersects(self) -> None:
        b1 = BoundingBox3D(min_x=0, max_x=4, min_y=0, max_y=4, min_z=0, max_z=4)

        # Overlapping interior
        b2 = BoundingBox3D(min_x=2, max_x=6, min_y=2, max_y=6, min_z=2, max_z=6)
        assert b1.intersects(b2)
        assert b2.intersects(b1)

        # Touching at face
        b3 = BoundingBox3D(min_x=4, max_x=8, min_y=0, max_y=4, min_z=0, max_z=4)
        assert b1.intersects(b3)
        assert b3.intersects(b1)

        # Touching at single corner
        b4 = BoundingBox3D(min_x=4, max_x=8, min_y=4, max_y=8, min_z=4, max_z=8)
        assert b1.intersects(b4)
        assert b4.intersects(b1)

        # Disjoint along X
        b5 = BoundingBox3D(min_x=5, max_x=8, min_y=0, max_y=4, min_z=0, max_z=4)
        assert not b1.intersects(b5)
        assert not b5.intersects(b1)

        # Disjoint along Y
        b6 = BoundingBox3D(min_x=0, max_x=4, min_y=5, max_y=8, min_z=0, max_z=4)
        assert not b1.intersects(b6)
        assert not b6.intersects(b1)

        # Disjoint along Z
        b7 = BoundingBox3D(min_x=0, max_x=4, min_y=0, max_y=4, min_z=-5, max_z=-1)
        assert not b1.intersects(b7)
        assert not b7.intersects(b1)

    def test_from_coordinates(self) -> None:
        coords = [
            Coord3D(1, 0, 0),
            Coord3D(-4, -5, -3),
            Coord3D(6, 7, 8),
        ]
        box = BoundingBox3D.from_coordinates(coords)
        assert box == BoundingBox3D(min_x=-4, max_x=6, min_y=-5, max_y=7, min_z=-3, max_z=8)

        # Single coordinate
        box_single = BoundingBox3D.from_coordinates([Coord3D(7, 8, 9)])
        assert box_single == BoundingBox3D(min_x=7, max_x=7, min_y=8, max_y=8, min_z=9, max_z=9)

        # Empty coordinate list
        with pytest.raises(ValueError, match="Cannot create BoundingBox3D from empty coordinates"):
            BoundingBox3D.from_coordinates([])


class TestSegment3D:
    """Tests for Segment3D data structure and operations."""

    def test_point_segment(self) -> None:
        pt = Coord3D(3, 4, 5)
        seg = Segment3D(pt, pt)
        assert seg.is_point()
        assert seg.is_axis_aligned()
        assert seg.bounding_box() == BoundingBox3D(3, 3, 4, 4, 5, 5)

    def test_is_axis_aligned(self) -> None:
        # X axis
        assert Segment3D(Coord3D(0, 0, 0), Coord3D(5, 0, 0)).is_axis_aligned()
        # Y axis
        assert Segment3D(Coord3D(0, 0, 0), Coord3D(0, -3, 0)).is_axis_aligned()
        # Z axis
        assert Segment3D(Coord3D(0, 0, 0), Coord3D(0, 0, 2)).is_axis_aligned()
        # Point
        assert Segment3D(Coord3D(1, 1, 1), Coord3D(1, 1, 1)).is_axis_aligned()
        # 2D Diagonal (False)
        assert not Segment3D(Coord3D(0, 0, 0), Coord3D(1, 1, 0)).is_axis_aligned()
        # 3D Diagonal (False)
        assert not Segment3D(Coord3D(0, 0, 0), Coord3D(1, 1, 1)).is_axis_aligned()

    def test_bounding_box(self) -> None:
        seg = Segment3D(Coord3D(5, -2, 10), Coord3D(-1, 4, 3))
        assert seg.bounding_box() == BoundingBox3D(
            min_x=-1, max_x=5, min_y=-2, max_y=4, min_z=3, max_z=10
        )


class TestPointInSegmentInterior:
    """Tests for point_in_segment_interior helper."""

    def test_axis_aligned_interior(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(4, 0, 0))
        assert point_in_segment_interior(Coord3D(1, 0, 0), seg)
        assert point_in_segment_interior(Coord3D(2, 0, 0), seg)
        assert point_in_segment_interior(Coord3D(3, 0, 0), seg)

        # Endpoints are excluded
        assert not point_in_segment_interior(Coord3D(0, 0, 0), seg)
        assert not point_in_segment_interior(Coord3D(4, 0, 0), seg)

        # Collinear outside
        assert not point_in_segment_interior(Coord3D(-1, 0, 0), seg)
        assert not point_in_segment_interior(Coord3D(5, 0, 0), seg)

        # Off-axis
        assert not point_in_segment_interior(Coord3D(2, 1, 0), seg)
        assert not point_in_segment_interior(Coord3D(2, 0, 1), seg)

    def test_negative_direction_segment(self) -> None:
        seg = Segment3D(Coord3D(0, 4, 0), Coord3D(0, -4, 0))
        assert point_in_segment_interior(Coord3D(0, 0, 0), seg)
        assert point_in_segment_interior(Coord3D(0, 2, 0), seg)
        assert point_in_segment_interior(Coord3D(0, -3, 0), seg)

        assert not point_in_segment_interior(Coord3D(0, 4, 0), seg)
        assert not point_in_segment_interior(Coord3D(0, -4, 0), seg)
        assert not point_in_segment_interior(Coord3D(0, 5, 0), seg)
        assert not point_in_segment_interior(Coord3D(0, -5, 0), seg)

        # Negative direction along X
        seg_x = Segment3D(Coord3D(10, 0, 0), Coord3D(0, 0, 0))
        assert point_in_segment_interior(Coord3D(5, 0, 0), seg_x)
        assert not point_in_segment_interior(Coord3D(15, 0, 0), seg_x)
        assert not point_in_segment_interior(Coord3D(-5, 0, 0), seg_x)

        # Negative direction along Z
        seg_z = Segment3D(Coord3D(0, 0, 10), Coord3D(0, 0, 0))
        assert point_in_segment_interior(Coord3D(0, 0, 5), seg_z)
        assert not point_in_segment_interior(Coord3D(0, 0, 15), seg_z)
        assert not point_in_segment_interior(Coord3D(0, 0, -5), seg_z)

    def test_yz_cross_ratio_mismatch(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 4, 4))
        assert point_in_segment_interior(Coord3D(0, 2, 2), seg)
        assert not point_in_segment_interior(Coord3D(0, 1, 2), seg)
        assert not point_in_segment_interior(Coord3D(0, 3, 1), seg)

    def test_z_axis_interior(self) -> None:
        seg = Segment3D(Coord3D(1, 1, 0), Coord3D(1, 1, 5))
        assert point_in_segment_interior(Coord3D(1, 1, 3), seg)
        assert not point_in_segment_interior(Coord3D(1, 1, 0), seg)
        assert not point_in_segment_interior(Coord3D(1, 1, 5), seg)
        assert not point_in_segment_interior(Coord3D(1, 1, 6), seg)

    def test_diagonal_interior(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(4, 4, 4))
        assert point_in_segment_interior(Coord3D(2, 2, 2), seg)
        assert not point_in_segment_interior(Coord3D(2, 2, 3), seg)
        assert not point_in_segment_interior(Coord3D(2, 3, 2), seg)
        assert not point_in_segment_interior(Coord3D(3, 2, 2), seg)

    def test_degenerate_point_segment(self) -> None:
        pt = Coord3D(2, 3, 4)
        seg = Segment3D(pt, pt)
        assert not point_in_segment_interior(pt, seg)
        assert not point_in_segment_interior(Coord3D(0, 0, 0), seg)


class TestSegmentRoomPenetration:
    """Tests for segment_room_penetration."""

    def test_positive_penetration(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 10, 0))
        assert segment_room_penetration(seg, Coord3D(0, 5, 0))

    def test_endpoint_rooms_allowed(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 10, 0))
        assert not segment_room_penetration(seg, Coord3D(0, 0, 0))
        assert not segment_room_penetration(seg, Coord3D(0, 10, 0))

    def test_collinear_outside_room_allowed(self) -> None:
        seg = Segment3D(Coord3D(0, 2, 0), Coord3D(0, 8, 0))
        assert not segment_room_penetration(seg, Coord3D(0, 0, 0))
        assert not segment_room_penetration(seg, Coord3D(0, 10, 0))

    def test_parallel_displaced_room_allowed(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 10, 0))
        assert not segment_room_penetration(seg, Coord3D(1, 5, 0))

    def test_skew_z_room_allowed(self) -> None:
        seg = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 10, 0))
        assert not segment_room_penetration(seg, Coord3D(0, 5, 1))


class TestSegmentsIntersect3D:
    """Tests for 3D segment intersection logic."""

    def test_coplanar_orthogonal_crossing(self) -> None:
        # Perpendicular crossing at (0, 0, 0)
        s1 = Segment3D(Coord3D(0, -2, 0), Coord3D(0, 2, 0))
        s2 = Segment3D(Coord3D(-2, 0, 0), Coord3D(2, 0, 0))
        assert segments_intersect_3d(s1, s2)
        assert segments_intersect_3d(s2, s1)

    def test_coplanar_orthogonal_non_crossing(self) -> None:
        s1 = Segment3D(Coord3D(0, -2, 0), Coord3D(0, 2, 0))
        s2 = Segment3D(Coord3D(5, 0, 0), Coord3D(10, 0, 0))
        assert not segments_intersect_3d(s1, s2)
        assert not segments_intersect_3d(s2, s1)

    def test_skew_segments_xy_projection_overlap(self) -> None:
        # Crosses in XY projection, but z=0 vs z=1 (disjoint bbox)
        s1 = Segment3D(Coord3D(0, -2, 0), Coord3D(0, 2, 0))
        s2 = Segment3D(Coord3D(-2, 0, 1), Coord3D(2, 0, 1))
        assert not segments_intersect_3d(s1, s2)
        assert not segments_intersect_3d(s2, s1)

        # Skew lines whose 3D bounding boxes DO intersect
        s3 = Segment3D(Coord3D(-2, 0, 0), Coord3D(2, 0, 2))
        s4 = Segment3D(Coord3D(0, -2, 3), Coord3D(0, 2, 1))
        assert not segments_intersect_3d(s3, s4)
        assert not segments_intersect_3d(s4, s3)

    def test_incident_segments_sharing_endpoint(self) -> None:
        # Meeting at an angle at common vertex (2, 0, 0)
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(2, 0, 0))
        s2 = Segment3D(Coord3D(2, 0, 0), Coord3D(2, 2, 0))
        assert not segments_intersect_3d(s1, s2)
        assert not segments_intersect_3d(s2, s1)

    def test_collinear_overlapping(self) -> None:
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(4, 0, 0))
        s2 = Segment3D(Coord3D(2, 0, 0), Coord3D(6, 0, 0))
        assert segments_intersect_3d(s1, s2)
        assert segments_intersect_3d(s2, s1)

    def test_collinear_overlapping_reversed(self) -> None:
        s1 = Segment3D(Coord3D(4, 0, 0), Coord3D(0, 0, 0))
        s2 = Segment3D(Coord3D(2, 0, 0), Coord3D(6, 0, 0))
        assert segments_intersect_3d(s1, s2)
        assert segments_intersect_3d(s2, s1)

    def test_collinear_overlapping_sharing_endpoint(self) -> None:
        # Shared endpoint (4, 0, 0), but overlapping along [2, 4]
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(4, 0, 0))
        s2 = Segment3D(Coord3D(2, 0, 0), Coord3D(4, 0, 0))
        assert segments_intersect_3d(s1, s2)
        assert segments_intersect_3d(s2, s1)

    def test_collinear_abutting_sharing_endpoint(self) -> None:
        # Touching only at single endpoint (2, 0, 0)
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(2, 0, 0))
        s2 = Segment3D(Coord3D(2, 0, 0), Coord3D(4, 0, 0))
        assert not segments_intersect_3d(s1, s2)
        assert not segments_intersect_3d(s2, s1)

    def test_collinear_disjoint(self) -> None:
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(1, 0, 0))
        s2 = Segment3D(Coord3D(3, 0, 0), Coord3D(4, 0, 0))
        assert not segments_intersect_3d(s1, s2)
        assert not segments_intersect_3d(s2, s1)

    def test_collinear_y_and_z_axes(self) -> None:
        # Y axis overlapping
        s1_y = Segment3D(Coord3D(1, 0, 2), Coord3D(1, 5, 2))
        s2_y = Segment3D(Coord3D(1, 3, 2), Coord3D(1, 8, 2))
        assert segments_intersect_3d(s1_y, s2_y)

        # Z axis overlapping
        s1_z = Segment3D(Coord3D(1, 2, 0), Coord3D(1, 2, 5))
        s2_z = Segment3D(Coord3D(1, 2, 2), Coord3D(1, 2, 7))
        assert segments_intersect_3d(s1_z, s2_z)

    def test_parallel_non_collinear(self) -> None:
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 4, 0))
        s2 = Segment3D(Coord3D(1, 0, 0), Coord3D(1, 4, 0))
        assert not segments_intersect_3d(s1, s2)
        assert not segments_intersect_3d(s2, s1)

        # Parallel non-collinear with overlapping bounding boxes
        s3 = Segment3D(Coord3D(0, 0, 0), Coord3D(2, 2, 0))
        s4 = Segment3D(Coord3D(0, 1, 0), Coord3D(2, 3, 0))
        assert not segments_intersect_3d(s3, s4)
        assert not segments_intersect_3d(s4, s3)

    def test_point_segments(self) -> None:
        # Two equal points share endpoint -> False
        pt = Coord3D(1, 2, 3)
        s_pt1 = Segment3D(pt, pt)
        s_pt2 = Segment3D(pt, pt)
        assert not segments_intersect_3d(s_pt1, s_pt2)

        # Point inside segment interior -> True
        s_line = Segment3D(Coord3D(0, 0, 0), Coord3D(4, 0, 0))
        s_mid = Segment3D(Coord3D(2, 0, 0), Coord3D(2, 0, 0))
        assert segments_intersect_3d(s_line, s_mid)
        assert segments_intersect_3d(s_mid, s_line)

        # Point at segment endpoint -> False (incident)
        s_end = Segment3D(Coord3D(0, 0, 0), Coord3D(0, 0, 0))
        assert not segments_intersect_3d(s_line, s_end)
        assert not segments_intersect_3d(s_end, s_line)

        # Point outside segment -> False
        s_out = Segment3D(Coord3D(5, 0, 0), Coord3D(5, 0, 0))
        assert not segments_intersect_3d(s_line, s_out)

    def test_non_integer_crossing(self) -> None:
        # Diagonal segments in XY plane crossing at (0.5, 0.5, 0)
        s1 = Segment3D(Coord3D(0, 0, 0), Coord3D(1, 1, 0))
        s2 = Segment3D(Coord3D(0, 1, 0), Coord3D(1, 0, 0))
        assert segments_intersect_3d(s1, s2)
        assert segments_intersect_3d(s2, s1)

    def test_coplanar_crossing_in_yz_and_zx(self) -> None:
        # Crossing in YZ plane (x=2)
        s1_yz = Segment3D(Coord3D(2, 0, -2), Coord3D(2, 0, 2))
        s2_yz = Segment3D(Coord3D(2, -2, 0), Coord3D(2, 2, 0))
        assert segments_intersect_3d(s1_yz, s2_yz)

        # Crossing in ZX plane (y=3)
        s1_zx = Segment3D(Coord3D(-2, 3, 0), Coord3D(2, 3, 0))
        s2_zx = Segment3D(Coord3D(0, 3, -2), Coord3D(0, 3, 2))
        assert segments_intersect_3d(s1_zx, s2_zx)


class TestSegmentBoxPenetration:
    """Tests for segment_box_penetration."""

    def test_interior_penetration_straight_through(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        seg = Segment3D(Coord3D(-2, 5, 5), Coord3D(12, 5, 5))
        assert segment_box_penetration(seg, box)

    def test_completely_outside(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        seg = Segment3D(Coord3D(20, 20, 20), Coord3D(30, 30, 30))
        assert not segment_box_penetration(seg, box)

    def test_terminating_on_boundary_surface(self) -> None:
        # Gateway doorway connecting to room on cluster boundary
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        # Terminating at face x=0 from outside
        seg = Segment3D(Coord3D(-5, 5, 5), Coord3D(0, 5, 5))
        assert not segment_box_penetration(seg, box)

        # Terminating at face x=10 from outside
        seg2 = Segment3D(Coord3D(15, 5, 5), Coord3D(10, 5, 5))
        assert not segment_box_penetration(seg2, box)

    def test_running_along_boundary_surface(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        # Runs on x=0 surface
        seg = Segment3D(Coord3D(0, 2, 2), Coord3D(0, 8, 8))
        assert not segment_box_penetration(seg, box)

        # Runs on edge (x=0, y=0)
        seg_edge = Segment3D(Coord3D(0, 0, 1), Coord3D(0, 0, 9))
        assert not segment_box_penetration(seg_edge, box)

    def test_starting_inside_and_exiting(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        seg = Segment3D(Coord3D(5, 5, 5), Coord3D(15, 5, 5))
        assert segment_box_penetration(seg, box)

    def test_completely_inside(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        seg = Segment3D(Coord3D(2, 3, 4), Coord3D(8, 7, 6))
        assert segment_box_penetration(seg, box)

    def test_starting_inside_terminating_on_boundary(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        # Starts strictly inside (5, 5, 5) and touches boundary (10, 5, 5)
        seg = Segment3D(Coord3D(5, 5, 5), Coord3D(10, 5, 5))
        assert segment_box_penetration(seg, box)

    def test_diagonal_passing_corner_outside(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        # Diagonal from (5, 15, 5) to (15, 5, 5) touches corner (10, 10, 5) without entering interior
        seg = Segment3D(Coord3D(5, 15, 5), Coord3D(15, 5, 5))
        assert not segment_box_penetration(seg, box)

    def test_point_segment_in_box(self) -> None:
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10)
        # Strictly inside
        assert segment_box_penetration(Segment3D(Coord3D(5, 5, 5), Coord3D(5, 5, 5)), box)
        # On boundary face
        assert not segment_box_penetration(Segment3D(Coord3D(0, 5, 5), Coord3D(0, 5, 5)), box)
        # Outside
        assert not segment_box_penetration(Segment3D(Coord3D(15, 5, 5), Coord3D(15, 5, 5)), box)

    def test_flat_box_zero_thickness(self) -> None:
        # Flat in Z: strict 3D interior is empty
        box = BoundingBox3D(min_x=0, max_x=10, min_y=0, max_y=10, min_z=5, max_z=5)
        seg = Segment3D(Coord3D(5, 5, 0), Coord3D(5, 5, 10))
        assert not segment_box_penetration(seg, box)
