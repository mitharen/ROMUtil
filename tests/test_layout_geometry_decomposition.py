"""Unit tests for discrete orthogonal bounding box decomposition."""

from __future__ import annotations

import pytest

from romutil.layout.geometry import BoundingBox3D, decompose_into_orthogonal_sub_boxes
from romutil.types import Coord3D


def _get_box_cells(box: BoundingBox3D) -> set[Coord3D]:
    """Helper returning the set of discrete integer cells enclosed by a bounding box."""
    return {
        Coord3D(x, y, z)
        for x in range(box.min_x, box.max_x + 1)
        for y in range(box.min_y, box.max_y + 1)
        for z in range(box.min_z, box.max_z + 1)
    }


class TestDecomposeIntoOrthogonalSubBoxes:
    """Test suite for decompose_into_orthogonal_sub_boxes computational geometry algorithm."""

    def test_empty_coordinates(self) -> None:
        """Empty coordinate sequence returns an empty tuple."""
        assert decompose_into_orthogonal_sub_boxes([]) == ()
        assert decompose_into_orthogonal_sub_boxes(set()) == ()

    def test_single_point(self) -> None:
        """Single coordinate returns a degenerate single unit box enclosing that point."""
        pt = Coord3D(3, -5, 2)
        boxes = decompose_into_orthogonal_sub_boxes([pt])
        assert len(boxes) == 1
        box = boxes[0]
        assert box.min_x == 3 and box.max_x == 3
        assert box.min_y == -5 and box.max_y == -5
        assert box.min_z == 2 and box.max_z == 2
        assert box.volume() == 1

    def test_convex_solid_box(self) -> None:
        """Convex solid 3D grid returns a single bounding box matching the full bounds."""
        coords = [
            Coord3D(x, y, z)
            for x in range(0, 3)
            for y in range(0, 4)
            for z in range(0, 2)
        ]
        assert len(coords) == 24
        boxes = decompose_into_orthogonal_sub_boxes(coords)
        assert len(boxes) == 1
        box = boxes[0]
        assert (box.min_x, box.max_x) == (0, 2)
        assert (box.min_y, box.max_y) == (0, 3)
        assert (box.min_z, box.max_z) == (0, 1)
        assert box.volume() == 24

    def test_l_shaped_decomposition(self) -> None:
        """L-shaped 2D arrangement is decomposed into disjoint pure boxes."""
        # Horizontal arm: (0,0,0) to (3,0,0)
        # Vertical arm: (0,0,0) to (0,2,0)
        coords = {
            Coord3D(0, 0, 0),
            Coord3D(1, 0, 0),
            Coord3D(2, 0, 0),
            Coord3D(3, 0, 0),
            Coord3D(0, 1, 0),
            Coord3D(0, 2, 0),
        }
        boxes = decompose_into_orthogonal_sub_boxes(coords)
        assert len(boxes) >= 2

        # Verify 100% density and exact disjoint partition
        union_cells: set[Coord3D] = set()
        for box in boxes:
            cells = _get_box_cells(box)
            # Must be pure: all box cells must belong to input coords
            assert cells.issubset(coords)
            # Must be disjoint: no overlap with previously covered cells
            assert union_cells.isdisjoint(cells)
            union_cells.update(cells)

        assert union_cells == coords
        assert sum(b.volume() for b in boxes) == len(coords)

    def test_u_shaped_decomposition(self) -> None:
        """U-shaped channel is partitioned into pure sub-boxes leaving the central concavity vacant."""
        # Base: (0,0,0), (1,0,0), (2,0,0)
        # Left wall: (0,1,0), (0,2,0)
        # Right wall: (2,1,0), (2,2,0)
        # Empty concavity: (1,1,0), (1,2,0)
        coords = {
            Coord3D(0, 0, 0),
            Coord3D(1, 0, 0),
            Coord3D(2, 0, 0),
            Coord3D(0, 1, 0),
            Coord3D(0, 2, 0),
            Coord3D(2, 1, 0),
            Coord3D(2, 2, 0),
        }
        boxes = decompose_into_orthogonal_sub_boxes(coords)

        concave_void = {Coord3D(1, 1, 0), Coord3D(1, 2, 0)}
        for box in boxes:
            cells = _get_box_cells(box)
            assert cells.isdisjoint(concave_void)

        union_cells = set().union(*(_get_box_cells(b) for b in boxes))
        assert union_cells == coords
        assert sum(b.volume() for b in boxes) == len(coords)

    def test_disconnected_points(self) -> None:
        """Isolated disconnected coordinates decompose into individual single-point boxes."""
        pts = [Coord3D(-10, -10, -10), Coord3D(0, 0, 0), Coord3D(10, 10, 10)]
        boxes = decompose_into_orthogonal_sub_boxes(pts)
        assert len(boxes) == 3
        for b in boxes:
            assert b.volume() == 1

        union_cells = set().union(*(_get_box_cells(b) for b in boxes))
        assert union_cells == set(pts)

    def test_3d_hollow_cube(self) -> None:
        """3x3x3 hollow cube missing center cell (1,1,1) decomposes without enclosing the center."""
        coords = {
            Coord3D(x, y, z)
            for x in range(3)
            for y in range(3)
            for z in range(3)
            if not (x == 1 and y == 1 and z == 1)
        }
        assert len(coords) == 26
        boxes = decompose_into_orthogonal_sub_boxes(coords)

        # Center point must not be in any sub-box
        center = Coord3D(1, 1, 1)
        for b in boxes:
            assert not b.contains_point(center)

        union_cells = set().union(*(_get_box_cells(b) for b in boxes))
        assert union_cells == coords
        assert sum(b.volume() for b in boxes) == 26

    def test_duplicate_coordinates_deduplicated(self) -> None:
        """Input containing duplicates is cleanly deduplicated without error."""
        coords = [
            Coord3D(0, 0, 0),
            Coord3D(0, 0, 0),
            Coord3D(1, 0, 0),
            Coord3D(1, 0, 0),
        ]
        boxes = decompose_into_orthogonal_sub_boxes(coords)
        assert len(boxes) == 1
        assert boxes[0].volume() == 2

    def test_negative_coordinates(self) -> None:
        """Handles negative coordinate quadrants properly."""
        coords = {
            Coord3D(-5, -5, 0),
            Coord3D(-4, -5, 0),
            Coord3D(-5, -4, 0),
        }
        boxes = decompose_into_orthogonal_sub_boxes(coords)
        assert sum(b.volume() for b in boxes) == 3
        union_cells = set().union(*(_get_box_cells(b) for b in boxes))
        assert union_cells == coords
