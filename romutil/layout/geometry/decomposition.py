"""Computational geometry algorithms for discrete orthogonal decomposition."""

from __future__ import annotations

from typing import Iterable, Mapping

from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.types import Coord3D


def _build_z_index(
    coords: Iterable[Coord3D],
) -> tuple[dict[int, dict[int, set[int]]], int]:
    """Index 3D coordinates into a nested ``z -> y -> set[x]`` map and total unique cell count."""
    by_z: dict[int, dict[int, set[int]]] = {}
    count = 0
    for pt in coords:
        y_map = by_z.get(pt.z)
        if y_map is None:
            by_z[pt.z] = {pt.y: {pt.x}}
            count += 1
        else:
            xs = y_map.get(pt.y)
            if xs is None:
                y_map[pt.y] = {pt.x}
                count += 1
            elif pt.x not in xs:
                xs.add(pt.x)
                count += 1
    return by_z, count


def _full_bounds_if_solid(
    by_z: Mapping[int, Mapping[int, set[int]]],
    remaining_count: int,
) -> BoundingBox3D | None:
    """Return the enclosing BoundingBox3D if remaining cells form a 100% dense box."""
    min_z = min(by_z)
    max_z = max(by_z)
    min_y = min(min(y_map) for y_map in by_z.values())
    max_y = max(max(y_map) for y_map in by_z.values())
    min_x = min(min(xs) for y_map in by_z.values() for xs in y_map.values())
    max_x = max(max(xs) for y_map in by_z.values() for xs in y_map.values())
    if (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1) == remaining_count:
        return BoundingBox3D(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
        )
    return None


def _intersect_z_slice(
    inter: Mapping[int, set[int]],
    z_map: Mapping[int, set[int]],
) -> tuple[dict[int, set[int]], int]:
    """Intersect a ``y -> set[x]`` slice with another z layer, returning ``(intersection, cell_count)``."""
    next_inter: dict[int, set[int]] = {}
    inter_count = 0
    for y, xs in inter.items():
        other_xs = z_map.get(y)
        if other_xs is not None:
            common = xs & other_xs
            if common:
                next_inter[y] = common
                inter_count += len(common)
    return next_inter, inter_count


def _max_rect_2d(
    inter: Mapping[int, set[int]],
    z_span: int,
) -> tuple[int, int, int, int, int]:
    """Find the maximal-volume 2D solid rectangle in a ``y -> set[x]`` slice map.

    Sweeps rows in ascending ``y`` order while maintaining accumulated contiguous
    vertical histogram heights and a monotonic stack across contiguous sorted ``x``
    runs. Ties on volume ``sh * width * z_span`` are broken lexicographically by
    ``(x1, y1, x2, y2)``.
    """
    best_key: tuple[int, int, int, int, int] | None = None
    best_vol = 0
    best_rect = (0, 0, 0, 0)

    prev_y: int | None = None
    heights: dict[int, int] = {}

    for y in sorted(inter):
        row_xs = inter[y]
        if prev_y is not None and y == prev_y + 1:
            heights = {x: heights.get(x, 0) + 1 for x in row_xs}
        else:
            heights = {x: 1 for x in row_xs}
        prev_y = y

        stack: list[tuple[int, int]] = []
        prev_x: int | None = None

        def _pop_taller(min_h: int, right_x: int) -> int:
            nonlocal best_key, best_vol, best_rect
            start = right_x + 1
            while stack and stack[-1][1] >= min_h:
                sx, sh = stack.pop()
                start = sx
                if sh > min_h:
                    vol = sh * (right_x - sx + 1) * z_span
                    y1 = y - sh + 1
                    key = (-vol, sx, y1, right_x, y)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_vol = vol
                        best_rect = (sx, y1, right_x, y)
            return start

        for x in sorted(row_xs):
            if prev_x is not None and x != prev_x + 1:
                _pop_taller(0, prev_x)
            h = heights[x]
            start_x = _pop_taller(h, x - 1) if (stack and stack[-1][1] >= h) else x
            stack.append((start_x, h))
            prev_x = x

        if prev_x is not None:
            _pop_taller(0, prev_x)

    return best_vol, best_rect[0], best_rect[1], best_rect[2], best_rect[3]


def decompose_into_orthogonal_sub_boxes(
    coords: Iterable[Coord3D],
) -> tuple[BoundingBox3D, ...]:
    """Decompose a discrete 3D coordinate set into minimal disjoint orthogonal bounding boxes.

    Partitions an arbitrary coordinate set into a collection of disjoint axis-aligned
    bounding boxes with 100% density with respect to the input set (i.e. containing
    zero empty cells). Uses an exact greedy maximal-volume selection strategy covering
    unassigned cells until the entire set is partitioned.

    Args:
        coords: Iterable of 3D integer coordinates.

    Returns:
        A tuple of disjoint BoundingBox3D instances whose union exactly covers coords.
    """
    by_z, remaining_count = _build_z_index(coords)
    if remaining_count == 0:
        return ()

    boxes: list[BoundingBox3D] = []

    while remaining_count > 0:
        solid_box = _full_bounds_if_solid(by_z, remaining_count)
        if solid_box is not None:
            boxes.append(solid_box)
            break

        active_zs = sorted(by_z)
        contig_end: dict[int, int] = {}
        for idx in range(len(active_zs) - 1, -1, -1):
            z = active_zs[idx]
            if idx + 1 < len(active_zs) and active_zs[idx + 1] == z + 1:
                contig_end[z] = contig_end[z + 1]
            else:
                contig_end[z] = z

        best_key: tuple[int, int, int, int, int, int, int] | None = None
        best_vol = 0
        best_coords = (0, 0, 0, 0, 0, 0)

        for i, z1 in enumerate(active_zs):
            z_end = contig_end[z1]
            inter: dict[int, set[int]] = dict(by_z[z1])
            inter_count = sum(len(xs) for xs in inter.values())

            for j in range(i, len(active_zs)):
                z2 = active_zs[j]
                if z2 > z_end:
                    break
                if j > i:
                    inter, inter_count = _intersect_z_slice(inter, by_z[z2])
                if inter_count == 0 or inter_count * (z_end - z1 + 1) < best_vol:
                    break
                z_span = z2 - z1 + 1
                if inter_count * z_span < best_vol:
                    continue
                vol, x1, y1, x2, y2 = _max_rect_2d(inter, z_span)
                key = (-vol, x1, y1, z1, x2, y2, z2)
                if best_key is None or key < best_key:
                    best_key = key
                    best_vol = vol
                    best_coords = (x1, y1, z1, x2, y2, z2)

        x1, y1, z1, x2, y2, z2 = best_coords
        boxes.append(
            BoundingBox3D(
                min_x=x1,
                max_x=x2,
                min_y=y1,
                max_y=y2,
                min_z=z1,
                max_z=z2,
            )
        )
        remaining_count -= best_vol
        x_remove = set(range(x1, x2 + 1))
        for z in range(z1, z2 + 1):
            y_map = by_z[z]
            for y in range(y1, y2 + 1):
                xs = y_map[y]
                xs.difference_update(x_remove)
                if not xs:
                    del y_map[y]
            if not y_map:
                del by_z[z]

    return tuple(boxes)
