"""Computational geometry algorithms for discrete orthogonal decomposition."""

from __future__ import annotations

from typing import Iterable

from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.types import Coord3D


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
    coords_set = set(coords)
    if not coords_set:
        return ()

    full_box = BoundingBox3D.from_coordinates(coords_set)
    if full_box.volume() == len(coords_set):
        return (full_box,)

    remaining = set(coords_set)
    boxes: list[BoundingBox3D] = []

    while remaining:
        n = len(remaining)
        full_b = BoundingBox3D.from_coordinates(remaining)
        if full_b.volume() == n:
            boxes.append(full_b)
            break

        pts = sorted(remaining, key=lambda p: (p.x, p.y, p.z))
        candidates: list[tuple[int, Coord3D, Coord3D]] = []
        for i, p1 in enumerate(pts):
            for p2 in pts[i:]:
                if p1.y <= p2.y and p1.z <= p2.z:
                    vol = (p2.x - p1.x + 1) * (p2.y - p1.y + 1) * (p2.z - p1.z + 1)
                    if vol <= n:
                        candidates.append((vol, p1, p2))

        # Sort candidates descending by volume, with deterministic coordinate tie-breaking
        candidates.sort(
            key=lambda c: (-c[0], c[1].x, c[1].y, c[1].z, c[2].x, c[2].y, c[2].z)
        )

        chosen: BoundingBox3D | None = None
        for _vol, p1, p2 in candidates:
            is_pure = True
            for x in range(p1.x, p2.x + 1):
                for y in range(p1.y, p2.y + 1):
                    for z in range(p1.z, p2.z + 1):
                        if Coord3D(x, y, z) not in remaining:
                            is_pure = False
                            break
                    if not is_pure:
                        break
                if not is_pure:
                    break
            if is_pure:
                chosen = BoundingBox3D(
                    min_x=p1.x,
                    max_x=p2.x,
                    min_y=p1.y,
                    max_y=p2.y,
                    min_z=p1.z,
                    max_z=p2.z,
                )
                break

        if chosen is None:
            first_pt = pts[0]
            chosen = BoundingBox3D(
                min_x=first_pt.x,
                max_x=first_pt.x,
                min_y=first_pt.y,
                max_y=first_pt.y,
                min_z=first_pt.z,
                max_z=first_pt.z,
            )

        boxes.append(chosen)
        for x in range(chosen.min_x, chosen.max_x + 1):
            for y in range(chosen.min_y, chosen.max_y + 1):
                for z in range(chosen.min_z, chosen.max_z + 1):
                    remaining.discard(Coord3D(x, y, z))

    return tuple(boxes)
