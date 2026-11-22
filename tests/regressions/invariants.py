"""Reusable layout invariant assertion helpers for regression and layout integrity testing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from romutil.layout.geometry import BoundingBox3D
from romutil.models import Direction, scale_to_extent
from romutil.types import Coord3D, SolvedLayout, WorldGraph


def assert_zero_coordinate_collisions(layout: SolvedLayout) -> None:
    """Assert that every integer cell (x, y, z) is occupied by at most one room entity.

    Args:
        layout: Solved layout containing computed room coordinates.

    Raises:
        AssertionError: If two or more room entities occupy identical or overlapping
            3D integer footprints.
    """
    by_coord: dict[Coord3D, list[int]] = {}
    for vnum, coord in layout.coordinates.items():
        by_coord.setdefault(coord, []).append(vnum)

    duplicates = {coord: vnums for coord, vnums in by_coord.items() if len(vnums) > 1}
    if duplicates:
        details = "; ".join(
            f"Coord3D(x={c.x}, y={c.y}, z={c.z}): rooms {sorted(vs)}"
            for c, vs in sorted(
                duplicates.items(), key=lambda item: (item[0].z, item[0].y, item[0].x)
            )
        )
        raise AssertionError(
            f"Detected {len(duplicates)} coordinate collision(s) across "
            f"{sum(len(vs) for vs in duplicates.values())} rooms: {details}"
        )

    room_scales = getattr(layout, "room_scales", {})
    coord_unit = max(1, int(getattr(layout, "coord_unit", 1)))
    if any(s > 1 for s in room_scales.values()) or coord_unit > 1:
        items = list(layout.coordinates.items())
        extents = {
            v: scale_to_extent(int(room_scales.get(v, 1)), coord_unit)
            for v, _ in items
        }
        overlaps: list[str] = []
        for i in range(len(items)):
            u, cu = items[i]
            u_lo, u_hi = extents[u]
            for j in range(i + 1, len(items)):
                v, cv = items[j]
                if cu.z != cv.z:
                    continue
                v_lo, v_hi = extents[v]
                if (
                    max(cu.x - u_lo, cv.x - v_lo) <= min(cu.x + u_hi, cv.x + v_hi)
                    and max(cu.y - u_lo, cv.y - v_lo) <= min(cu.y + u_hi, cv.y + v_hi)
                ):
                    overlaps.append(
                        f"rooms {u} ({cu}, scale={room_scales.get(u, 1)}) and "
                        f"{v} ({cv}, scale={room_scales.get(v, 1)})"
                    )
        if overlaps:
            raise AssertionError(
                f"Detected {len(overlaps)} room footprint overlap(s): {'; '.join(overlaps[:10])}"
            )


def assert_axis_aligned_exits(
    layout: SolvedLayout,
    graph: WorldGraph,
    exits: Sequence[tuple[int, int]],
) -> None:
    """Assert that designated cardinal street exits have zero perpendicular shear.

    For an exit between room u and room v along a cardinal axis:
    - East/West (X-axis): delta_y == 0 and delta_z == 0.
    - North/South (Y-axis): delta_x == 0 and delta_z == 0.
    - Up/Down (Z-axis): delta_x == 0 and delta_y == 0.

    Args:
        layout: Solved layout containing computed room coordinates.
        graph: World graph containing topological exit definitions and directions.
        exits: Sequence of (source_vnum, target_vnum) exit pairs to verify.

    Raises:
        AssertionError: If any designated exit exhibits non-zero perpendicular shear.
        KeyError: If an exit room vnum is missing from the solved layout coordinates.
    """
    violations: list[str] = []

    for u, v in exits:
        if u not in layout.coordinates:
            raise KeyError(f"Room {u} from exit ({u}, {v}) missing from layout coordinates.")
        if v not in layout.coordinates:
            raise KeyError(f"Room {v} from exit ({u}, {v}) missing from layout coordinates.")

        cu = layout.coordinates[u]
        cv = layout.coordinates[v]

        direction: Direction | None = None
        room_u = graph.rooms.get(u)
        if room_u is not None:
            for ex in room_u.exits:
                if ex.dst_vnum == v:
                    direction = ex.direction
                    break

        if direction is None:
            room_v = graph.rooms.get(v)
            if room_v is not None:
                for ex in room_v.exits:
                    if ex.dst_vnum == u:
                        direction = ex.direction.invert()
                        break

        dx = cv.x - cu.x
        dy = cv.y - cu.y
        dz = cv.z - cu.z

        if direction is not None:
            if direction in (Direction.east, Direction.west):
                if dy != 0 or dz != 0:
                    violations.append(
                        f"Exit ({u} -> {v}) along {direction.name}: expected dy=0, dz=0 but got "
                        f"dy={dy}, dz={dz} (coords: {cu} -> {cv})"
                    )
            elif direction in (Direction.north, Direction.south):
                if dx != 0 or dz != 0:
                    violations.append(
                        f"Exit ({u} -> {v}) along {direction.name}: expected dx=0, dz=0 but got "
                        f"dx={dx}, dz={dz} (coords: {cu} -> {cv})"
                    )
            elif direction in (Direction.up, Direction.down):
                if dx != 0 or dy != 0:
                    violations.append(
                        f"Exit ({u} -> {v}) along {direction.name}: expected dx=0, dy=0 but got "
                        f"dx={dx}, dy={dy} (coords: {cu} -> {cv})"
                    )
        else:
            non_zero_axes = sum(1 for d in (dx, dy, dz) if d != 0)
            if non_zero_axes > 1:
                violations.append(
                    f"Exit ({u} -> {v}): expected single cardinal axis but got "
                    f"dx={dx}, dy={dy}, dz={dz} (coords: {cu} -> {cv})"
                )

    if violations:
        details = "; ".join(violations)
        raise AssertionError(
            f"Found {len(violations)} axis alignment violation(s): {details}"
        )


def assert_corridor_non_penetration(
    layout: SolvedLayout,
    graph: WorldGraph,
    keepout_boxes: Sequence[BoundingBox3D] | Mapping[Any, Sequence[BoundingBox3D]] | None = None,
) -> None:
    """Assert that no active corridor segment penetrates any room entity or keep-out box.

    Verifies that straight active corridor segments between connected rooms do not
    intersect the interior unit bounding box of any unrelated room or any designated
    child keep-out bounding box.

    Args:
        layout: Solved layout containing computed room coordinates.
        graph: World graph containing topological exit definitions.
        keepout_boxes: Optional sequence or mapping of interior child keep-out bounding boxes.

    Raises:
        AssertionError: If any active corridor segment penetrates a room or keep-out box.
    """
    coords = layout.coordinates
    room_scales = getattr(layout, "room_scales", {})
    coord_unit = max(1, int(getattr(layout, "coord_unit", 1)))
    extents = {
        v: scale_to_extent(int(room_scales.get(v, 1)), coord_unit)
        for v in coords
    }
    seen_edges: set[tuple[int, int]] = set()
    violations: list[str] = []

    boxes: list[BoundingBox3D] = []
    if keepout_boxes is not None:
        if isinstance(keepout_boxes, Mapping):
            for b_list in keepout_boxes.values():
                boxes.extend(b_list)
        else:
            boxes.extend(keepout_boxes)

    for room in graph.rooms.values():
        u = room.vnum
        if u not in coords:
            continue
        cu = coords[u]
        u_lo, u_hi = extents[u]

        for exit_def in room.exits:
            v = exit_def.dst_vnum
            if v not in coords:
                continue

            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            cv = coords[v]
            if cu == cv:
                continue
            v_lo, v_hi = extents[v]

            dx = cv.x - cu.x
            dy = cv.y - cu.y
            dz = cv.z - cu.z

            if sum(1 for d in (dx, dy, dz) if d != 0) != 1:
                continue

            if dx > 0:
                c_min_x, c_max_x = cu.x + u_hi + 1, cv.x - v_lo - 1
                c_min_y = c_max_y = cu.y
                c_min_z = c_max_z = cu.z
            elif dx < 0:
                c_min_x, c_max_x = cv.x + v_hi + 1, cu.x - u_lo - 1
                c_min_y = c_max_y = cu.y
                c_min_z = c_max_z = cu.z
            elif dy > 0:
                c_min_x = c_max_x = cu.x
                c_min_y, c_max_y = cu.y + u_hi + 1, cv.y - v_lo - 1
                c_min_z = c_max_z = cu.z
            elif dy < 0:
                c_min_x = c_max_x = cu.x
                c_min_y, c_max_y = cv.y + v_hi + 1, cu.y - u_lo - 1
                c_min_z = c_max_z = cu.z
            else:
                c_min_x = c_max_x = cu.x
                c_min_y = c_max_y = cu.y
                c_min_z, c_max_z = min(cu.z, cv.z) + 1, max(cu.z, cv.z) - 1

            if c_min_x > c_max_x or c_min_y > c_max_y or c_min_z > c_max_z:
                continue

            corridor_box = BoundingBox3D(
                c_min_x, c_max_x, c_min_y, c_max_y, c_min_z, c_max_z
            )

            for w, cw in coords.items():
                if w == u or w == v:
                    continue
                w_lo, w_hi = extents[w]
                if (
                    c_min_z <= cw.z <= c_max_z
                    and max(c_min_x, cw.x - w_lo) <= min(c_max_x, cw.x + w_hi)
                    and max(c_min_y, cw.y - w_lo) <= min(c_max_y, cw.y + w_hi)
                ):
                    violations.append(
                        f"Corridor ({u} <-> {v}) interior {corridor_box} penetrates "
                        f"interior room {w} at {cw}"
                    )

            for kbox in boxes:
                if corridor_box.intersects(kbox):
                    violations.append(
                        f"Corridor ({u} <-> {v}) interior {corridor_box} "
                        f"intersects keep-out box {kbox}"
                    )

    if violations:
        details = "; ".join(violations)
        raise AssertionError(
            f"Found {len(violations)} corridor penetration violation(s): {details}"
        )
