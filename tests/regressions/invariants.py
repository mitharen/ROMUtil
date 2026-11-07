"""Reusable layout invariant assertion helpers for regression and layout integrity testing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from romutil.layout.geometry import BoundingBox3D, Segment3D, segment_room_penetration
from romutil.models import Direction
from romutil.types import Coord3D, SolvedLayout, WorldGraph


def assert_zero_coordinate_collisions(layout: SolvedLayout) -> None:
    """Assert that every integer coordinate (x, y, z) is occupied by at most one room entity.

    Args:
        layout: Solved layout containing computed room coordinates.

    Raises:
        AssertionError: If two or more room entities occupy identical 3D integer coordinates.
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

            dx = cv.x - cu.x
            dy = cv.y - cu.y
            dz = cv.z - cu.z

            if sum(1 for d in (dx, dy, dz) if d != 0) != 1:
                continue

            seg = Segment3D(cu, cv)

            for w, cw in coords.items():
                if w == u or w == v:
                    continue
                if segment_room_penetration(seg, cw):
                    violations.append(
                        f"Corridor ({u} <-> {v}) penetrates interior room {w} at {cw}"
                    )

            if boxes:
                c_min_x, c_max_x = min(cu.x, cv.x), max(cu.x, cv.x)
                c_min_y, c_max_y = min(cu.y, cv.y), max(cu.y, cv.y)
                c_min_z, c_max_z = min(cu.z, cv.z), max(cu.z, cv.z)

                if dx != 0:
                    c_min_x += 1
                    c_max_x -= 1
                elif dy != 0:
                    c_min_y += 1
                    c_max_y -= 1
                elif dz != 0:
                    c_min_z += 1
                    c_max_z -= 1

                if c_min_x <= c_max_x and c_min_y <= c_max_y and c_min_z <= c_max_z:
                    corridor_box = BoundingBox3D(
                        c_min_x, c_max_x, c_min_y, c_max_y, c_min_z, c_max_z
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
