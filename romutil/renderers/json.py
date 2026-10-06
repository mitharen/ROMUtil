"""JSON map data renderer and serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from romutil.models import Direction
from romutil.renderers.base import BaseRenderer
from romutil.types import RenderOptions, SolvedLayout, WorldGraph


def build_area_json(
    graph: WorldGraph,
    layout: SolvedLayout,
    options: RenderOptions | None = None,
    *,
    bounds: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Construct a JSON-serializable dictionary representing the solved area map.

    Consumes WorldGraph and SolvedLayout strictly.
    """
    if not isinstance(graph, WorldGraph):
        raise TypeError(f"Expected WorldGraph as first argument, got {type(graph).__name__}")
    if not isinstance(layout, SolvedLayout):
        raise TypeError(f"Expected SolvedLayout as second argument, got {type(layout).__name__}")

    resolved_bounds = bounds
    if resolved_bounds is None and isinstance(options, dict):
        resolved_bounds = options.get("bounds")

    # Determine area name and filename
    area_name = ""
    area_file = ""
    if graph.header:
        area_name = graph.header.name
        area_file = graph.header.filename

    # Calculate bounding box from layout coordinates
    if resolved_bounds is not None:
        computed_bounds = {
            "min_x": int(resolved_bounds.get("min_x", 0)),
            "max_x": int(resolved_bounds.get("max_x", 0)),
            "min_y": int(resolved_bounds.get("min_y", 0)),
            "max_y": int(resolved_bounds.get("max_y", 0)),
            "min_z": int(resolved_bounds.get("min_z", 0)),
            "max_z": int(resolved_bounds.get("max_z", 0)),
        }
    elif layout.coordinates:
        xs = [int(round(c.x)) for c in layout.coordinates.values()]
        ys = [int(round(c.y)) for c in layout.coordinates.values()]
        zs = [int(round(c.z)) for c in layout.coordinates.values()]
        computed_bounds = {
            "min_x": min(xs) if xs else 0,
            "max_x": max(xs) if xs else 0,
            "min_y": min(ys) if ys else 0,
            "max_y": max(ys) if ys else 0,
            "min_z": min(zs) if zs else 0,
            "max_z": max(zs) if zs else 0,
        }
    else:
        computed_bounds = {
            "min_x": 0,
            "max_x": 0,
            "min_y": 0,
            "max_y": 0,
            "min_z": 0,
            "max_z": 0,
        }

    # Sort rooms deterministically by VNUM
    sorted_rooms = sorted(graph.rooms.values(), key=lambda r: int(r.vnum))

    rooms_data = []
    for r in sorted_rooms:
        vnum = int(r.vnum)
        coord = layout.coordinates.get(vnum)
        x_val = int(round(coord.x)) if coord is not None else 0
        y_val = int(round(coord.y)) if coord is not None else 0
        z_val = int(round(coord.z)) if coord is not None else 0

        # Sort exits by direction for deterministic export
        def _get_dir_val(e: Any) -> int:
            d = getattr(e, "direction", 0)
            return d.value if hasattr(d, "value") else int(d)

        sorted_exits = sorted(r.exits, key=_get_dir_val)
        exits_data = []
        for e in sorted_exits:
            dir_attr = getattr(e, "direction", 0)
            dir_int = dir_attr.value if hasattr(dir_attr, "value") else int(dir_attr)
            try:
                dir_name = Direction(dir_int).name
            except (ValueError, KeyError):
                dir_name = str(dir_attr)

            target = getattr(e, "target_vnum", None)
            dst_val = (
                target
                if target is not None and target != -1
                else getattr(e, "dst_vnum", getattr(e, "dst", 0))
            )

            # Step distance calculation
            if getattr(e, "distance", None) is not None:
                dist_val = int(getattr(e, "distance"))
            elif vnum in layout.coordinates and dst_val in layout.coordinates:
                c1 = layout.coordinates[vnum]
                c2 = layout.coordinates[dst_val]
                dist_val = max(abs(c2.x - c1.x), abs(c2.y - c1.y), abs(c2.z - c1.z))
                if dist_val == 0:
                    dist_val = 1
            else:
                dist_val = 1

            # One-way status detection
            if getattr(e, "one_way", None) is not None and getattr(e, "one_way") is not None:
                one_way = bool(getattr(e, "one_way"))
            elif dst_val in graph.rooms:
                dst_room = graph.rooms[dst_val]
                try:
                    inv_dir = Direction(dir_int).invert().value
                    has_return = any(
                        (getattr(re.direction, "value", re.direction) == inv_dir)
                        and (getattr(re, "target_vnum", getattr(re, "dst_vnum", getattr(re, "dst", None))) == vnum)
                        for re in dst_room.exits
                    )
                    one_way = not has_return
                except Exception:
                    one_way = False
            else:
                one_way = True

            exit_dict = {
                "direction": dir_name,
                "dst": int(dst_val),
                "distance": int(dist_val),
                "one_way": bool(one_way),
            }
            if target is not None:
                exit_dict["target_vnum"] = int(target)
            exits_data.append(exit_dict)

        room_dict: dict[str, Any] = {
            "vnum": vnum,
            "name": str(r.name),
            "description": str(r.description),
            "desc": str(r.description),
            "coords": {
                "x": x_val,
                "y": y_val,
                "z": z_val,
            },
            "exits": exits_data,
        }
        r_area = getattr(r, "area_name", None) or (area_name if area_name else None)
        if r_area is not None:
            room_dict["area_name"] = str(r_area)
        r_file = getattr(r, "area_file", None) or (area_file if area_file else None)
        if r_file is not None:
            room_dict["area_file"] = str(r_file)

        rooms_data.append(room_dict)

    return {
        "area": {
            "name": area_name,
            "file": area_file,
        },
        "bounds": computed_bounds,
        "rooms": rooms_data,
    }


def export_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> Path:
    """Serialize map dictionary to a JSON file on disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path


class JSONRenderer:
    """Renderer exporting map data to structured JSON format conforming to BaseRenderer."""

    def render(
        self,
        graph: WorldGraph,
        layout: SolvedLayout,
        output_path: Path | str,
        options: RenderOptions | None = None,
        *,
        indent: int = 2,
        bounds: Optional[Dict[str, int]] = None,
    ) -> Path:
        """Render a solved layout of a WorldGraph to output_path as JSON."""
        if not isinstance(graph, WorldGraph):
            raise TypeError(f"Expected WorldGraph as first argument, got {type(graph).__name__}")
        if not isinstance(layout, SolvedLayout):
            raise TypeError(f"Expected SolvedLayout as second argument, got {type(layout).__name__}")
        if output_path is None:
            raise TypeError("output_path must be specified")
        actual_output_path = Path(output_path)
        data = build_area_json(graph, layout, options, bounds=bounds)
        return export_json(data, actual_output_path, indent=indent)


__all__ = [
    "JSONRenderer",
    "build_area_json",
    "export_json",
]
