"""JSON map data renderer and serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, overload

from romutil.models import AreaHeader, Direction, Room
from romutil.renderers.base import BaseRenderer, adapt_legacy_rdb
from romutil.types import Coord3D, LayoutDiagnostics, RenderOptions, SolvedLayout, SolverStatus, WorldGraph


def build_area_json(
    graph_or_rooms: WorldGraph | Dict[int, Any] | List[Any],
    layout_or_meta: Optional[Any] = None,
    options_or_bounds: Optional[Any] = None,
    *,
    area_meta: Optional[Any] = None,
    bounds: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Construct a JSON-serializable dictionary representing the solved area map.

    Consumes WorldGraph and SolvedLayout with zero access to legacy Room.x/y/z/dummy.
    Provides backward-compatible fallback for legacy callers passing (rdb, area_meta, bounds).
    """
    if isinstance(graph_or_rooms, WorldGraph):
        graph = graph_or_rooms
        if isinstance(layout_or_meta, SolvedLayout):
            layout = layout_or_meta
        else:
            layout = SolvedLayout(
                coordinates={},
                diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
            )
        resolved_bounds = bounds
        if resolved_bounds is None and isinstance(options_or_bounds, dict):
            resolved_bounds = options_or_bounds.get("bounds")
    else:
        # Legacy caller passing rooms: dict[int, Room] or list[Room]
        meta = area_meta if area_meta is not None else layout_or_meta
        resolved_bounds = bounds if bounds is not None else (
            options_or_bounds if isinstance(options_or_bounds, dict) else None
        )
        graph, layout = adapt_legacy_rdb(graph_or_rooms, meta)

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
    sorted_rooms = sorted(graph.rooms.values(), key=lambda r: int(getattr(r, "vnum", 0)))

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

    @overload
    def render(
        self,
        graph: WorldGraph,
        layout: SolvedLayout,
        output_path: Path | str,
        options: RenderOptions | None = None,
        *,
        header: AreaHeader | None = None,
        **extra_options: Any,
    ) -> Path:
        ...

    @overload
    def render(
        self,
        graph: dict[int, Any] | list[Any],
        layout: str | Path,
        output_path: AreaHeader | Any | None = None,
        options: Any = None,
        *,
        header: AreaHeader | None = None,
        **extra_options: Any,
    ) -> Path:
        ...

    def render(
        self,
        graph: Any,
        layout: Any,
        output_path: Any = None,
        options: Any = None,
        *,
        header: AreaHeader | None = None,
        **extra_options: Any,
    ) -> Path:
        """Render a solved layout of a WorldGraph to output_path as JSON.

        Primary signature: render(graph, layout, output_path, options=None)
        Legacy signature: render(rdb, output_path, header=None, **options)
        """
        if isinstance(graph, WorldGraph):
            actual_graph = graph
            if not isinstance(layout, SolvedLayout):
                raise TypeError(f"Expected SolvedLayout as second argument, got {type(layout).__name__}")
            actual_layout = layout
            if output_path is None:
                raise TypeError("output_path must be specified")
            actual_output_path = Path(output_path)
            actual_options = options
            indent = extra_options.get("indent", 2)
            bounds = extra_options.get("bounds")
        else:
            # Legacy signature: render(rdb, output_path, header=None, **options)
            legacy_rdb = graph
            actual_output_path = Path(layout if isinstance(layout, (str, Path)) else str(layout))
            legacy_header = header if header is not None else (
                output_path if isinstance(output_path, (AreaHeader, tuple, list)) or hasattr(output_path, "filename") or hasattr(output_path, "name") else None
            )
            actual_graph, actual_layout = adapt_legacy_rdb(legacy_rdb, legacy_header)
            actual_options = options
            indent = extra_options.get("indent", 2)
            bounds = extra_options.get("bounds")

        data = build_area_json(actual_graph, actual_layout, actual_options, bounds=bounds)
        return export_json(data, actual_output_path, indent=indent)


__all__ = [
    "JSONRenderer",
    "build_area_json",
    "export_json",
]
