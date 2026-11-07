"""JSON map data renderer and serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from romutil.models import Direction
from romutil.renderers.base import BaseRenderer
from romutil.types import RenderOptions, SolvedLayout, WorldGraph


def _format_dice(val: Any) -> str:
    """Format dice tuple or representation into standard dice notation (e.g. 1d1+999)."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        if len(val) == 3:
            s0, s1, s2 = val
            s1_str = str(s1)
            if not s1_str.startswith(("d", "D")):
                s1_str = f"d{s1_str}"
            s2_str = str(s2)
            if not s2_str.startswith(("+", "-")):
                s2_str = f"+{s2_str}"
            return f"{s0}{s1_str}{s2_str}"
        return " ".join(str(v) for v in val)
    return str(val)


def _serialize_flags(val: Any) -> Any:
    """Ensure bitmask/act flags are JSON-serializable."""
    if val is None:
        return ""
    if isinstance(val, (int, str, bool, float)):
        return val
    if isinstance(val, (list, tuple, set)):
        return [str(v) for v in val]
    return str(val)


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

    # Parse resets to track mob spawns per room and equipped/carried items per mob
    room_mob_spawns: dict[int, list[dict[str, Any]]] = {}
    mob_equipment: dict[int, list[dict[str, Any]]] = {}
    last_mob_vnum: int | None = None

    for reset in getattr(graph, "resets", ()):
        cmd = str(getattr(reset, "command", "")).upper()
        args: tuple[Any, ...] = tuple(getattr(reset, "args", ()))
        if not cmd or cmd == "*":
            continue

        if cmd == "M":
            if len(args) >= 4:
                try:
                    mob_vnum = int(args[1])
                    global_limit = int(args[2])
                    room_vnum = int(args[3])
                    room_limit = int(args[4]) if len(args) > 4 else global_limit
                except (ValueError, TypeError):
                    last_mob_vnum = None
                    continue

                last_mob_vnum = mob_vnum
                if room_vnum not in room_mob_spawns:
                    room_mob_spawns[room_vnum] = []

                existing = next(
                    (item for item in room_mob_spawns[room_vnum] if item["vnum"] == mob_vnum),
                    None,
                )
                if existing is not None:
                    existing["count"] += 1
                else:
                    room_mob_spawns[room_vnum].append({
                        "vnum": mob_vnum,
                        "count": 1,
                        "global_limit": global_limit,
                        "room_limit": room_limit,
                    })
            else:
                last_mob_vnum = None
        elif cmd in ("E", "G"):
            if last_mob_vnum is not None and len(args) >= 2:
                try:
                    obj_vnum = int(args[1])
                except (ValueError, TypeError):
                    continue

                obj = (
                    graph.objects.get(obj_vnum)
                    if hasattr(graph, "objects") and graph.objects
                    else None
                )
                short_desc = (
                    getattr(obj, "short_desc", "") if obj else f"Object #{obj_vnum}"
                )

                if cmd == "E":
                    wear_loc_raw = args[3] if len(args) > 3 else -1
                    try:
                        wear_loc: Any = int(wear_loc_raw)
                    except (ValueError, TypeError):
                        wear_loc = wear_loc_raw
                else:
                    wear_loc = "inventory"

                if last_mob_vnum not in mob_equipment:
                    mob_equipment[last_mob_vnum] = []

                item_entry = {
                    "vnum": obj_vnum,
                    "short_desc": str(short_desc),
                    "wear_loc": wear_loc,
                }
                if item_entry not in mob_equipment[last_mob_vnum]:
                    mob_equipment[last_mob_vnum].append(item_entry)
        else:
            last_mob_vnum = None

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
            "mobs": room_mob_spawns.get(vnum, []),
        }
        r_area = getattr(r, "area_name", None) or (area_name if area_name else None)
        if r_area is not None:
            room_dict["area_name"] = str(r_area)
        r_file = getattr(r, "area_file", None) or (area_file if area_file else None)
        if r_file is not None:
            room_dict["area_file"] = str(r_file)

        rooms_data.append(room_dict)

    # Build top-level mobiles dictionary
    mobiles_data: dict[str, Any] = {}
    mobiles_map = getattr(graph, "mobiles", {}) or {}
    for m_vnum, mob in mobiles_map.items():
        vnum_int = int(m_vnum)
        shop_data = None
        if hasattr(graph, "shops") and graph.shops and vnum_int in graph.shops:
            s = graph.shops[vnum_int]
            shop_data = {
                "keeper": int(getattr(s, "keeper", vnum_int)),
                "buy_types": [int(bt) for bt in getattr(s, "buy_types", ())],
                "profit_buy": int(getattr(s, "profit_buy", 100)),
                "profit_sell": int(getattr(s, "profit_sell", 100)),
                "open_hour": int(getattr(s, "open_hour", 0)),
                "close_hour": int(getattr(s, "close_hour", 24)),
            }

        special_data = None
        if hasattr(graph, "specials") and graph.specials and vnum_int in graph.specials:
            sp = graph.specials[vnum_int]
            special_data = getattr(sp, "spec_fun", None)

        mobiles_data[str(vnum_int)] = {
            "vnum": vnum_int,
            "name": str(getattr(mob, "player_name", "")),
            "short_desc": str(getattr(mob, "short_desc", "")),
            "long_desc": str(getattr(mob, "long_desc", "")),
            "level": int(getattr(mob, "level", 0)),
            "race": str(getattr(mob, "race", "")),
            "sex": str(getattr(mob, "sex", "")),
            "alignment": int(getattr(mob, "alignment", 0)),
            "act_flags": _serialize_flags(getattr(mob, "act_flags", None)),
            "affected_by": _serialize_flags(getattr(mob, "affected_by", None)),
            "hit": _format_dice(getattr(mob, "hit", None)),
            "damage": _format_dice(getattr(mob, "damage", None)),
            "dam_type": str(getattr(mob, "dam_type", "")),
            "ac_pierce": int(getattr(mob, "ac_pierce", 0)),
            "ac_bash": int(getattr(mob, "ac_bash", 0)),
            "ac_slash": int(getattr(mob, "ac_slash", 0)),
            "ac_exotic": int(getattr(mob, "ac_exotic", 0)),
            "wealth": int(getattr(mob, "wealth", 0)),
            "shop": shop_data,
            "special": special_data,
            "equipment": mob_equipment.get(vnum_int, []),
        }

    # Include any spawned mob definitions that were not explicitly in graph.mobiles
    for room_mobs in room_mob_spawns.values():
        for item in room_mobs:
            m_vnum_str = str(item["vnum"])
            if m_vnum_str not in mobiles_data:
                mobiles_data[m_vnum_str] = {
                    "vnum": item["vnum"],
                    "name": f"Mobile #{item['vnum']}",
                    "short_desc": f"Mobile #{item['vnum']}",
                    "long_desc": "",
                    "level": 0,
                    "race": "",
                    "sex": "",
                    "alignment": 0,
                    "act_flags": "",
                    "affected_by": "",
                    "hit": "",
                    "damage": "",
                    "dam_type": "",
                    "ac_pierce": 0,
                    "ac_bash": 0,
                    "ac_slash": 0,
                    "ac_exotic": 0,
                    "wealth": 0,
                    "shop": None,
                    "special": None,
                    "equipment": mob_equipment.get(item["vnum"], []),
                }

    return {
        "area": {
            "name": area_name,
            "file": area_file,
        },
        "bounds": computed_bounds,
        "rooms": rooms_data,
        "mobiles": mobiles_data,
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
