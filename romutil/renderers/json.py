"""JSON map data renderer and serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from romutil.models import AreaHeader, Room
from romutil.renderers.base import BaseRenderer


def build_area_json(
    rooms: Union[Dict[int, Room], List[Room]],
    area_meta: Optional[Any] = None,
    bounds: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Constructs a JSON-serializable dictionary representing the solved area map
    matching the Task 3 specification schema.
    """
    if isinstance(rooms, dict):
        room_list = [r for r in rooms.values() if not getattr(r, 'dummy', False)]
    else:
        room_list = [r for r in rooms if not getattr(r, 'dummy', False)]

    # Determine area name and filename
    area_name = ''
    area_file = ''
    if area_meta:
        if isinstance(area_meta, (tuple, list)):
            area_file = str(area_meta[0]) if len(area_meta) > 0 else ''
            area_name = str(area_meta[1]) if len(area_meta) > 1 else ''
        else:
            area_file = str(getattr(area_meta, 'filename', getattr(area_meta, 'file', '')))
            area_name = str(getattr(area_meta, 'name', ''))

    # Calculate bounding box
    if bounds is not None:
        computed_bounds = {
            'min_x': int(bounds.get('min_x', 0)),
            'max_x': int(bounds.get('max_x', 0)),
            'min_y': int(bounds.get('min_y', 0)),
            'max_y': int(bounds.get('max_y', 0)),
            'min_z': int(bounds.get('min_z', 0)),
            'max_z': int(bounds.get('max_z', 0)),
        }
    elif room_list:
        xs = [int(round(r.x)) for r in room_list if r.x is not None]
        ys = [int(round(r.y)) for r in room_list if r.y is not None]
        zs = [int(round(r.z)) for r in room_list if r.z is not None]
        computed_bounds = {
            'min_x': min(xs) if xs else 0,
            'max_x': max(xs) if xs else 0,
            'min_y': min(ys) if ys else 0,
            'max_y': max(ys) if ys else 0,
            'min_z': min(zs) if zs else 0,
            'max_z': max(zs) if zs else 0,
        }
    else:
        computed_bounds = {
            'min_x': 0, 'max_x': 0,
            'min_y': 0, 'max_y': 0,
            'min_z': 0, 'max_z': 0,
        }

    # Sort rooms deterministically by VNUM
    sorted_rooms = sorted(room_list, key=lambda r: getattr(r, 'vnum', 0))

    rooms_data = []
    for r in sorted_rooms:
        x_val = int(round(r.x)) if r.x is not None else 0
        y_val = int(round(r.y)) if r.y is not None else 0
        z_val = int(round(r.z)) if r.z is not None else 0

        # Sort exits by direction for deterministic export
        sorted_exits = sorted(r.exits, key=lambda e: getattr(e.direction, 'value', 0))
        exits_data = []
        for e in sorted_exits:
            dir_name = e.direction.name if hasattr(e.direction, 'name') else str(e.direction)
            target = getattr(e, 'target_vnum', None)
            dst_val = target if target is not None and target != -1 else e.dst
            exit_dict = {
                'direction': dir_name,
                'dst': int(dst_val),
                'distance': int(e.distance),
                'one_way': bool(getattr(e, 'one_way', False)),
            }
            if target is not None:
                exit_dict['target_vnum'] = int(target)
            exits_data.append(exit_dict)

        room_dict: dict[str, Any] = {
            'vnum': int(r.vnum),
            'name': str(r.name),
            'desc': str(r.desc),
            'coords': {
                'x': x_val,
                'y': y_val,
                'z': z_val,
            },
            'exits': exits_data,
        }
        r_area = getattr(r, 'area_name', None) or (area_name if area_name else None)
        if r_area is not None:
            room_dict['area_name'] = str(r_area)
        r_file = getattr(r, 'area_file', None) or (area_file if area_file else None)
        if r_file is not None:
            room_dict['area_file'] = str(r_file)

        rooms_data.append(room_dict)

    return {
        'area': {
            'name': area_name,
            'file': area_file,
        },
        'bounds': computed_bounds,
        'rooms': rooms_data,
    }


def export_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> Path:
    """
    Serializes map dictionary to a JSON file on disk.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path


class JSONRenderer:
    """Renderer exporting map data to structured JSON format conforming to BaseRenderer."""

    def render(
        self,
        rdb: dict[int, Room],
        output_path: str | Path,
        header: AreaHeader | None = None,
        **options: Any,
    ) -> Path:
        indent = options.get('indent', 2)
        bounds = options.get('bounds')
        data = build_area_json(rdb, area_meta=header, bounds=bounds)
        return export_json(data, output_path, indent=indent)
