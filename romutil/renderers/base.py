"""Base protocol definition for map renderers and legacy adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, overload, runtime_checkable

from romutil.models import AreaHeader, ExitDef, Room, RoomDef
from romutil.types import (
    Coord3D,
    LayoutDiagnostics,
    RenderOptions,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


@runtime_checkable
class BaseRenderer(Protocol):
    """Protocol for renderers outputting solved layouts to visual or structured formats.

    Primary signature: render(graph, layout, output_path, options=None)
    Legacy fallback: render(rdb, output_path, header=None, **options)
    """

    @overload
    def render(
        self,
        graph: WorldGraph,
        layout: SolvedLayout,
        output_path: Path | str,
        options: RenderOptions | None = None,
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
        header: AreaHeader | Any = None,
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
        header: AreaHeader | Any = None,
        **extra_options: Any,
    ) -> Path:
        """Render a solved layout of a WorldGraph or legacy room database to output_path."""
        ...


def adapt_legacy_rdb(
    rdb: dict[int, Any] | list[Any],
    header: AreaHeader | Any = None,
) -> tuple[WorldGraph, SolvedLayout]:
    """Adapt legacy room database (dict or list of Room objects) to WorldGraph and SolvedLayout.

    Extracts (x, y, z) coordinates into SolvedLayout without mutating room objects,
    and filters dummy fixup rooms.
    """
    if isinstance(rdb, dict):
        room_list = list(rdb.values())
    elif isinstance(rdb, (list, tuple)):
        room_list = list(rdb)
    else:
        room_list = []

    # Filter out dummy fixup rooms
    valid_rooms = [r for r in room_list if not getattr(r, "dummy", False)]

    # Area header normalization
    area_header: AreaHeader | None = None
    if isinstance(header, AreaHeader):
        area_header = header
    elif header is not None:
        if isinstance(header, (tuple, list)):
            fn = str(header[0]) if len(header) > 0 else ""
            nm = str(header[1]) if len(header) > 1 else ""
            area_header = AreaHeader(filename=fn, name=nm, builder="", vnum_min=0, vnum_max=0)
        else:
            fn = str(getattr(header, "filename", getattr(header, "file", "")))
            nm = str(getattr(header, "name", ""))
            bld = str(getattr(header, "builder", ""))
            vmin = int(getattr(header, "vnum_min", 0))
            vmax = int(getattr(header, "vnum_max", 0))
            area_header = AreaHeader(filename=fn, name=nm, builder=bld, vnum_min=vmin, vnum_max=vmax)

    rooms_dict: dict[int, RoomDef] = {}
    coords_dict: dict[int, Coord3D] = {}

    for r in valid_rooms:
        vnum = int(getattr(r, "vnum", 0))
        rx = getattr(r, "x", None)
        ry = getattr(r, "y", None)
        rz = getattr(r, "z", None)
        if rx is not None and ry is not None and rz is not None:
            coords_dict[vnum] = Coord3D(x=int(round(rx)), y=int(round(ry)), z=int(round(rz)))

        if isinstance(r, RoomDef):
            rooms_dict[vnum] = r
        else:
            name = str(getattr(r, "name", ""))
            desc = str(getattr(r, "description", getattr(r, "desc", "")))
            area_name = getattr(r, "area_name", None)
            area_file = getattr(r, "area_file", None)
            room_flags = getattr(r, "room_flags", 0)
            sector = getattr(r, "sector", 0)

            converted_exits: list[ExitDef] = []
            for e in getattr(r, "exits", ()):
                if isinstance(e, ExitDef):
                    converted_exits.append(e)
                else:
                    dir_attr = getattr(e, "direction", 0)
                    dir_val = dir_attr.value if hasattr(dir_attr, "value") else int(dir_attr)
                    target = getattr(e, "target_vnum", None)
                    dst_val = (
                        target
                        if target is not None and target != -1
                        else getattr(e, "dst", getattr(e, "dst_vnum", 0))
                    )
                    converted_exits.append(
                        ExitDef(
                            direction=dir_val,
                            dst_vnum=int(dst_val),
                            description=str(getattr(e, "description", "")),
                            keyword=str(getattr(e, "keyword", "")),
                            key_vnum=int(getattr(e, "key_vnum", 0)),
                            flags=int(getattr(e, "flags", 0)),
                        )
                    )

            rooms_dict[vnum] = RoomDef(
                vnum=vnum,
                name=name,
                description=desc,
                room_flags=room_flags,
                sector=sector,
                exits=tuple(converted_exits),
                area_name=area_name,
                area_file=area_file,
            )

    all_exits = tuple(e for r_def in rooms_dict.values() for e in r_def.exits)
    graph = WorldGraph(rooms=rooms_dict, exits=all_exits, header=area_header)
    layout = SolvedLayout(
        coordinates=coords_dict,
        diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
    )
    return graph, layout


__all__ = [
    "BaseRenderer",
    "adapt_legacy_rdb",
]
