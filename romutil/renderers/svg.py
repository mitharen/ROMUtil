"""Isometric SVG vector map renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, overload

import svgwrite
from svgwrite import cm

from romutil.models import AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.renderers.base import BaseRenderer, adapt_legacy_rdb
from romutil.types import Coord3D, LayoutDiagnostics, RenderOptions, SolvedLayout, SolverStatus, WorldGraph


class _DynamicPalette:
    """Dynamic palette mapping Z index to an HSL color for backward compatibility."""

    def __init__(self, renderer: SVGRenderer) -> None:
        self._renderer: SVGRenderer = renderer

    def __getitem__(self, idx: int) -> str:
        return self._renderer.get_color(idx)

    def __len__(self) -> int:
        return len(self._renderer.get_unique_elevations())


class SVGRenderer:
    """Renderer generating 2D/3D isometric SVG vector maps conforming to BaseRenderer."""

    lift: float = 0.15
    colors: Any = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

    def __init__(
        self,
        name: str | Path = "",
        graph: WorldGraph | Dict[int, Room] | None = None,
        layout: SolvedLayout | List[Any] | None = None,
        options: RenderOptions | None = None,
        target_z: Optional[int] = None,
        *,
        rdb: Dict[int, Room] | None = None,
        exits: List[Any] | None = None,
    ) -> None:
        self.name: str = str(name)
        self.target_z: Optional[int] = target_z
        self.x_max: float = 0.0
        self.y_max: float = 0.0
        self.z_max: float = 0.0
        self.colors: Any = _DynamicPalette(self)
        self.custom_exits: Optional[List[Any]] = exits
        self.options: Optional[RenderOptions] = options

        if rdb is not None:
            self.graph, self.layout = adapt_legacy_rdb(rdb)
            self.rdb: Dict[int, Room] = rdb
        elif isinstance(graph, WorldGraph):
            self.graph = graph
            if isinstance(layout, SolvedLayout):
                self.layout = layout
            else:
                self.layout = SolvedLayout(
                    coordinates={},
                    diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
                )
            self.rdb = {}
        elif isinstance(graph, dict):
            # Legacy constructor: SVGRenderer(name, rdb, exits, target_z)
            if isinstance(options, int):
                self.target_z = options
                self.options = None
            self.graph, self.layout = adapt_legacy_rdb(graph)
            self.rdb = graph
            if isinstance(layout, list):
                self.custom_exits = layout
        else:
            self.graph = WorldGraph(rooms={}, exits=(), header=None)
            self.layout = (
                layout
                if isinstance(layout, SolvedLayout)
                else SolvedLayout(
                    coordinates={},
                    diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
                )
            )
            self.rdb = {}

    def get_unique_elevations(self) -> List[int]:
        if self.layout and self.layout.coordinates:
            zs = sorted(list(set(int(c.z) for c in self.layout.coordinates.values())))
            return zs if zs else [0]
        return [0]

    def get_color(self, z: int) -> str:
        unique_zs = self.get_unique_elevations()
        z_int = int(z)
        if len(unique_zs) <= 1:
            return "hsl(0, 75%, 50%)"
        if z_int in unique_zs:
            idx = unique_zs.index(z_int)
            hue = round((idx / (len(unique_zs) - 1)) * 300, 1)
        else:
            hue = round((z_int * 37) % 360, 1)
        return f"hsl({hue}, 75%, 50%)"

    def proj_room(self, room: Any) -> Optional[tuple[float, float]]:
        coord: Coord3D | None = None
        if isinstance(room, Coord3D):
            coord = room
        elif isinstance(room, int):
            coord = self.layout.coordinates.get(room) if self.layout else None
        else:
            vnum = getattr(room, "vnum", None)
            coord = self.layout.coordinates.get(vnum) if self.layout and vnum is not None else None

        if coord is None:
            return None
        return (
            2.0 + coord.x + self.lift * coord.z,
            2.0 + self.lift * self.z_max + (self.y_max - coord.y) - self.lift * coord.z,
        )

    def proj_exit(
        self,
        ex: Any,
        src_vnum: int | None = None,
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        if src_vnum is None:
            src_vnum = getattr(ex, "src", None)
        if src_vnum is None or self.layout is None or src_vnum not in self.layout.coordinates:
            return None

        src_coord = self.layout.coordinates[src_vnum]
        start = (
            2.0 + src_coord.x + 0.25 + self.lift * src_coord.z,
            2.0 + self.lift * self.z_max + (self.y_max - src_coord.y) + 0.25 - self.lift * src_coord.z,
        )

        target = getattr(ex, "target_vnum", None)
        dst_vnum = (
            target
            if target is not None and target != -1
            else getattr(ex, "dst_vnum", getattr(ex, "dst", None))
        )

        if dst_vnum is not None and dst_vnum in self.layout.coordinates:
            dst_coord = self.layout.coordinates[dst_vnum]
            end = (
                2.0 + dst_coord.x + 0.25 + self.lift * dst_coord.z,
                2.0 + self.lift * self.z_max + (self.y_max - dst_coord.y) + 0.25 - self.lift * dst_coord.z,
            )
        else:
            dir_attr = getattr(ex, "direction", 0)
            dir_val = dir_attr.value if hasattr(dir_attr, "value") else int(dir_attr)
            try:
                direction = Direction(dir_val)
            except Exception:
                direction = None

            if direction == Direction.north:
                end = (start[0], start[1] - 1.0)
            elif direction == Direction.east:
                end = (start[0] + 1.0, start[1])
            elif direction == Direction.south:
                end = (start[0], start[1] + 1.0)
            elif direction == Direction.west:
                end = (start[0] - 1.0, start[1])
            elif direction == Direction.up:
                end = (start[0] + self.lift, start[1] - self.lift)
            elif direction == Direction.down:
                end = (start[0] - self.lift, start[1] + self.lift)
            elif direction == Direction.northeast:
                end = (start[0] + 1.0, start[1] - 1.0)
            elif direction == Direction.northwest:
                end = (start[0] - 1.0, start[1] - 1.0)
            elif direction == Direction.southeast:
                end = (start[0] + 1.0, start[1] + 1.0)
            elif direction == Direction.southwest:
                end = (start[0] - 1.0, start[1] + 1.0)
            else:
                end = start

        return (start, end)

    def _add_controls(self, dwg: svgwrite.Drawing, plot_zs: List[int]) -> None:
        style_content = """
.elevation-layer { transition: opacity 0.2s ease, visibility 0.2s ease; }
.layer-toggle { cursor: pointer; user-select: none; }
.layer-toggle:hover rect { stroke-width: 0.08; filter: drop-shadow(0 0 2px rgba(0,0,0,0.5)); }
.layer-toggle.inactive rect { fill-opacity: 0.3; stroke: #999; }
.layer-toggle.inactive text { fill: #888; }
"""
        dwg.defs.add(dwg.style(style_content))

        script_content = """
function toggleElevation(z) {
    var layer = document.getElementById('elevation-' + z);
    var btn = document.getElementById('toggle-btn-' + z);
    if (!layer) return;
    var isHidden = layer.getAttribute('visibility') === 'hidden';
    if (isHidden) {
        layer.setAttribute('visibility', 'visible');
        if (btn) {
            btn.classList.remove('inactive');
            var r = btn.querySelector('rect');
            if (r) r.setAttribute('fill-opacity', '1.0');
        }
    } else {
        layer.setAttribute('visibility', 'hidden');
        if (btn) {
            btn.classList.add('inactive');
            var r = btn.querySelector('rect');
            if (r) r.setAttribute('fill-opacity', '0.3');
        }
    }
}
"""
        dwg.defs.add(dwg.script(content=script_content))

        ctrl_g = dwg.g(id="elevation-controls", class_="elevation-controls")
        ctrl_g.add(
            dwg.text(
                "Elevation Layers:",
                insert=(0.5, 0.7),
                font_size="0.4",
                font_family="Arial",
                font_weight="bold",
                fill="#222",
            )
        )

        for i, z in enumerate(plot_zs):
            btn_x = 4.2 + i * 2.0
            btn_y = 0.35
            color = self.get_color(z)
            btn_g = dwg.g(id=f"toggle-btn-{z}", class_="layer-toggle", onclick=f"toggleElevation('{z}')")
            btn_g.add(
                dwg.rect(
                    insert=(btn_x, btn_y),
                    size=(1.8, 0.55),
                    rx=0.1,
                    ry=0.1,
                    fill=color,
                    stroke="black",
                    stroke_width=0.03,
                    fill_opacity=1.0,
                )
            )
            btn_g.add(
                dwg.text(
                    f"Level {z}",
                    insert=(btn_x + 0.9, btn_y + 0.38),
                    text_anchor="middle",
                    font_size="0.28",
                    font_family="Arial",
                    font_weight="bold",
                    fill="black",
                )
            )
            ctrl_g.add(btn_g)

        dwg.add(ctrl_g)

    def plot(self) -> None:
        if self.layout and self.layout.coordinates:
            coords = list(self.layout.coordinates.values())
            self.x_max = float(max(c.x for c in coords))
            self.y_max = float(max(c.y for c in coords))
            self.z_max = float(max(c.z for c in coords))
        else:
            self.x_max = 0.0
            self.y_max = 0.0
            self.z_max = 0.0

        z_space = self.z_max * self.lift
        all_unique_zs = self.get_unique_elevations()
        if self.target_z is not None:
            plot_zs = [int(self.target_z)]
        else:
            plot_zs = sorted(all_unique_zs)

        needed_width = 4.5 + len(plot_zs) * 2.0
        svg_width = max(self.x_max + 4 + 11 + z_space, needed_width)
        svg_height = self.y_max + 4 + 4 + z_space

        Path(self.name).parent.mkdir(parents=True, exist_ok=True)
        dwg = svgwrite.Drawing(
            self.name,
            profile="full",
            size=(svg_width * cm, svg_height * cm),
            viewBox=f"0 0 {svg_width} {svg_height}",
            debug=False,
        )

        self._add_controls(dwg, plot_zs)

        for z in plot_zs:
            layer = dwg.g(id=f"elevation-{z}", class_="elevation-layer")
            layer.attribs["data-z"] = str(z)

            # Gather exits to draw for this layer
            layer_exit_items: List[tuple[int, Any]] = []
            if self.custom_exits is not None:
                for ex in self.custom_exits:
                    s_v = getattr(ex, "src", None)
                    if s_v is not None and s_v in self.layout.coordinates:
                        layer_exit_items.append((s_v, ex))
            else:
                for r in self.graph.rooms.values():
                    if r.vnum in self.layout.coordinates:
                        for ex in r.exits:
                            layer_exit_items.append((r.vnum, ex))

            active_layer_exits: List[tuple[int, Any]] = []
            for src_v, ex in layer_exit_items:
                src_z = self.layout.coordinates[src_v].z
                target = getattr(ex, "target_vnum", None)
                dst_v = (
                    target
                    if target is not None and target != -1
                    else getattr(ex, "dst_vnum", getattr(ex, "dst", None))
                )
                dst_z = (
                    self.layout.coordinates[dst_v].z
                    if dst_v is not None and dst_v in self.layout.coordinates
                    else None
                )

                if self.target_z is not None:
                    if src_z == z or dst_z == z:
                        active_layer_exits.append((src_v, ex))
                else:
                    higher_z = max(src_z, dst_z) if dst_z is not None else src_z
                    if higher_z == z:
                        active_layer_exits.append((src_v, ex))

            for src_v, ex in active_layer_exits:
                exit_projection = self.proj_exit(ex, src_vnum=src_v)
                if exit_projection is None:
                    continue

                target = getattr(ex, "target_vnum", None)
                dst_v = (
                    target
                    if target is not None and target != -1
                    else getattr(ex, "dst_vnum", getattr(ex, "dst", None))
                )

                if getattr(ex, "one_way", None) is not None and not isinstance(ex, ExitDef):
                    is_one_way = bool(getattr(ex, "one_way"))
                elif dst_v is not None and dst_v in self.graph.rooms:
                    dst_room = self.graph.rooms[dst_v]
                    try:
                        dir_attr = getattr(ex, "direction", 0)
                        dir_val = dir_attr.value if hasattr(dir_attr, "value") else int(dir_attr)
                        inv_dir = Direction(dir_val).invert().value
                        has_return = any(
                            (getattr(re.direction, "value", re.direction) == inv_dir)
                            and (getattr(re, "target_vnum", getattr(re, "dst_vnum", getattr(re, "dst", None))) == src_v)
                            for re in dst_room.exits
                        )
                        is_one_way = not has_return
                    except Exception:
                        is_one_way = False
                else:
                    is_one_way = True

                color = "red" if is_one_way else "black"
                layer.add(
                    dwg.line(
                        start=exit_projection[0],
                        end=exit_projection[1],
                        stroke_width=0.05,
                        stroke=color,
                    )
                )

            layer_rooms = [
                r
                for r in self.graph.rooms.values()
                if r.vnum in self.layout.coordinates and self.layout.coordinates[r.vnum].z == z
            ]
            layer_rooms.sort(
                key=lambda r: (
                    -float(self.layout.coordinates[r.vnum].y),
                    float(self.layout.coordinates[r.vnum].x),
                    int(r.vnum),
                )
            )
            layer_descs = []

            for room in layer_rooms:
                room_pos = self.proj_room(room)
                if room_pos is None:
                    continue

                g = dwg.g(visibility="hidden")
                exit_names = []
                for ex in room.exits:
                    dir_attr = getattr(ex, "direction", 0)
                    dir_val = dir_attr.value if hasattr(dir_attr, "value") else int(dir_attr)
                    try:
                        exit_names.append(Direction(dir_val).name)
                    except (ValueError, KeyError):
                        exit_names.append(str(dir_attr))

                etext = "Exits: " + ", ".join(exit_names)
                desc = room.description.split("\n") + [etext]
                g.add(
                    dwg.rect(
                        fill="white",
                        insert=(room_pos[0] + 0.5, room_pos[1] + 0.5),
                        size=(11, 2 + len(desc) / 3),
                        stroke="black",
                        stroke_width=0.05,
                    )
                )
                text = dwg.text(
                    "",
                    insert=(room_pos[0] + 0.7, room_pos[1] + 1.1),
                    font_size="0.3",
                    font_family="Arial",
                    fill="black",
                )
                text.add(dwg.tspan(room.name, font_size="0.4"))
                for line in desc:
                    text.add(dwg.tspan(line, x=[room_pos[0] + 0.7], dy=["1.4em"]))
                g.add(text)

                room_z = int(self.layout.coordinates[room.vnum].z)
                r_elem = dwg.rect(
                    insert=room_pos,
                    size=(0.5, 0.5),
                    fill=self.get_color(room_z),
                    stroke="black",
                    stroke_width=0.025,
                )
                s = dwg.set(to="visible")
                s.set_target("visibility")
                s.set_timing(begin=r_elem.get_id() + ".mouseover", end=r_elem.get_id() + ".mouseout")
                g.add(s)
                layer.add(r_elem)
                layer_descs.append(g)

            for g in layer_descs:
                layer.add(g)

            dwg.add(layer)

        dwg.save()

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
        """Render a solved layout of a WorldGraph to output_path as isometric SVG.

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
            split_levels = (
                actual_options.split_levels if actual_options else False
            ) or bool(extra_options.get("split_levels", False))
            target_z = extra_options.get("target_z")
            outbase = extra_options.get("outbase")
            exits = extra_options.get("exits")
        else:
            # Legacy signature: render(rdb, output_path, header=None, **options)
            legacy_rdb = graph
            actual_output_path = Path(layout if isinstance(layout, (str, Path)) else str(layout))
            legacy_header = header if header is not None else (
                output_path if isinstance(output_path, (AreaHeader, tuple, list)) or hasattr(output_path, "filename") or hasattr(output_path, "name") else None
            )
            actual_graph, actual_layout = adapt_legacy_rdb(legacy_rdb, legacy_header)
            actual_options = options
            split_levels = (
                actual_options.split_levels if actual_options else False
            ) or bool(extra_options.get("split_levels", False))
            target_z = extra_options.get("target_z")
            outbase = extra_options.get("outbase")
            exits = extra_options.get("exits")

        if split_levels:
            unique_zs = sorted(list(set(int(c.z) for c in actual_layout.coordinates.values())))
            if not unique_zs:
                unique_zs = [0]
            base = outbase if outbase else (
                str(actual_output_path)[:-4] if str(actual_output_path).endswith(".svg") else str(actual_output_path)
            )
            for z in unique_zs:
                level_name = f"{base}_z{z}.svg"
                renderer = SVGRenderer(
                    name=level_name,
                    graph=actual_graph,
                    layout=actual_layout,
                    options=actual_options,
                    target_z=z,
                    exits=exits,
                )
                renderer.plot()
            return actual_output_path
        else:
            renderer = SVGRenderer(
                name=str(actual_output_path),
                graph=actual_graph,
                layout=actual_layout,
                options=actual_options,
                target_z=target_z,
                exits=exits,
            )
            renderer.plot()
            return actual_output_path


__all__ = [
    "SVGRenderer",
    "_DynamicPalette",
]
