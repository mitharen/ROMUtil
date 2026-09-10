"""Isometric SVG vector map renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import svgwrite
from svgwrite import cm

from romutil.models import AreaHeader, Direction, Exit, Room
from romutil.renderers.base import BaseRenderer


class _DynamicPalette:
    """Dynamic palette mapping Z index to an HSL color for backward compatibility."""

    def __init__(self, plotter: Plotter) -> None:
        self._plotter = plotter

    def __getitem__(self, idx: int) -> str:
        return self._plotter.get_color(idx)

    def __len__(self) -> int:
        return len(self._plotter.get_unique_elevations())


class Plotter:
    lift: float = 0.15
    colors: Any = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

    def __init__(
        self,
        name: str | Path,
        rdb: Dict[int, Room],
        exits: List[Exit],
        target_z: Optional[int] = None,
    ) -> None:
        self.name: str = str(name)
        self.rdb: Dict[int, Room] = rdb
        self.exits: List[Exit] = exits
        self.target_z: Optional[int] = target_z
        self.x_max: float = 0.0
        self.y_max: float = 0.0
        self.z_max: float = 0.0
        self.colors: Any = _DynamicPalette(self)

    def get_unique_elevations(self) -> List[int]:
        valid = [r for r in self.rdb.values() if r.z is not None and not r.dummy]
        zs = sorted(list(set(int(r.z) for r in valid if r.z is not None)))
        return zs if zs else [0]

    def get_color(self, z: int) -> str:
        unique_zs = self.get_unique_elevations()
        z_int = int(z)
        if len(unique_zs) <= 1:
            return 'hsl(0, 75%, 50%)'
        if z_int in unique_zs:
            idx = unique_zs.index(z_int)
            hue = round((idx / (len(unique_zs) - 1)) * 300, 1)
        else:
            hue = round((z_int * 37) % 360, 1)
        return f'hsl({hue}, 75%, 50%)'

    def proj_room(self, room: Room) -> Optional[tuple[float, float]]:
        if room.x is None or room.y is None or room.z is None:
            return None
        return (
            2.0 + room.x + self.lift * room.z,
            2.0 + self.lift * self.z_max + (self.y_max - room.y) - self.lift * room.z,
        )

    def proj_exit(self, ex: Exit) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        if ex.src not in self.rdb:
            return None
        src_room = self.rdb[ex.src]
        if src_room.x is None or src_room.y is None or src_room.z is None:
            return None
        start = (
            2.0 + src_room.x + 0.25 + self.lift * src_room.z,
            2.0 + self.lift * self.z_max + (self.y_max - src_room.y) + 0.25 - self.lift * src_room.z,
        )

        if ex.dst in self.rdb and not getattr(self.rdb[ex.dst], 'dummy', False):
            dst_room = self.rdb[ex.dst]
            if dst_room.x is None or dst_room.y is None or dst_room.z is None:
                return None
            end = (
                2.0 + dst_room.x + 0.25 + self.lift * dst_room.z,
                2.0 + self.lift * self.z_max + (self.y_max - dst_room.y) + 0.25 - self.lift * dst_room.z,
            )
        else:
            if ex.direction == Direction.north:
                end = (start[0], start[1] - 1.0)
            elif ex.direction == Direction.east:
                end = (start[0] + 1.0, start[1])
            elif ex.direction == Direction.south:
                end = (start[0], start[1] + 1.0)
            elif ex.direction == Direction.west:
                end = (start[0] - 1.0, start[1])
            elif ex.direction == Direction.up:
                end = (start[0] + self.lift, start[1] - self.lift)
            elif ex.direction == Direction.down:
                end = (start[0] - self.lift, start[1] + self.lift)
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

        ctrl_g = dwg.g(id='elevation-controls', class_='elevation-controls')
        ctrl_g.add(
            dwg.text(
                'Elevation Layers:',
                insert=(0.5, 0.7),
                font_size='0.4',
                font_family='Arial',
                font_weight='bold',
                fill='#222',
            )
        )

        for i, z in enumerate(plot_zs):
            btn_x = 4.2 + i * 2.0
            btn_y = 0.35
            color = self.get_color(z)
            btn_g = dwg.g(id=f'toggle-btn-{z}', class_='layer-toggle', onclick=f"toggleElevation('{z}')")
            btn_g.add(
                dwg.rect(
                    insert=(btn_x, btn_y),
                    size=(1.8, 0.55),
                    rx=0.1,
                    ry=0.1,
                    fill=color,
                    stroke='black',
                    stroke_width=0.03,
                    fill_opacity=1.0,
                )
            )
            btn_g.add(
                dwg.text(
                    f'Level {z}',
                    insert=(btn_x + 0.9, btn_y + 0.38),
                    text_anchor='middle',
                    font_size='0.28',
                    font_family='Arial',
                    font_weight='bold',
                    fill='black',
                )
            )
            ctrl_g.add(btn_g)

        dwg.add(ctrl_g)

    def plot(self) -> None:
        valid_rooms = [
            r for r in self.rdb.values()
            if r.x is not None and r.y is not None and r.z is not None
        ]
        if valid_rooms:
            self.x_max = float(max(r.x for r in valid_rooms if r.x is not None))
            self.y_max = float(max(r.y for r in valid_rooms if r.y is not None))
            self.z_max = float(max(r.z for r in valid_rooms if r.z is not None))
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
            profile='full',
            size=(svg_width * cm, svg_height * cm),
            viewBox=f'0 0 {svg_width} {svg_height}',
            debug=False,
        )

        self._add_controls(dwg, plot_zs)

        for z in plot_zs:
            layer = dwg.g(id=f'elevation-{z}', class_='elevation-layer')
            layer.attribs['data-z'] = str(z)

            if self.target_z is not None:
                layer_exits = [
                    ex
                    for ex in self.exits
                    if ex.src in self.rdb
                    and self.rdb[ex.src].z is not None
                    and (
                        self.rdb[ex.src].z == z
                        or (
                            ex.dst in self.rdb
                            and self.rdb[ex.dst].z is not None
                            and self.rdb[ex.dst].z == z
                        )
                    )
                ]
            else:
                layer_exits = []
                for ex in self.exits:
                    if ex.src not in self.rdb or self.rdb[ex.src].z is None:
                        continue
                    src_z = self.rdb[ex.src].z
                    assert src_z is not None
                    if ex.dst in self.rdb and self.rdb[ex.dst].z is not None:
                        dst_z = self.rdb[ex.dst].z
                        assert dst_z is not None
                        higher_z = max(src_z, dst_z)
                    else:
                        higher_z = src_z
                    if higher_z == z:
                        layer_exits.append(ex)

            for ex in layer_exits:
                exit_projection = self.proj_exit(ex)
                if exit_projection is None:
                    continue
                color = 'red' if ex.one_way else 'black'
                layer.add(dwg.line(start=exit_projection[0], end=exit_projection[1], stroke_width=0.05, stroke=color))

            layer_exits.sort(key=lambda ex: (getattr(ex, 'src', 0), getattr(ex, 'dst', 0)))

            layer_rooms = [r for r in valid_rooms if not r.dummy and r.z == z]
            layer_rooms.sort(
                key=lambda r: (
                    -float(r.y if r.y is not None else 0),
                    float(r.x if r.x is not None else 0),
                    getattr(r, 'vnum', 0),
                )
            )
            layer_descs = []

            for room in layer_rooms:
                room_pos = self.proj_room(room)
                if room_pos is None:
                    continue

                g = dwg.g(visibility='hidden')
                etext = 'Exits: ' + ', '.join([ex.direction.name for ex in room.exits])
                desc = room.desc.split('\n') + [etext]
                g.add(
                    dwg.rect(
                        fill='white',
                        insert=(room_pos[0] + 0.5, room_pos[1] + 0.5),
                        size=(11, 2 + len(desc) / 3),
                        stroke='black',
                        stroke_width=0.05,
                    )
                )
                text = dwg.text(
                    '',
                    insert=(room_pos[0] + 0.7, room_pos[1] + 1.1),
                    font_size='0.3',
                    font_family='Arial',
                    fill='black',
                )
                text.add(dwg.tspan(room.name, font_size='0.4'))
                for line in desc:
                    text.add(dwg.tspan(line, x=[room_pos[0] + 0.7], dy=['1.4em']))
                g.add(text)

                room_z = int(room.z if room.z is not None else 0)
                r_elem = dwg.rect(
                    insert=room_pos,
                    size=(0.5, 0.5),
                    fill=self.get_color(room_z),
                    stroke='black',
                    stroke_width=0.025,
                )
                s = dwg.set(to='visible')
                s.set_target('visibility')
                s.set_timing(begin=r_elem.get_id() + '.mouseover', end=r_elem.get_id() + '.mouseout')
                g.add(s)
                layer.add(r_elem)
                layer_descs.append(g)

            for g in layer_descs:
                layer.add(g)

            dwg.add(layer)

        dwg.save()


class SVGRenderer:
    """Renderer generating 2D/3D isometric SVG vector maps conforming to BaseRenderer."""

    def render(
        self,
        rdb: dict[int, Room],
        output_path: str | Path,
        header: AreaHeader | None = None,
        **options: Any,
    ) -> Path:
        out_path = Path(output_path)
        exits = options.get('exits')
        if exits is None:
            exits = [
                e
                for r in rdb.values()
                if not getattr(r, 'dummy', False)
                for e in r.exits
            ]

        split_levels = bool(options.get('split_levels', False))
        target_z = options.get('target_z')
        outbase = options.get('outbase')

        if split_levels:
            unique_zs = sorted(
                list(
                    set(
                        int(r.z)
                        for r in rdb.values()
                        if not getattr(r, 'dummy', False) and r.z is not None
                    )
                )
            )
            if not unique_zs:
                unique_zs = [0]
            base = outbase if outbase else (str(out_path)[:-4] if str(out_path).endswith('.svg') else str(out_path))
            for z in unique_zs:
                level_name = f"{base}_z{z}.svg"
                plotter = Plotter(level_name, rdb, exits, target_z=z)
                plotter.plot()
            return out_path
        else:
            plotter = Plotter(str(out_path), rdb, exits, target_z=target_z)
            plotter.plot()
            return out_path
