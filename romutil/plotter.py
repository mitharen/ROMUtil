import svgwrite
from svgwrite import cm
from romutil.models import Direction


class _DynamicPalette:
    """Dynamic palette mapping Z index to an HSL color for backward compatibility."""

    def __init__(self, plotter):
        self._plotter = plotter

    def __getitem__(self, idx):
        return self._plotter.get_color(idx)

    def __len__(self):
        return len(self._plotter.get_unique_elevations())


class Plotter:
    lift = 0.15
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

    def __init__(self, name, rdb, exits, target_z=None):
        self.name = name
        self.rdb = rdb
        self.exits = exits
        self.target_z = target_z
        self.x_max = 0
        self.y_max = 0
        self.z_max = 0
        self.colors = _DynamicPalette(self)

    def get_unique_elevations(self):
        valid = [r for r in self.rdb.values() if r.z is not None and not r.dummy]
        zs = sorted(list(set(int(r.z) for r in valid)))
        return zs if zs else [0]

    def get_color(self, z):
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

    def proj_room(self, room):
        if None in (room.x, room.y, room.z):
            return None
        return (
            2 + room.x + self.lift * room.z,
            2 + self.lift * self.z_max + (self.y_max - room.y) - self.lift * room.z,
        )

    def proj_exit(self, ex):
        if ex.src not in self.rdb or None in (self.rdb[ex.src].x, self.rdb[ex.src].y, self.rdb[ex.src].z):
            return None
        start = (
            2 + self.rdb[ex.src].x + 0.25 + self.lift * self.rdb[ex.src].z,
            2 + self.lift * self.z_max + (self.y_max - self.rdb[ex.src].y) + 0.25 - self.lift * self.rdb[ex.src].z,
        )

        if ex.dst in self.rdb:
            if None in (self.rdb[ex.dst].x, self.rdb[ex.dst].y, self.rdb[ex.dst].z):
                return None
            end = (
                2 + self.rdb[ex.dst].x + 0.25 + self.lift * self.rdb[ex.dst].z,
                2 + self.lift * self.z_max + (self.y_max - self.rdb[ex.dst].y) + 0.25 - self.lift * self.rdb[ex.dst].z,
            )
        else:
            if ex.direction == Direction.north:
                end = (start[0], start[1] - 1)
            elif ex.direction == Direction.east:
                end = (start[0] + 1, start[1])
            elif ex.direction == Direction.south:
                end = (start[0], start[1] + 1)
            elif ex.direction == Direction.west:
                end = (start[0] - 1, start[1])
            elif ex.direction == Direction.up:
                end = (start[0] + self.lift, start[1] - self.lift)
            elif ex.direction == Direction.down:
                end = (start[0] - self.lift, start[1] + self.lift)
            else:
                end = start

        return (start, end)

    def _add_controls(self, dwg, plot_zs):
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

    def plot(self):
        valid_rooms = [r for r in self.rdb.values() if r.x is not None and r.y is not None and r.z is not None]
        if valid_rooms:
            self.x_max = max(r.x for r in valid_rooms)
            self.y_max = max(r.y for r in valid_rooms)
            self.z_max = max(r.z for r in valid_rooms)
        else:
            self.x_max = 0
            self.y_max = 0
            self.z_max = 0
        z_space = self.z_max * self.lift

        all_unique_zs = self.get_unique_elevations()
        if self.target_z is not None:
            plot_zs = [int(self.target_z)]
        else:
            plot_zs = all_unique_zs

        needed_width = 4.5 + len(plot_zs) * 2.0
        svg_width = max(self.x_max + 4 + 11 + z_space, needed_width)
        svg_height = self.y_max + 4 + 4 + z_space

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
                        int(self.rdb[ex.src].z) == z
                        or (
                            ex.dst in self.rdb
                            and self.rdb[ex.dst].z is not None
                            and int(self.rdb[ex.dst].z) == z
                        )
                    )
                ]
            else:
                layer_exits = [
                    ex
                    for ex in self.exits
                    if ex.src in self.rdb
                    and self.rdb[ex.src].z is not None
                    and (
                        int(
                            max(
                                self.rdb[ex.src].z,
                                self.rdb[ex.dst].z
                                if ex.dst in self.rdb and self.rdb[ex.dst].z is not None
                                else self.rdb[ex.src].z,
                            )
                        )
                        == z
                    )
                ]

            for ex in layer_exits:
                projection = self.proj_exit(ex)
                if projection is None:
                    continue
                color = 'red' if ex.one_way else 'black'
                layer.add(dwg.line(start=projection[0], end=projection[1], stroke_width=0.05, stroke=color))

            layer_rooms = [r for r in valid_rooms if not r.dummy and int(r.z) == z]
            layer_descs = []

            for room in layer_rooms:
                projection = self.proj_room(room)
                if projection is None:
                    continue

                g = dwg.g(visibility='hidden')
                etext = 'Exits: ' + ', '.join([ex.direction.name for ex in room.exits])
                desc = room.desc.split('\n') + [etext]
                g.add(
                    dwg.rect(
                        fill='white',
                        insert=(projection[0] + 0.5, projection[1] + 0.5),
                        size=(11, 2 + len(desc) / 3),
                        stroke='black',
                        stroke_width=0.05,
                    )
                )
                text = dwg.text(
                    '',
                    insert=(projection[0] + 0.7, projection[1] + 1.1),
                    font_size='0.3',
                    font_family='Arial',
                    fill='black',
                )
                text.add(dwg.tspan(room.name, font_size='0.4'))
                for line in desc:
                    text.add(dwg.tspan(line, x=[projection[0] + 0.7], dy=['1.4em']))
                g.add(text)

                r_elem = dwg.rect(
                    insert=projection,
                    size=(0.5, 0.5),
                    fill=self.get_color(room.z),
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
