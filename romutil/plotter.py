import svgwrite
from svgwrite import cm
from romutil.models import Direction

class Plotter:
    lift = 0.15
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

    def __init__(self, name, rdb, exits):
        self.name = name
        self.rdb = rdb
        self.exits = exits
        self.x_max = 0
        self.y_max = 0
        self.z_max = 0

    def proj_room(self, room):
        if None in (room.x, room.y, room.z):
            return None
        return (
            2 + room.x + self.lift * room.z,
            2 + self.lift * self.z_max + (self.y_max - room.y) - self.lift * room.z
        )

    def proj_exit(self, ex):
        if None in (self.rdb[ex.src].x, self.rdb[ex.src].y, self.rdb[ex.src].z):
            return None
        start = (
            2 + self.rdb[ex.src].x + 0.25 + self.lift * self.rdb[ex.src].z,
            2 + self.lift * self.z_max + (self.y_max - self.rdb[ex.src].y) + 0.25 - self.lift * self.rdb[ex.src].z
        )

        if ex.dst in self.rdb:
            if None in (self.rdb[ex.dst].x, self.rdb[ex.dst].y, self.rdb[ex.dst].z):
                return None
            end = (
                2 + self.rdb[ex.dst].x + 0.25 + self.lift * self.rdb[ex.dst].z,
                2 + self.lift * self.z_max + (self.y_max - self.rdb[ex.dst].y) + 0.25 - self.lift * self.rdb[ex.dst].z
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

    def plot(self):
        self.x_max = max([r.x for r in self.rdb.values()])
        self.y_max = max([r.y for r in self.rdb.values()])
        self.z_max = max([r.z for r in self.rdb.values()])
        z_space = self.z_max * self.lift

        dwg = svgwrite.Drawing(
            self.name,
            profile='full',
            size=((self.x_max + 4 + 11 + z_space) * cm, (self.y_max + 4 + 4 + z_space) * cm),
            viewBox=f'0 0 {self.x_max + 4 + 11 + z_space} {self.y_max + 4 + 4 + z_space}'
        )

        exits = sorted(self.exits, key=lambda x: max(self.rdb[x.src].z, self.rdb[x.dst].z))
        rooms = sorted(self.rdb.values(), key=lambda r: r.z)
        descs = []

        while len(exits) or len(rooms):
            e = self.rdb[exits[0].src].z if len(exits) else None
            r = rooms[0].z if len(rooms) else None

            if e is not None and (r is None or r >= e):
                ex = exits.pop(0)
                projection = self.proj_exit(ex)
                if projection is None:
                    continue
                color = 'red' if ex.one_way else 'black'
                dwg.add(dwg.line(start=projection[0], end=projection[1], stroke_width=0.05, stroke=color))
            else:
                room = rooms.pop(0)
                if room.dummy:
                    continue
                projection = self.proj_room(room)
                if projection is None:
                    continue
                g = dwg.g(visibility='hidden')
                etext = 'Exits: ' + ', '.join([ex.direction.name for ex in room.exits])
                desc = room.desc.split('\n') + [etext]
                g.add(dwg.rect(
                    fill='white',
                    insert=(projection[0] + 0.5, projection[1] + 0.5),
                    size=(11, 2 + len(desc) / 3),
                    stroke='black',
                    stroke_width=0.05
                ))
                text = dwg.text(
                    '',
                    insert=(projection[0] + 0.7, projection[1] + 1.1),
                    font_size='0.3',
                    font_family='Arial',
                    fill='black'
                )
                text.add(dwg.tspan(room.name, font_size='0.4'))
                for line in desc:
                    text.add(dwg.tspan(line, x=[projection[0] + 0.7], dy=['1.4em']))
                g.add(text)
                r_elem = dwg.rect(
                    insert=projection,
                    size=(0.5, 0.5),
                    fill=self.colors[min(6, int(room.z))],
                    stroke='black',
                    stroke_width=0.025
                )
                s = dwg.set(to='visible')
                s.set_target('visibility')
                s.set_timing(begin=r_elem.get_id() + '.mouseover', end=r_elem.get_id() + '.mouseout')
                g.add(s)
                dwg.add(r_elem)
                descs.append(g)

        for g in descs:
            dwg.add(g)

        dwg.save()
