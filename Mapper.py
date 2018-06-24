#!/usr/bin/env python

import sys
import AreaParser
from pyomo.environ import *
import svgwrite
from svgwrite import cm
import enum

class Direction(enum.IntEnum):
    north=0
    east=1
    up=2
    south=3
    west=4
    down=5
    mod=6
    def invert(self):
        return Direction((self + self.mod/2)%self.mod)

direction_matrix = [Direction.north,
                    Direction.east,
                    Direction.south,
                    Direction.west,
                    Direction.up,
                    Direction.down]

class Room():
    def __init__(self, r):
        self.vnum = r[0]
        self.name = r[1]
        self.exits = [] if not r[2] else [Exit(e, self.vnum) for e in r[2] if e is not None]
        self.fixups = []

    def replace_exit(self, orig, replacement, distance):
        for e in self.exits:
            if e.n_room == orig:
                e.n_room = replacement
                e.distance += distance

    def __repr__(self):
        return '[%d: %s] {%s}'%(self.vnum, self.name, self.exits)
            
class Exit():
    def __init__(self, e, source, fake=False, distance=1):
        self.p_room = source
        self.n_room = e[1]
        self.direction = Direction(direction_matrix[e[0]])
        self.distance = distance

    def __eq__(self, e):
        if self.p_room == e.p_room and self.n_room == e.n_room and \
           self.direction == e.direction: return True
        if self.p_room == e.n_room and self.n_room == e.p_room and \
           self.direction == e.direction.invert(): return True

    def __contains__(self, e):
        return e in (self.p_room, self.n_room)

    def __repr__(self):
        return '%d -> %d (%d %s)'%(self.p_room, self.n_room, self.distance, self.direction.name)

def main():
    parser = AreaParser.Parser()
    with open(sys.argv[1], 'r') as f:
        area = parser.parse(f.read())

    for section in area:
        if section[0] == '#ROOMS':
            rooms = section[1]
            print('%s: %d'%(section[0], len(section[1]) if section[1] else 0))
        elif section[0] == '#AREA':
            area = section[1]
            print(area)
#    return

    rdb = {r[0]: Room(r) for r in rooms}
    for vnum, r in rdb.items():
        # remove exits to other areas
        r.exits = [e for e in r.exits if e.n_room in rdb.keys()]
        # add placeholder for one-ways so we don't remove them later
        for e in r.exits:
             if not vnum in [e1.n_room for e1 in rdb[e.n_room].exits]:
                 rdb[e.n_room].exits.append(e)

    for vnum, r in list(rdb.items()):
        # clean up potential hallways
        if len(r.exits) == 2:
            # if bidirectional and straight
            if not all(vnum in [e1.n_room for e1 in rdb[e0.n_room].exits] for e0 in r.exits): continue
            if r.exits[0].direction == r.exits[1].direction.invert():
                rdb[r.exits[0].n_room].replace_exit(vnum, r.exits[1].n_room, r.exits[1].distance)
                rdb[r.exits[1].n_room].replace_exit(vnum, r.exits[0].n_room, r.exits[0].distance)
                rdb[r.exits[0].n_room].fixups.append((r, r.exits[0].direction.invert()))
                del rdb[vnum]
                print('[*] Trimmed hallway %d.'%(vnum))

    # collect unique exits
    exits = []
    for r in rdb.values():
        for e in r.exits:
            if e not in exits: exits.append(e)
    
    model = solve(rdb, exits)

    x_min = min([model.x[i].value for i in model.x if model.x[i].value])
    x_max = max([model.x[i].value for i in model.x if model.x[i].value])
    y_min = min([model.y[i].value for i in model.y if model.y[i].value])
    y_max = max([model.y[i].value for i in model.y if model.y[i].value])
    print(x_min, x_max, y_min, y_max)

    dwg = svgwrite.Drawing(sys.argv[1]+'.svg', profile='tiny',
                           size=((x_max - x_min + 6)*cm, (y_max - y_min + 6)*cm),
                           viewBox='%d %d %d %d'%(x_min-2, y_min-2, x_max+3, y_max+6))
    dwg_exits = dwg.add(dwg.g(id='exits', stroke='black'))
    dwg_rooms = dwg.add(dwg.g(id='rooms', fill='red'))

    def proj_room(room, border=2):
        return (model.x[room].value, y_max - model.y[room].value + y_min)

    def proj_exit(ex, border=2):
        return ((model.x[ex.p_room].value + .25, y_max - model.y[ex.p_room].value + y_min + .25),
                (model.x[ex.n_room].value + .25, y_max - model.y[ex.n_room].value + y_min + .25))

    def plot_fixup(dwg, room, x, y):
        for r, d in room.fixups:
            mod_x, mod_y = x, y
            if d == Direction.north:
                mod_y -= 1
            elif d == Direction.east:
                mod_x += 1
            elif d == Direction.south:
                mod_y += 1
            elif d == Direction.west:
                mod_x -= 1
            elif d == Direction.up:
                pass
            elif d == Direction.down:
                pass
            dwg_rooms.add(dwg.rect(insert=(mod_x, mod_y), size=(.5,.5), fill='blue'))
            plot_fixup(dwg, r, mod_x, mod_y)

    for ex in exits:
        projection = proj_exit(ex)
        dwg_exits.add(dwg.line(start=projection[0], end=projection[1], stroke_width=.1))
    for room in rdb.keys():
        print('[%d] %s:'%(rdb[room].vnum, rdb[room].name), model.x[room].value, model.y[room].value, model.z[room].value)
        if model.x[room].stale == True: continue
        projection = proj_room(room)
        dwg_rooms.add(dwg.rect(insert=projection, size=(.5,.5)))
        plot_fixup(dwg, rdb[room], projection[0], projection[1])
        
    dwg.save()
    exit(0)

def solve(rdb, exits):
    # construct model
    model = ConcreteModel()
    model.Rooms = Set(initialize=rdb.keys())
    model.Exits = RangeSet(0, len(exits)-1)
    model.Directions = RangeSet(0, Direction.mod.value)
    model.M = Param(initialize=sum([e.distance for e in exits]))
    # room position
    model.x = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    model.y = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    model.z = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    # exit lengths
    model.l_max = Var(model.Exits, within=PositiveIntegers)
    model.l_min = Param(model.Exits, initialize=lambda model, x: exits[x].distance)

    # objective to minimize max exit lengths
    model.obj = Objective(expr=sum([model.l_max[e] for e in model.Exits]))

    # constraints for exit crossings
    model.d_min = Param(initialize=1)
    model.crossings = ConstraintList()

    # constraints for relative position of rooms
    model.relations = Var(model.Exits, model.Exits, model.Directions, within=Boolean)
    model.relative_pos = ConstraintList()

    # add constraints
    for i, ex in enumerate(exits):
        print('[*] Exit %d/%d'%(i+1,len(exits)))

        # relative position O(e)
        if ex.direction not in (Direction.east, Direction.west):
            model.relative_pos.add(model.x[ex.p_room] == model.x[ex.n_room])
        if ex.direction not in (Direction.north, Direction.south):
            model.relative_pos.add(model.y[ex.p_room] == model.y[ex.n_room])
        if ex.direction not in (Direction.up, Direction.down):
            model.relative_pos.add(model.z[ex.p_room] == model.z[ex.n_room])

        if ex.direction == Direction.north:
            model.relative_pos.add(model.y[ex.p_room] + model.l_min[i] <= model.y[ex.n_room])
            model.relative_pos.add(model.y[ex.p_room] + model.l_max[i] >= model.y[ex.n_room])
        elif ex.direction == Direction.east:
            model.relative_pos.add(model.x[ex.p_room] + model.l_min[i] <= model.x[ex.n_room])
            model.relative_pos.add(model.x[ex.p_room] + model.l_max[i] >= model.x[ex.n_room])
        elif ex.direction == Direction.south:
            model.relative_pos.add(model.y[ex.p_room] >= model.y[ex.n_room] + model.l_min[i])
            model.relative_pos.add(model.y[ex.p_room] <= model.y[ex.n_room] + model.l_max[i])
        elif ex.direction == Direction.west:
            model.relative_pos.add(model.x[ex.p_room] >= model.x[ex.n_room] + model.l_min[i])
            model.relative_pos.add(model.x[ex.p_room] <= model.x[ex.n_room] + model.l_max[i])
        elif ex.direction == Direction.up:
            model.relative_pos.add(model.z[ex.p_room] + model.l_min[i] <= model.z[ex.n_room])
            model.relative_pos.add(model.z[ex.p_room] + model.l_max[i] >= model.z[ex.n_room])
        elif ex.direction == Direction.down:
            model.relative_pos.add(model.z[ex.p_room] >= model.z[ex.n_room] + model.l_min[i])
            model.relative_pos.add(model.z[ex.p_room] <= model.z[ex.n_room] + model.l_max[i])
        else:
            print('[!] New Direction')

        # ex crossings O(e^2)
        non_incidents = [j for j,v in enumerate(exits) if \
                         ex.p_room not in v and ex.n_room not in v]
        for non_incident in non_incidents:
            a, b = i, non_incident
            model.crossings.add(sum([model.relations[a,b,i] for i in range(Direction.mod)]) >= 1)
            # north
            model.crossings.add(model.y[exits[a].p_room] - model.y[exits[b].p_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.north]))
            model.crossings.add(model.y[exits[a].p_room] - model.y[exits[b].n_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.north]))
            model.crossings.add(model.y[exits[a].n_room] - model.y[exits[b].p_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.north]))
            model.crossings.add(model.y[exits[a].n_room] - model.y[exits[b].n_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.north]))
            # east
            model.crossings.add(model.x[exits[a].p_room] - model.x[exits[b].p_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.east]) - model.d_min)
            model.crossings.add(model.x[exits[a].p_room] - model.x[exits[b].n_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.east]) - model.d_min)
            model.crossings.add(model.x[exits[a].n_room] - model.x[exits[b].p_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.east]) - model.d_min)
            model.crossings.add(model.x[exits[a].n_room] - model.x[exits[b].n_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.east]) - model.d_min)
            # south
            model.crossings.add(model.y[exits[a].p_room] - model.y[exits[b].p_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.south]) - model.d_min)
            model.crossings.add(model.y[exits[a].p_room] - model.y[exits[b].n_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.south]) - model.d_min)
            model.crossings.add(model.y[exits[a].n_room] - model.y[exits[b].p_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.south]) - model.d_min)
            model.crossings.add(model.y[exits[a].n_room] - model.y[exits[b].n_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.south]) - model.d_min)
            # west
            model.crossings.add(model.x[exits[a].p_room] - model.x[exits[b].p_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.west]))
            model.crossings.add(model.x[exits[a].p_room] - model.x[exits[b].n_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.west]))
            model.crossings.add(model.x[exits[a].n_room] - model.x[exits[b].p_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.west]))
            model.crossings.add(model.x[exits[a].n_room] - model.x[exits[b].n_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.west]))
            # up
            model.crossings.add(model.z[exits[a].p_room] - model.z[exits[b].p_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.up]))
            model.crossings.add(model.z[exits[a].p_room] - model.z[exits[b].n_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.up]))
            model.crossings.add(model.z[exits[a].n_room] - model.z[exits[b].p_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.up]))
            model.crossings.add(model.z[exits[a].n_room] - model.z[exits[b].n_room] >= \
                                model.d_min - model.M*(1 - model.relations[a,b,Direction.up]))
            # down
            model.crossings.add(model.z[exits[a].p_room] - model.z[exits[b].p_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.down]) - model.d_min)
            model.crossings.add(model.z[exits[a].p_room] - model.z[exits[b].n_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.down]) - model.d_min)
            model.crossings.add(model.z[exits[a].n_room] - model.z[exits[b].p_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.down]) - model.d_min)
            model.crossings.add(model.z[exits[a].n_room] - model.z[exits[b].n_room] <= \
                                model.M*(1 - model.relations[a,b,Direction.down]) - model.d_min)

    SolverFactory('cbc').solve(model, tee=True)
    #model.pprint()
    return model

if __name__=='__main__':
    main()
