#!/usr/bin/env python

import sys
import enum

import svgwrite
from svgwrite import cm
import pyomo.opt
from pyomo.environ import *

import AreaParser

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

    def __hash__(self):
        r0, r1, d = (self.p_room, self.n_room, self.direction) if self.p_room < self.n_room \
            else (self.n_room, self.p_room, self.direction.invert())
        return hash('%d %d %d'%(r0, r1, d))

class Plotter():
    lift = 0.15
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

    def proj_room(self, room):
        if None in (room.x, room.y, room.z): return None
        return ((2 + room.x + self.lift*room.z,
                 3 + (self.y_max - room.y) - self.lift*room.z),
                room.z)

    def proj_exit(self, rdb, ex):
        if None in (rdb[ex.p_room].x, rdb[ex.p_room].y, rdb[ex.p_room].z): return None
        start = (2 + rdb[ex.p_room].x + .25 + self.lift*rdb[ex.p_room].z,
                 3 + (self.y_max - rdb[ex.p_room].y) + .25 - self.lift*rdb[ex.p_room].z)

        if ex.n_room in rdb.keys():
            if None in (rdb[ex.n_room].x, rdb[ex.n_room].y, rdb[ex.n_room].z): return None
            end = (2 + rdb[ex.n_room].x + .25 + self.lift*rdb[ex.n_room].z,
                   3 + (self.y_max - rdb[ex.n_room].y) + .25 - self.lift*rdb[ex.n_room].z)
        else:
            if ex.direction == Direction.north:
                end = (start[0], start[1]-1)
            elif ex.direction == Direction.east:
                end = (start[0]+1, start[1])
            elif ex.direction == Direction.south:
                end = (start[0], start[1]+1)
            elif ex.direction == Direction.west:
                end = (start[0]-1, start[1])
            elif ex.direction == Direction.up:
                end = (start[0]+self.lift, start[1]-self.lift)
            elif ex.direction == Direction.down:
                end = (start[0]-self.lift, start[1]+self.lift)

        return (start, end, rdb[ex.p_room].z)

    def __init__(self, name, rdb, exits):
        self.name = name
        self.rdb = rdb
        self.exits = exits

    def plot(self):
        self.x_max = max([r.x for r in self.rdb.values()])
        self.y_max = max([r.y for r in self.rdb.values()])
    
        dwg = svgwrite.Drawing(self.name, profile='tiny',
                               size=((self.x_max+4)*cm, (self.y_max+6)*cm),
                               viewBox='0 0 %d %d'%(self.x_max+4, self.y_max+6))

        exits = []
        rooms = []
        for ex in self.exits:
            projection = self.proj_exit(self.rdb, ex)
            if not projection: continue
            exits.append((projection[2], projection[0], projection[1]))
        for room in self.rdb.values():
            projection = self.proj_room(room)
            if not projection: continue
            rooms.append((projection[1], projection[0]))
        exits.sort(key=lambda x: x[0])
        rooms.sort(key=lambda x: x[0])

        while len(exits) or len(rooms):
            e = exits[0][0] if len(exits) else None
            r = rooms[0][0] if len(rooms) else None
            if e is not None and (r is None or r >= e):
                dimen = exits.pop(0)
                dwg.add(dwg.line(start=dimen[1], end=dimen[2], stroke_width=.05, stroke='black'))
            else:
                dimen = rooms.pop(0)
                dwg.add(dwg.rect(insert=dimen[1], size=(.5,.5), fill=self.colors[min(6, int(dimen[0]))], stroke='black', stroke_width=0.025))
        dwg.save()
        return

def restore_rooms(room):
    rooms = []
    for r, d, dist in room.fixups:
        r.x, r.y, r.z = room.x, room.y, room.z
        if d == Direction.north:
            r.y += dist
        elif d == Direction.east:
            r.x += dist
        elif d == Direction.south:
            r.y -= dist
        elif d == Direction.west:
            r.x -= dist
        elif d == Direction.up:
            r.z += dist
        elif d == Direction.down:
            r.z -= dist
        rooms.append(r)
        rooms += restore_rooms(r)
    return rooms

def solve(rdb, exits):
    # construct model
    model = ConcreteModel()
    model.Rooms = Set(initialize=rdb.keys())
    model.Exits = RangeSet(0, len(exits)-1)
    model.Directions = RangeSet(0, Direction.mod.value-1)
    model.M = Param(initialize=sum([e.distance for e in exits]))

    # room position
    model.x = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    model.y = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    model.z = Var(model.Rooms, within=Integers, bounds=(0,len(exits)))
    # exits
    model.l_max = Var(model.Exits, within=PositiveIntegers)
    model.l_min = Param(model.Exits, initialize=lambda model, x: exits[x].distance)

    # constraints for exit crossings
    model.d_min = Param(initialize=1)
    model.crossings = ConstraintList()
    # constraints for relative position of rooms
    model.relations = Var(model.Exits, model.Exits, model.Directions, within=Boolean)
    model.relative_pos = ConstraintList()

    # add constraints
    for i, ex in enumerate(exits):
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

    # objective to minimize max exit lengths
    model.obj = Objective(expr=sum([model.l_max[e] for e in model.Exits]))

    solver = SolverFactory('cbc')
    solver.options['ratio'] = .05
    return model, solver.solve(model, tee=False)

def main():
    parser = AreaParser.Parser()
    with open(sys.argv[1], 'r') as f:
        area = parser.parse(f.read())

    for section in area:
        if section[0] == '#ROOMS':
            rooms = section[1]
        elif section[0] == '#AREA':
            area = section[1]

    # construct rooms for manipulation
    rdb = {r[0]: Room(r) for r in rooms}
    odd_exits = []

    for vnum, r in list(rdb.items()):
        # remove exits to other areas and one-ways (they tend to violate embedding constraints)
        for e in list(r.exits):
            if e.n_room not in rdb.keys():
                odd_exits.append(e)
                r.exits.remove(e)
            elif not any(vnum == e0.n_room and e0.direction == e.direction.invert() \
                         for e0 in rdb[e.n_room].exits):
                print('%s [*] Removed one-way or zone exit at %d.'%(area[1], vnum))
                odd_exits.append(e)
                r.exits.remove(e)

        # clean up potential hallways
        if len(r.exits) == 2:
            # if bidirectional and straight
            if not all(vnum in [e1.n_room for e1 in rdb[e0.n_room].exits] for e0 in r.exits): continue
            if r.exits[0].direction == r.exits[1].direction.invert():
                rdb[r.exits[0].n_room].replace_exit(vnum, r.exits[1].n_room, r.exits[1].distance)
                rdb[r.exits[1].n_room].replace_exit(vnum, r.exits[0].n_room, r.exits[0].distance)
                rdb[r.exits[0].n_room].fixups.append((r, r.exits[0].direction.invert(), r.exits[0].distance))
                del rdb[vnum]
                print('%s [*] Trimmed hallway %d.'%(area[1], vnum))

    # collect normal exits for solve/plotting
    exits = list(set([e for r in rdb.values() for e in r.exits]))

    # solve
    print('%s [+] Solving for %d exits...'%(area[1], len(exits)))
    model, results = solve(rdb, exits)
    if not results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
        print('%s [-] Solver failed!%s' %(area[1], str(results.solver)))
        exit(0)
    else:
        print('%s [+] Solve completed. Plotting...'%(area[1]))
    # model.display()

    # retrieve room positions and restore collapsed rooms
    for vnum, room in list(rdb.items()):
        room.x = model.x[vnum].value
        room.y = model.y[vnum].value
        room.z = model.z[vnum].value
        for r in restore_rooms(room):
            rdb[r.vnum] = r

    # shift room base to (0,0,0) 
    x_min = min([r.x for r in rdb.values()])
    y_min = min([r.y for r in rdb.values()])
    z_min = min([r.z for r in rdb.values()])
    for r in rdb.values():
        r.x -= x_min
        r.y -= y_min
        r.z -= z_min

    # plot
    dwg = Plotter(sys.argv[2], rdb, exits+odd_exits)
    dwg.plot()

    exit(0)

if __name__=='__main__':
    main()
