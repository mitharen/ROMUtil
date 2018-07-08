#!/usr/bin/env python

import sys
import os
import enum
import itertools

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
        self.desc = r[2]
        self.exits = [] if not r[2] else [Exit(e, self.vnum) for e in r[3] if e is not None]
        self.fixups = []
        self.dummy = False

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
        self.one_way = False

    def __eq__(self, e):
        if self.p_room == e.p_room and self.n_room == e.n_room and \
           self.direction == e.direction: return True
        if self.p_room == e.n_room and self.n_room == e.p_room and \
           self.direction == e.direction.invert(): return True
        return False

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
        return (2 + room.x + self.lift*room.z,
                2+self.lift*self.z_max + (self.y_max - room.y) - self.lift*room.z)

    def proj_exit(self, ex):
        if None in (self.rdb[ex.p_room].x, self.rdb[ex.p_room].y, self.rdb[ex.p_room].z): return None
        start = (2 + self.rdb[ex.p_room].x + .25 + self.lift*self.rdb[ex.p_room].z,
                 2+self.lift*self.z_max + (self.y_max - self.rdb[ex.p_room].y) + .25 - self.lift*self.rdb[ex.p_room].z)

        if ex.n_room in self.rdb.keys():
            if None in (self.rdb[ex.n_room].x, self.rdb[ex.n_room].y, self.rdb[ex.n_room].z): return None
            end = (2 + self.rdb[ex.n_room].x + .25 + self.lift*self.rdb[ex.n_room].z,
                   2+self.lift*self.z_max + (self.y_max - self.rdb[ex.n_room].y) + .25 - self.lift*self.rdb[ex.n_room].z)
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

        return (start, end)

    def __init__(self, name, rdb, exits):
        self.name = name
        self.rdb = rdb
        self.exits = exits

    def plot(self):
        self.x_max = max([r.x for r in self.rdb.values()])
        self.y_max = max([r.y for r in self.rdb.values()])
        self.z_max = max([r.y for r in self.rdb.values()])
        z_space = self.z_max*self.lift

        dwg = svgwrite.Drawing(self.name, profile='full',
                               size=((self.x_max+4+11+z_space)*cm, (self.y_max+4+4+z_space)*cm),
                               viewBox='0 0 %d %d'%(self.x_max+4+11+z_space, self.y_max+4+4+z_space))

        exits = sorted(self.exits, key=lambda x: max(self.rdb[x.p_room].z, self.rdb[x.n_room].z))
        rooms = sorted(self.rdb.values(), key=lambda r: r.z)
        descs = []

        while len(exits) or len(rooms):
            e = self.rdb[exits[0].p_room].z if len(exits) else None
            r = rooms[0].z if len(rooms) else None
            if e is not None and (r is None or r >= e):
                ex = exits.pop(0)
                projection = self.proj_exit(ex)
                color = 'red' if ex.one_way else 'black'
                dwg.add(dwg.line(start=projection[0], end=projection[1], stroke_width=.05, stroke=color))
            else:
                room = rooms.pop(0)
                if room.dummy: continue
                projection = self.proj_room(room)
                g = dwg.g(visibility='hidden')
                g.add(dwg.rect(fill='white', insert=(projection[0]+.5, projection[1]+.5), size=(11,4),
                               stroke='black', stroke_width=0.05))
                text = dwg.text('', insert=(projection[0]+.7, projection[1]+1.1), #size=(100,100),
                                font_size='.3', font_family='Arial', fill='black')
                text.add(dwg.tspan(room.name, font_size='.4'))
                etext = 'Exits: ' + ', '.join([ex.direction.name for ex in room.exits])
                for line in room.desc.split('\n')+[etext]:
                    text.add(dwg.tspan(line, x=[projection[0]+.7], dy=['1.4em']))
                g.add(text)
                r = dwg.rect(insert=projection, size=(.5,.5),
                             fill=self.colors[min(6, int(room.z))], stroke='black', stroke_width=0.025)
                s = dwg.set(to='visible')
                s.set_target('visibility')
                s.set_timing(begin=r.get_id()+'.mouseover', end=r.get_id()+'.mouseout')
                g.add(s)
                dwg.add(r)
                descs.append(g)

        # add these after so they float above
        for g in descs: dwg.add(g)

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
    # add fake looped exits for no-exit rooms to prevent overlapping placement
    for room in rdb.values():
        if not len(room.exits):
            exit = Exit((0, room.vnum), room.vnum)
            room.exits.append(exit)
            exits.append(exit)

    # construct model
    model = ConcreteModel()
    model.Rooms = Set(initialize=rdb.keys())
    model.Exits = RangeSet(0, len(exits)-1)
    model.Directions = RangeSet(0, Direction.mod.value-1)
    model.M = Param(initialize=sum([e.distance for e in exits]))
    model.d_min = Param(initialize=1)

    # room position
    model.x = Var(model.Rooms, within=Integers, bounds=(0,model.M))
    model.y = Var(model.Rooms, within=Integers, bounds=(0,model.M))
    model.z = Var(model.Rooms, within=Integers, bounds=(0,model.M))
    # exits
    model.l_max = Var(model.Exits, within=PositiveIntegers)
    model.l_min = Param(model.Exits, initialize=lambda model, x: exits[x].distance)

    # constraints for relative position of rooms
    model.relative_pos = ConstraintList()
    model.one_ways = VarList(within=NonNegativeIntegers, bounds=(0,model.M))
    model.one_way_pos = ConstraintList()
    one_ways = []

    # add constraints
    for i, ex in enumerate(exits):
        # one-ways tend to violate embedding constraints, so add them to objective and then ignore
        # mazes also break embedding constraints, so let's treat obvious ones as one-ways
        if not ex in rdb[ex.n_room].exits or \
           len([e for e in rdb[ex.p_room].exits if ex.n_room in e]) > 1:
            ex.one_way = True
            x_off = model.d_min if ex.direction == Direction.east else -model.d_min if ex.direction == Direction.west else 0
            y_off = model.d_min if ex.direction == Direction.north else -model.d_min if ex.direction == Direction.south else 0
            z_off = model.d_min if ex.direction == Direction.up else -model.d_min if ex.direction == Direction.down else 0
            X = model.one_ways.add()
            Y = model.one_ways.add()
            Z = model.one_ways.add()
            x_diff = model.x[ex.n_room] - model.x[ex.p_room] - x_off
            model.one_way_pos.add(x_diff <= X)
            model.one_way_pos.add(-x_diff <= X)
            y_diff = model.y[ex.n_room] - model.y[ex.p_room] - y_off
            model.one_way_pos.add(y_diff <= Y)
            model.one_way_pos.add(-y_diff <= Y)
            z_diff = model.z[ex.n_room] - model.z[ex.p_room] - z_off
            model.one_way_pos.add(z_diff <= Z)
            model.one_way_pos.add(-z_diff <= Z)
            one_ways.append(X+Y+Z)
            continue

        # relative position O(e)
        # loops create contradictions for relative position
        if ex.n_room != ex.p_room:
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

    # objective to minimize max exit lengths and distance of one-ways
    model.obj = Objective(expr=sum([model.l_max[e] for e in model.Exits]) + sum([way for way in one_ways]))

    print('[+] Entering solving loop...')

    # constraints for exit crossings
    # loops don't help for crossings unless they're the only exit in a room
    considered = [ex for ex in exits if ex.n_room != ex.p_room or len(rdb[ex.n_room].exits) > 1]
    non_incidents = list(filter(lambda ex: ex[0].p_room not in ex[1] and ex[0].n_room not in ex[1],
                                itertools.combinations(considered, 2)))
    print('[!] %d possible overlaps.'%(len(non_incidents)))
    model.crossings = ConstraintList()
    relations=0

    solver = SolverFactory('cbc')
    solver.options['ratio'] = .05

    while True:
        result = solver.solve(model, tee=False)
        for pair in non_incidents:
            ex, nx = pair
            if None in [model.x[ex.p_room].value, model.x[ex.n_room].value, model.x[nx.p_room].value, model.x[nx.n_room].value,
                        model.y[ex.p_room].value, model.y[ex.n_room].value, model.y[nx.p_room].value, model.y[nx.n_room].value,
                        model.z[ex.p_room].value, model.z[ex.n_room].value, model.z[nx.p_room].value, model.z[nx.n_room].value]: continue
            if max(model.x[ex.p_room].value, model.x[ex.n_room].value) < min(model.x[nx.p_room].value, model.x[nx.n_room].value) or \
               min(model.x[ex.p_room].value, model.x[ex.n_room].value) < max(model.x[nx.p_room].value, model.x[nx.n_room].value) or \
               max(model.y[ex.p_room].value, model.y[ex.n_room].value) < min(model.y[nx.p_room].value, model.y[nx.n_room].value) or \
               min(model.y[ex.p_room].value, model.y[ex.n_room].value) < max(model.y[nx.p_room].value, model.y[nx.n_room].value) or \
               max(model.z[ex.p_room].value, model.z[ex.n_room].value) < min(model.z[nx.p_room].value, model.z[nx.n_room].value) or \
               min(model.z[ex.p_room].value, model.z[ex.n_room].value) < max(model.z[nx.p_room].value, model.z[nx.n_room].value): continue

            relation = Var(model.Directions, within=Boolean)
            model.add_component('relation%d'%(relations), relation)
            relations += 1
            model.crossings.add(sum([relation[i] for i in range(Direction.mod)]) >= 1)

            # north
            model.crossings.add(model.y[ex.p_room] - model.y[nx.p_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.north]))
            model.crossings.add(model.y[ex.p_room] - model.y[nx.n_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.north]))
            model.crossings.add(model.y[ex.n_room] - model.y[nx.p_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.north]))
            model.crossings.add(model.y[ex.n_room] - model.y[nx.n_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.north]))
            # east
            model.crossings.add(model.x[ex.p_room] - model.x[nx.p_room] <= \
                                model.M*(1 - relation[Direction.east]) - model.d_min)
            model.crossings.add(model.x[ex.p_room] - model.x[nx.n_room] <= \
                                model.M*(1 - relation[Direction.east]) - model.d_min)
            model.crossings.add(model.x[ex.n_room] - model.x[nx.p_room] <= \
                                model.M*(1 - relation[Direction.east]) - model.d_min)
            model.crossings.add(model.x[ex.n_room] - model.x[nx.n_room] <= \
                                model.M*(1 - relation[Direction.east]) - model.d_min)
            # south
            model.crossings.add(model.y[ex.p_room] - model.y[nx.p_room] <= \
                                model.M*(1 - relation[Direction.south]) - model.d_min)
            model.crossings.add(model.y[ex.p_room] - model.y[nx.n_room] <= \
                                model.M*(1 - relation[Direction.south]) - model.d_min)
            model.crossings.add(model.y[ex.n_room] - model.y[nx.p_room] <= \
                                model.M*(1 - relation[Direction.south]) - model.d_min)
            model.crossings.add(model.y[ex.n_room] - model.y[nx.n_room] <= \
                                model.M*(1 - relation[Direction.south]) - model.d_min)
            # west
            model.crossings.add(model.x[ex.p_room] - model.x[nx.p_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.west]))
            model.crossings.add(model.x[ex.p_room] - model.x[nx.n_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.west]))
            model.crossings.add(model.x[ex.n_room] - model.x[nx.p_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.west]))
            model.crossings.add(model.x[ex.n_room] - model.x[nx.n_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.west]))
            # up
            model.crossings.add(model.z[ex.p_room] - model.z[nx.p_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.up]))
            model.crossings.add(model.z[ex.p_room] - model.z[nx.n_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.up]))
            model.crossings.add(model.z[ex.n_room] - model.z[nx.p_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.up]))
            model.crossings.add(model.z[ex.n_room] - model.z[nx.n_room] >= \
                                model.d_min - model.M*(1 - relation[Direction.up]))
            # down
            model.crossings.add(model.z[ex.p_room] - model.z[nx.p_room] <= \
                                model.M*(1 - relation[Direction.down]) - model.d_min)
            model.crossings.add(model.z[ex.p_room] - model.z[nx.n_room] <= \
                                model.M*(1 - relation[Direction.down]) - model.d_min)
            model.crossings.add(model.z[ex.n_room] - model.z[nx.p_room] <= \
                                model.M*(1 - relation[Direction.down]) - model.d_min)
            model.crossings.add(model.z[ex.n_room] - model.z[nx.n_room] <= \
                                model.M*(1 - relation[Direction.down]) - model.d_min)

            non_incidents.remove(pair)
            break
        else:
            break
            
    print('[!] %d/%d overlaps converted into constraints.'%(relations, len(non_incidents)))
    return model, result

def graph(rdb, name, area):
    # clean up hallways (improves performance)
    for vnum, r in list(rdb.items()):
        if len(r.exits) == 2:
            # if real, bidirectional and straight
            if not all(e.n_room in rdb.keys() for e in r.exits): continue
            if not all([e in rdb[e.n_room].exits for e in r.exits]): continue
            if r.exits[0].direction == r.exits[1].direction.invert():
                rdb[r.exits[0].n_room].replace_exit(vnum, r.exits[1].n_room, r.exits[1].distance)
                rdb[r.exits[1].n_room].replace_exit(vnum, r.exits[0].n_room, r.exits[0].distance)
                rdb[r.exits[0].n_room].fixups.append((r, r.exits[0].direction.invert(), r.exits[0].distance))
                del rdb[vnum]
                print('%s [*] Trimmed hallway %d.'%(area[1], vnum))

    # collect normal exits for solve/plotting
    exits = list(set([e for r in rdb.values() for e in r.exits]))

    # insert dummy rooms for zone exits
    for e in exits:
        if e.n_room not in rdb:
            rdb[e.n_room] = Room((e.p_room, '', None))
            rdb[e.n_room].dummy = True

    # solve
    print('%s [+] Solving for %d exits...'%(area[1], len(exits)))
    model, results = solve(rdb, exits)
    if not results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
        print('%s [-] Solver failed!%s' %(area[1], str(results.solver)))
    else:
        print('%s [+] Solve completed. Plotting...'%(area[1]))

    # retrieve room positions and restore collapsed rooms
    for vnum, room in list(rdb.items()):
        room.x = model.x[vnum].value if model.x[vnum].value else 0
        room.y = model.y[vnum].value if model.y[vnum].value else 0
        room.z = model.z[vnum].value if model.z[vnum].value else 0
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
    dwg = Plotter(name, rdb, exits)
    dwg.plot()

    return

def main():
    parser = AreaParser.Parser()
    with open(sys.argv[1], 'r') as f:
        area = parser.parse(f.read())

    for section in area:
        if section[0] == '#ROOMS':
            rooms = section[1]
        elif section[0] == '#AREA':
            area = section[1]

    # construct rooms for graphing
    rdb = {r[0]: Room(r) for r in rooms}

    # break into connected graphs
    count = 0
    graphs = []
    while len(rdb):
        stack = [rdb.popitem()[1]]
        sub_graph = {}
        while len(stack):
            r = stack.pop()
            sub_graph[r.vnum] = r
            for e in r.exits:
                if e.n_room not in sub_graph.keys():
                    if e.n_room in rdb.keys():
                        stack.append(rdb.pop(e.n_room))
                    else:
                        for g in graphs:
                            if e.n_room in g:
                                graphs.remove(g)
                                sub_graph.update(g)
        graphs.append(sub_graph)

    for g in graphs:
        name, ext = os.path.splitext(sys.argv[2])
        graph(g, name+str(count)+ext, area)
        count += 1

    exit(0)

if __name__=='__main__':
    main()
