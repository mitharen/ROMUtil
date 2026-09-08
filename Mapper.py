#!/usr/bin/env python

import argparse
import enum
import itertools
import logging
import os
import sys

import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
import pyomo.opt
import pyomo.util.infeasible
from pyomo.environ import *
import svgwrite
from svgwrite import cm
import tqdm

import AreaParser

logging.basicConfig()
log = logging.getLogger('Mapper')

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
        self.exits = [] if not r[3] else [Exit(e, self.vnum) for e in r[3] if e is not None]
        self.fixups = []
        self.dummy = False

    def replace_exit(self, orig, replacement, distance):
        for e in self.exits:
            if e.dst == orig:
                e.dst = replacement
                e.distance += distance

    def __repr__(self):
        return f'[{self.vnum}: {self.name}] {{{self.exits}}}'

class Exit():
    def __init__(self, e, source, distance=1):
        self.src = source
        self.dst = e[1]
        self.direction = Direction(direction_matrix[e[0]])
        self.distance = distance
        self.one_way = False

    def __eq__(self, e):
        if self.src == e.src and self.dst == e.dst and \
           self.direction == e.direction: return True
        if self.src == e.dst and self.dst == e.src and \
           self.direction == e.direction.invert(): return True
        return False

    def __contains__(self, room):
        return room in (self.src, self.dst)

    def __repr__(self):
        return f'{self.src} -> {self.dst} ({self.distance} {self.direction.name})'

    def __hash__(self):
        r0, r1, d = (self.src, self.dst, self.direction) if self.src < self.dst \
            else (self.dst, self.src, self.direction.invert())
        return hash(f'{r0} {r1} {d}')

class Plotter():
    lift = 0.15
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

    def proj_room(self, room):
        if None in (room.x, room.y, room.z): return None
        return (2 + room.x + self.lift*room.z,
                2+self.lift*self.z_max + (self.y_max - room.y) - self.lift*room.z)

    def proj_exit(self, ex):
        if None in (self.rdb[ex.src].x, self.rdb[ex.src].y, self.rdb[ex.src].z): return None
        start = (2 + self.rdb[ex.src].x + .25 + self.lift*self.rdb[ex.src].z,
                 2+self.lift*self.z_max + (self.y_max - self.rdb[ex.src].y) + .25 - self.lift*self.rdb[ex.src].z)

        if ex.dst in self.rdb.keys():
            if None in (self.rdb[ex.dst].x, self.rdb[ex.dst].y, self.rdb[ex.dst].z): return None
            end = (2 + self.rdb[ex.dst].x + .25 + self.lift*self.rdb[ex.dst].z,
                   2+self.lift*self.z_max + (self.y_max - self.rdb[ex.dst].y) + .25 - self.lift*self.rdb[ex.dst].z)
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
                               viewBox=f'0 0 {self.x_max+4+11+z_space} {self.y_max+4+4+z_space}')

        exits = sorted(self.exits, key=lambda x: max(self.rdb[x.src].z, self.rdb[x.dst].z))
        rooms = sorted(self.rdb.values(), key=lambda r: r.z)
        descs = []

        while len(exits) or len(rooms):
            e = self.rdb[exits[0].src].z if len(exits) else None
            r = rooms[0].z if len(rooms) else None
            #
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
                etext = 'Exits: ' + ', '.join([ex.direction.name for ex in room.exits])
                desc = room.desc.split('\n')+[etext]
                g.add(dwg.rect(fill='white', insert=(projection[0]+.5, projection[1]+.5), size=(11,2+len(desc)/3),
                               stroke='black', stroke_width=0.05))
                text = dwg.text('', insert=(projection[0]+.7, projection[1]+1.1), #size=(100,100),
                                font_size='.3', font_family='Arial', fill='black')
                text.add(dwg.tspan(room.name, font_size='.4'))
                for line in desc:
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
        if d == Direction.north: r.y += dist
        elif d == Direction.east: r.x += dist
        elif d == Direction.south: r.y -= dist
        elif d == Direction.west: r.x -= dist
        elif d == Direction.up: r.z += dist
        elif d == Direction.down: r.z -= dist
        rooms.append(r)
        rooms += restore_rooms(r)
    return rooms

def mfas(edges):
    '''
    minimum feedback arc set for a graph
    '''
    graph = nx.DiGraph(edges)

    labels = {u:v for u,v in zip(graph.nodes, range(graph.order()))}
    nx.relabel.relabel_nodes(graph, labels, copy=False)
    nx.drawing.nx_pydot.to_pydot(graph).write_svg('test.svg')

    model = ConcreteModel()
    model.Nodes = RangeSet(0, graph.order())

    model.N = Param(initialize=graph.order())

    model.b = Var(model.Nodes, model.Nodes, within=Binary)
    model.p = Var(model.Nodes, within=NonNegativeIntegers, bounds=(0,model.N))
    model.order = ConstraintList()

    # each node must be ordered per the edges unless excluded
    for u,v in graph.edges:
        model.order.add((model.p[v] - model.p[u]) + (model.N * model.b[u,v]) >= 1)

    model.obj = Objective(expr=sum([model.b[u,v] for u,v in graph.edges]))

    solver = SolverFactory('cbc')
    #solver.options['ratio'] = .05
    solver.options['threads'] = 8
    result = solver.solve(model, tee=False)
    return [(u,v) for u,v in edges if model.b[labels[u],labels[v]].value]

def non_euler(rdb, exits):
    # construct model
    m = ConcreteModel()

    # statics
    m.Rooms = Set(initialize=rdb.keys())
    m.Exits = RangeSet(0, len(exits)-1)
    m.Directions = RangeSet(0, Direction.mod.value-1)

    # maximum value (infinity)
    m.M = Param(initialize=sum([e.distance for e in exits]))

    # room position (constraints will determine these)
    m.x = Var(m.Rooms, within=Integers, bounds=(0,m.M))
    m.y = Var(m.Rooms, within=Integers, bounds=(0,m.M))
    m.z = Var(m.Rooms, within=Integers, bounds=(0,m.M))
    # allow cutting exits to handle mazes (objective will minimize this)
    m.cut = Var(m.Exits, within=Binary)
    # constraints for relative position of rooms
    m.relative_pos = ConstraintList()

    # one-ways are often non-euclidean should be cut
    one_way_exits = [e for e in exits if not e in rdb[e.dst].exits]

    # add constraints for relative position excepting cut exits - O(e)
    for i, e in enumerate(exits):
        # ignore one-ways
        if e in one_way_exits:
            continue

        if e.dst != e.src:
            if e.direction not in (Direction.east, Direction.west):
                m.relative_pos.add(m.x[e.src] - m.x[e.dst] + m.M * m.cut[i] >= 0)
                m.relative_pos.add(m.x[e.dst] - m.x[e.src] + m.M * m.cut[i] >= 0)
            if e.direction not in (Direction.north, Direction.south):
                m.relative_pos.add(m.y[e.src] - m.y[e.dst] + m.M * m.cut[i] >= 0)
                m.relative_pos.add(m.y[e.dst] - m.y[e.src] + m.M * m.cut[i] >= 0)
            if e.direction not in (Direction.up, Direction.down):
                m.relative_pos.add(m.z[e.src] - m.z[e.dst] + m.M * m.cut[i] >= 0)
                m.relative_pos.add(m.z[e.dst] - m.z[e.src] + m.M * m.cut[i] >= 0)

            if e.direction == Direction.east:
                m.relative_pos.add(m.x[e.dst] - m.x[e.src] + m.M * m.cut[i] >= 1)
            elif e.direction == Direction.north:
                m.relative_pos.add(m.y[e.dst] - m.y[e.src] + m.M * m.cut[i] >= 1)
            elif e.direction == Direction.up:
                m.relative_pos.add(m.z[e.dst] - m.z[e.src] + m.M * m.cut[i] >= 1)
            elif e.direction == Direction.west:
                m.relative_pos.add(m.x[e.src] - m.x[e.dst] + m.M * m.cut[i] >= 1)
            elif e.direction == Direction.south:
                m.relative_pos.add(m.y[e.src] - m.y[e.dst] + m.M * m.cut[i] >= 1)
            elif e.direction == Direction.down:
                m.relative_pos.add(m.z[e.src] - m.z[e.dst] + m.M * m.cut[i] >= 1)

    # objective to minimize number of cut exits
    m.obj = Objective(expr=sum(m.cut[i] for i in range(len(exits))))

    solver = SolverFactory('cbc')
    solver.options['sec'] = 20
    result = solver.solve(m, tee=False)
    return m

def solve(rdb, area_exits):
    # remove loop exits since they're unhelpful in placement
    exits = [e for e in area_exits if e.src != e.dst]

    m = non_euler(rdb, exits)
    for i in range(len(exits)):
        if m.cut[i].value: exits[i].one_way = True

    # add a fake looped exit for no-exit rooms to prevent overlap
    for room in rdb.values():
        if not len(room.exits):
            e = Exit((0, room.vnum), room.vnum)
            exits.append(e)

    # construct model
    m = ConcreteModel()

    # statics
    m.Rooms = Set(initialize=rdb.keys())
    m.Exits = RangeSet(0, len(exits)-1)
    m.Directions = RangeSet(0, Direction.mod.value-1)

    # maximum value (infinity)
    m.M = Param(initialize=sum([e.distance for e in exits]))
    # minimum spacing between rooms
    m.d_min = Param(initialize=1)
    # l_min is the minimum length of the exit to account for hallways
    m.l_min = Param(m.Exits, initialize=lambda m, x: exits[x].distance)

    # room position (constraints will determine these)
    m.x = Var(m.Rooms, within=Integers, bounds=(0,m.M))
    m.y = Var(m.Rooms, within=Integers, bounds=(0,m.M))
    m.z = Var(m.Rooms, within=Integers, bounds=(0,m.M))
    # allow cutting exits to handle mazes (objective will minimize this)
    m.cut = Var(m.Exits, within=Binary)
    # l_max is the maximum length of each exit (objective will minimize this)
    m.l_max = Var(m.Exits, within=PositiveIntegers)
    # variables to account for one-way exit length
    m.one_ways = VarList(within=NonNegativeIntegers, bounds=(0,m.M))

    # constraints for relative position of rooms
    m.relative_pos = ConstraintList()
    # constraints for one-way positioning
    m.one_way_pos = ConstraintList()
    # constraints for exit crossings
    m.crossings = ConstraintList()

    # bin for objective
    one_ways = []

    # one-ways are often non-euclidean and would be cut, but we want to keep the rooms close together
    for e in exits:
        if not e in rdb[e.dst].exits: e.one_way = True

    # add constraints for relative position excepting cut exits - O(e)
    for i, x in enumerate(exits):
        # we add one-ways to the objective to try to keep the ends close
        if x.one_way:
            x_off = m.d_min if x.direction == Direction.east else -m.d_min if x.direction == Direction.west else 0
            y_off = m.d_min if x.direction == Direction.north else -m.d_min if x.direction == Direction.south else 0
            z_off = m.d_min if x.direction == Direction.up else -m.d_min if x.direction == Direction.down else 0
            X = m.one_ways.add()
            Y = m.one_ways.add()
            Z = m.one_ways.add()
            x_diff = m.x[x.dst] - m.x[x.src] - x_off
            m.one_way_pos.add(x_diff <= X)
            m.one_way_pos.add(-x_diff <= X)
            y_diff = m.y[x.dst] - m.y[x.src] - y_off
            m.one_way_pos.add(y_diff <= Y)
            m.one_way_pos.add(-y_diff <= Y)
            z_diff = m.z[x.dst] - m.z[x.src] - z_off
            m.one_way_pos.add(z_diff <= Z)
            m.one_way_pos.add(-z_diff <= Z)
            one_ways.append(X+Y+Z)
            continue

        if x.dst != x.src:
            if x.direction not in (Direction.east, Direction.west):
                m.relative_pos.add(m.x[x.src] + m.M * m.cut[i] >= m.x[x.dst])
                m.relative_pos.add(m.x[x.dst] + m.M * m.cut[i] >= m.x[x.src])
            if x.direction not in (Direction.north, Direction.south):
                m.relative_pos.add(m.y[x.src] + m.M * m.cut[i] >= m.y[x.dst])
                m.relative_pos.add(m.y[x.dst] + m.M * m.cut[i] >= m.y[x.src])
            if x.direction not in (Direction.up, Direction.down):
                m.relative_pos.add(m.z[x.src] + m.M * m.cut[i] >= m.z[x.dst])
                m.relative_pos.add(m.z[x.dst] + m.M * m.cut[i] >= m.z[x.src])

            if x.direction == Direction.east:
                m.relative_pos.add(m.x[x.dst] - m.x[x.src] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.x[x.dst] - m.x[x.src] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.north:
                m.relative_pos.add(m.y[x.dst] - m.y[x.src] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.y[x.dst] - m.y[x.src] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.up:
                m.relative_pos.add(m.z[x.dst] - m.z[x.src] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.z[x.dst] - m.z[x.src] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.west:
                m.relative_pos.add(m.x[x.src] - m.x[x.dst] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.x[x.src] - m.x[x.dst] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.south:
                m.relative_pos.add(m.y[x.src] - m.y[x.dst] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.y[x.src] - m.y[x.dst] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.down:
                m.relative_pos.add(m.z[x.src] - m.z[x.dst] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.z[x.src] - m.z[x.dst] - m.M * m.cut[i] <= m.l_max[i])

    # objective to minimize number of cut exits (first), exit lengths, and one-way lengths
    m.obj = Objective(expr=m.M*m.M*(sum(m.cut[i] for i in range(len(exits))))
                      + sum([m.l_max[e] for e in m.Exits])
                      + sum([way for way in one_ways]))

    non_incidents = itertools.combinations(range(len(exits)), 2)
    # filter exit pairs that share a room
    non_incidents = filter(lambda pair: exits[pair[0]].src not in exits[pair[1]] and exits[pair[0]].dst not in exits[pair[1]], non_incidents)
    # filter one-ways since they're likely make the problem significantly more complex
    non_incidents = list(filter(lambda pair: not exits[pair[0]].one_way and not exits[pair[1]].one_way, non_incidents))
    relations=0
    log.info(f'{len(non_incidents)} possible overlaps.')

    solver = SolverFactory('cbc', tee=False)
    solver.options['sec'] = 300

    # iteratively solve by progressively adding more constraints (O(e^2))
    while True:
        result = solver.solve(m, tee=False)
        if result.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
#            pyomo.util.infeasible.log_infeasible_constraints(m, log_expression=True)
            return m, result
        # plot intermediate solve
        for vnum, room in list(rdb.items()):
            room.x = m.x[vnum].value if m.x[vnum].value else 0
            room.y = m.y[vnum].value if m.y[vnum].value else 0
            room.z = m.z[vnum].value if m.z[vnum].value else 0
        Plotter('progress.svg', rdb, exits).plot()
        # add number of constraints equal to the square root of the possible overlaps
        added_constraints = 0
        # examine exit pairs and add constraints if intersecting
        for left, right in tqdm.tqdm(non_incidents, desc='Finding Overlaps'):
            ex, nx = (exits[left], exits[right])

            # ignore if either exit has been cut
            if m.cut[left].value or m.cut[right].value: continue
            # ignore if either room doesn't have a concrete position
            if None in [m.x[ex.src].value, m.x[ex.dst].value, m.x[nx.src].value, m.x[nx.dst].value,
                        m.y[ex.src].value, m.y[ex.dst].value, m.y[nx.src].value, m.y[nx.dst].value,
                        m.z[ex.src].value, m.z[ex.dst].value, m.z[nx.src].value, m.z[nx.dst].value]: continue
            # ignore if not intersecting
            if max(m.x[ex.src].value, m.x[ex.dst].value) < min(m.x[nx.src].value, m.x[nx.dst].value) or \
               min(m.x[ex.src].value, m.x[ex.dst].value) > max(m.x[nx.src].value, m.x[nx.dst].value) or \
               max(m.y[ex.src].value, m.y[ex.dst].value) < min(m.y[nx.src].value, m.y[nx.dst].value) or \
               min(m.y[ex.src].value, m.y[ex.dst].value) > max(m.y[nx.src].value, m.y[nx.dst].value) or \
               max(m.z[ex.src].value, m.z[ex.dst].value) < min(m.z[nx.src].value, m.z[nx.dst].value) or \
               min(m.z[ex.src].value, m.z[ex.dst].value) > max(m.z[nx.src].value, m.z[nx.dst].value): continue

            log.debug(f'Found intersection:\n\t\t{rdb[ex.src]}\n\t\tto\n\t\t{rdb[ex.dst]}\n\tand\n\t\t{rdb[nx.src]}\n\t\tto\n\t\t{rdb[nx.dst]}\ncoordinates: [({m.x[ex.src].value}, {m.y[ex.src].value}, {m.z[ex.src].value}) to ({m.x[ex.dst].value}, {m.y[ex.dst].value}, {m.z[ex.dst].value})] and [({m.x[nx.src].value}, {m.y[nx.src].value}, {m.z[nx.src].value}) to ({m.x[nx.dst].value}, {m.y[nx.dst].value}, {m.z[nx.dst].value})]')

            # pick at least one direction to enforce a set of non-intersection inequalities
            relation = Var(m.Directions, within=Boolean)
            m.add_component(f'relation{relations}', relation)
            relations += 1
            m.crossings.add(sum([relation[i] for i in range(Direction.mod)]) >= 1)

            # each set of inequalities is enough to guarantee that the first exit line falls outside the second
            prod = list(itertools.product((ex.src, ex.dst), (nx.src, nx.dst)))
            for p,q in prod: m.crossings.add(m.x[p] - m.x[q] + m.M*((1 - relation[Direction.east]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p,q in prod: m.crossings.add(m.y[p] - m.y[q] + m.M*((1 - relation[Direction.north]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p,q in prod: m.crossings.add(m.z[p] - m.z[q] + m.M*((1 - relation[Direction.up]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p,q in prod: m.crossings.add(m.x[q] - m.x[p] + m.M*((1 - relation[Direction.west]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p,q in prod: m.crossings.add(m.y[q] - m.y[p] + m.M*((1 - relation[Direction.south]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p,q in prod: m.crossings.add(m.z[q] - m.z[p] + m.M*((1 - relation[Direction.down]) + m.cut[left] + m.cut[right]) >= m.d_min)

            non_incidents.remove((left,right))
            added_constraints += 1
            log.debug(f'found {added_constraints}/{int(sqrt(len(non_incidents)))} new constraints')
            if added_constraints == int(sqrt(len(non_incidents))): break
        else:
            if not added_constraints: break

    log.debug(f'cut {sum([m.cut[i].value for i in range(len(exits))])} exits')
    log.debug(f'{relations}/{len(non_incidents)} overlaps converted into constraints.')
    return m, result

def graph(rdb, name, area):
    # collapse hallways (improves performance)
    for vnum, r in list(rdb.items()):
        if len(r.exits) == 2:
            # if real, bidirectional and straight
            if not all(e.dst in rdb.keys() for e in r.exits): continue
            if not all([e in rdb[e.dst].exits for e in r.exits]): continue
            if r.exits[0].direction == r.exits[1].direction.invert():
                rdb[r.exits[0].dst].replace_exit(vnum, r.exits[1].dst, r.exits[1].distance)
                rdb[r.exits[1].dst].replace_exit(vnum, r.exits[0].dst, r.exits[0].distance)
                rdb[r.exits[0].dst].fixups.append((r, r.exits[0].direction.invert(), r.exits[0].distance))
                del rdb[vnum]
                log.debug(f'{area[1]} Trimmed hallway {vnum}.')

    # collect normal exits for solve/plotting
    exits = list(set([e for r in rdb.values() for e in r.exits]))

    # fix undefined exits
    for e in exits:
        # incomplete areas should just make new dummy rooms and fall through
        if e.dst == -1:
            e.dst = max(rdb.keys()) + 1
        # insert dummy rooms for zone exits
        if e.dst not in rdb:
            rdb[e.dst] = Room((e.dst, '', '', []))
            rdb[e.dst].exits.append(e)
            rdb[e.dst].dummy = True

    if not len(exits):
        log.warning(f'Ignoring disconnected room: {rdb.popitem()}')
        return

    # solve
    log.info(f'{area[1]} Solving for {len(exits)} exits...')
    model, results = solve(rdb, exits)
    if not results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
        log.error(f'{area[1]} Solver failed!')
        log.debug(f'{str(results.solver)}')
        return
    else:
        log.info(f'{area[1]} Solve completed. Plotting...')

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

def main(area_files, outbase):
    rdb = {}
    for area_file in area_files:
        parser = AreaParser.Parser()
        try:
            area = parser.parse(area_file.read())
        except Exception as e:
            log.error(e)
            continue

        rooms = []
        for section in area:
            if section[0] == '#ROOMS':
                rooms = section[1]
            elif section[0] == '#AREA':
                area = section[1]

        # construct rooms for graphing
        rdb.update({r[0]: Room(r) for r in rooms})

    if not rdb:
        log.error('No rooms to plot.')
        exit(0)

    # make a graph from the exits
    edges = [(e.src, e.dst)
             for r in rdb.values()
             for e in r.exits
             if e.src in rdb.keys() and e.dst in rdb.keys()]
    g = nx.DiGraph(edges)

    # graph each connected graph separately
    for i, sub_graph in enumerate(nx.connected_components(g.to_undirected())):
        graph({node: rdb[node] for node in sub_graph}, f'{outbase}{i}.svg', area)

    exit(0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('areas', nargs='+', type=argparse.FileType('r'), help='.ARE file for parsing')
    parser.add_argument('-outbase', help='output base name')
    parser.add_argument('-d', '--debug', action='store_true', help="Show debug info")
    args = parser.parse_args()

    pyomo.common.config.logger.setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)

    outbase = args.outbase
    if not outbase:
        outbase, _ = os.path.splitext(args.areas[0].name)

    main(args.areas, outbase)
