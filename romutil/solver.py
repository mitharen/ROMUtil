import itertools
import logging
from math import sqrt
import pyomo.opt
from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Objective,
    ConstraintList, Integers, Binary, PositiveIntegers,
    NonNegativeIntegers, Boolean, VarList, SolverFactory
)
import tqdm

from romutil.models import Direction, Exit
from romutil.plotter import Plotter

log = logging.getLogger('Mapper.solver')

def non_euler(rdb, exits):
    m = ConcreteModel()

    m.Rooms = Set(initialize=rdb.keys())
    m.Exits = RangeSet(0, len(exits) - 1)
    m.Directions = RangeSet(0, Direction.mod.value - 1)

    m.M = Param(initialize=sum([e.distance for e in exits]))

    m.x = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.y = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.z = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.cut = Var(m.Exits, within=Binary)
    m.relative_pos = ConstraintList()

    one_way_exits = [e for e in exits if e not in rdb[e.dst].exits]

    for i, e in enumerate(exits):
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

    m.obj = Objective(expr=sum(m.cut[i] for i in range(len(exits))))

    solver = SolverFactory('cbc')
    solver.options['sec'] = 20
    solver.solve(m, tee=False)
    return m

def solve(rdb, area_exits):
    exits = [e for e in area_exits if e.src != e.dst]

    m = non_euler(rdb, exits)
    for i in range(len(exits)):
        if m.cut[i].value:
            exits[i].one_way = True

    for room in rdb.values():
        if not len(room.exits):
            e = Exit((0, room.vnum), room.vnum)
            exits.append(e)

    m = ConcreteModel()

    m.Rooms = Set(initialize=rdb.keys())
    m.Exits = RangeSet(0, len(exits) - 1)
    m.Directions = RangeSet(0, Direction.mod.value - 1)

    m.M = Param(initialize=sum([e.distance for e in exits]))
    m.d_min = Param(initialize=1)
    m.l_min = Param(m.Exits, initialize=lambda m, x: exits[x].distance)

    m.x = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.y = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.z = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.cut = Var(m.Exits, within=Binary)
    m.l_max = Var(m.Exits, within=PositiveIntegers)
    m.one_ways = VarList(within=NonNegativeIntegers, bounds=(0, m.M))

    m.relative_pos = ConstraintList()
    m.one_way_pos = ConstraintList()
    m.crossings = ConstraintList()

    one_ways = []

    for e in exits:
        if e not in rdb[e.dst].exits:
            e.one_way = True

    for i, x in enumerate(exits):
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
            one_ways.append(X + Y + Z)
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

    m.obj = Objective(
        expr=m.M * m.M * (sum(m.cut[i] for i in range(len(exits))))
        + sum([m.l_max[e] for e in m.Exits])
        + sum([way for way in one_ways])
    )

    non_incidents = list(itertools.combinations(range(len(exits)), 2))
    non_incidents = [
        pair for pair in non_incidents
        if exits[pair[0]].src not in exits[pair[1]] and exits[pair[0]].dst not in exits[pair[1]]
    ]
    non_incidents = [
        pair for pair in non_incidents
        if not exits[pair[0]].one_way and not exits[pair[1]].one_way
    ]

    relations = 0
    log.info(f'{len(non_incidents)} possible overlaps.')

    solver = SolverFactory('cbc', tee=False)
    solver.options['sec'] = 300

    while True:
        result = solver.solve(m, tee=False)
        if result.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
            return m, result

        for vnum, room in list(rdb.items()):
            room.x = m.x[vnum].value if m.x[vnum].value else 0
            room.y = m.y[vnum].value if m.y[vnum].value else 0
            room.z = m.z[vnum].value if m.z[vnum].value else 0
        Plotter('progress.svg', rdb, exits).plot()

        added_constraints = 0
        for left, right in tqdm.tqdm(list(non_incidents), desc='Finding Overlaps'):
            ex, nx = (exits[left], exits[right])

            if m.cut[left].value or m.cut[right].value:
                continue
            if None in [
                m.x[ex.src].value, m.x[ex.dst].value, m.x[nx.src].value, m.x[nx.dst].value,
                m.y[ex.src].value, m.y[ex.dst].value, m.y[nx.src].value, m.y[nx.dst].value,
                m.z[ex.src].value, m.z[ex.dst].value, m.z[nx.src].value, m.z[nx.dst].value
            ]:
                continue

            if (
                max(m.x[ex.src].value, m.x[ex.dst].value) < min(m.x[nx.src].value, m.x[nx.dst].value) or
                min(m.x[ex.src].value, m.x[ex.dst].value) > max(m.x[nx.src].value, m.x[nx.dst].value) or
                max(m.y[ex.src].value, m.y[ex.dst].value) < min(m.y[nx.src].value, m.y[nx.dst].value) or
                min(m.y[ex.src].value, m.y[ex.dst].value) > max(m.y[nx.src].value, m.y[nx.dst].value) or
                max(m.z[ex.src].value, m.z[ex.dst].value) < min(m.z[nx.src].value, m.z[nx.dst].value) or
                min(m.z[ex.src].value, m.z[ex.dst].value) > max(m.z[nx.src].value, m.z[nx.dst].value)
            ):
                continue

            relation = Var(m.Directions, within=Boolean)
            m.add_component(f'relation{relations}', relation)
            relations += 1
            m.crossings.add(sum([relation[j] for j in range(Direction.mod)]) >= 1)

            prod = list(itertools.product((ex.src, ex.dst), (nx.src, nx.dst)))
            for p, q in prod:
                m.crossings.add(m.x[p] - m.x[q] + m.M * ((1 - relation[Direction.east]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p, q in prod:
                m.crossings.add(m.y[p] - m.y[q] + m.M * ((1 - relation[Direction.north]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p, q in prod:
                m.crossings.add(m.z[p] - m.z[q] + m.M * ((1 - relation[Direction.up]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p, q in prod:
                m.crossings.add(m.x[q] - m.x[p] + m.M * ((1 - relation[Direction.west]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p, q in prod:
                m.crossings.add(m.y[q] - m.y[p] + m.M * ((1 - relation[Direction.south]) + m.cut[left] + m.cut[right]) >= m.d_min)
            for p, q in prod:
                m.crossings.add(m.z[q] - m.z[p] + m.M * ((1 - relation[Direction.down]) + m.cut[left] + m.cut[right]) >= m.d_min)

            non_incidents.remove((left, right))
            added_constraints += 1
            if added_constraints == int(sqrt(len(non_incidents))):
                break
        else:
            if not added_constraints:
                break

    log.debug(f'cut {sum([m.cut[i].value for i in range(len(exits))])} exits')
    log.debug(f'{relations}/{len(non_incidents)} overlaps converted into constraints.')
    return m, result
