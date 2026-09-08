import logging
import networkx as nx
import pyomo.opt
from pyomo.environ import ConcreteModel, RangeSet, Param, Var, Objective, ConstraintList, Binary, NonNegativeIntegers, SolverFactory

from romutil.models import Direction, Room, Exit, RoomDef
from romutil.plotter import Plotter
from romutil.solver import solve

log = logging.getLogger('Mapper.graph')

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

def mfas(edges):
    '''
    minimum feedback arc set for a graph
    '''
    graph_obj = nx.DiGraph(edges)
    labels = {u: v for u, v in zip(graph_obj.nodes, range(graph_obj.order()))}
    nx.relabel.relabel_nodes(graph_obj, labels, copy=False)
    nx.drawing.nx_pydot.to_pydot(graph_obj).write_svg('test.svg')

    model = ConcreteModel()
    model.Nodes = RangeSet(0, graph_obj.order())
    model.N = Param(initialize=graph_obj.order())
    model.b = Var(model.Nodes, model.Nodes, within=Binary)
    model.p = Var(model.Nodes, within=NonNegativeIntegers, bounds=(0, model.N))
    model.order = ConstraintList()

    for u, v in graph_obj.edges:
        model.order.add((model.p[v] - model.p[u]) + (model.N * model.b[u, v]) >= 1)

    model.obj = Objective(expr=sum([model.b[u, v] for u, v in graph_obj.edges]))

    solver = SolverFactory('cbc')
    solver.options['threads'] = 8
    solver.solve(model, tee=False)
    return [(u, v) for u, v in edges if model.b[labels[u], labels[v]].value]

def graph(rdb, name, area):
    area_name = area.name if hasattr(area, 'name') else (area[1] if isinstance(area, (list, tuple)) and len(area) > 1 else str(area or ''))
    # collapse straight bidirectional hallways
    for vnum, r in list(rdb.items()):
        if len(r.exits) == 2:
            if not all(e.dst in rdb.keys() for e in r.exits):
                continue
            if not all([e in rdb[e.dst].exits for e in r.exits]):
                continue
            if r.exits[0].direction == r.exits[1].direction.invert():
                rdb[r.exits[0].dst].replace_exit(vnum, r.exits[1].dst, r.exits[1].distance)
                rdb[r.exits[1].dst].replace_exit(vnum, r.exits[0].dst, r.exits[0].distance)
                rdb[r.exits[0].dst].fixups.append((r, r.exits[0].direction.invert(), r.exits[0].distance))
                del rdb[vnum]
                log.debug(f'{area_name} Trimmed hallway {vnum}.')

    exits = list(set([e for r in rdb.values() for e in r.exits]))

    for e in exits:
        if e.dst == -1:
            e.dst = max(rdb.keys()) + 1
        if e.dst not in rdb:
            rdb[e.dst] = Room(RoomDef(vnum=e.dst, name='', description='', exits=()))
            rdb[e.dst].exits.append(e)
            rdb[e.dst].dummy = True

    if not len(exits):
        log.warning(f'Ignoring disconnected room: {rdb.popitem()}')
        return

    log.info(f'{area_name} Solving for {len(exits)} exits...')
    model, results = solve(rdb, exits)
    if not results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
        log.error(f'{area_name} Solver failed!')
        log.debug(f'{str(results.solver)}')
        return
    else:
        log.info(f'{area_name} Solve completed. Plotting...')

    for vnum, room in list(rdb.items()):
        room.x = model.x[vnum].value if model.x[vnum].value else 0
        room.y = model.y[vnum].value if model.y[vnum].value else 0
        room.z = model.z[vnum].value if model.z[vnum].value else 0
        for r in restore_rooms(room):
            rdb[r.vnum] = r

    x_min = min([r.x for r in rdb.values()])
    y_min = min([r.y for r in rdb.values()])
    z_min = min([r.z for r in rdb.values()])
    for r in rdb.values():
        r.x -= x_min
        r.y -= y_min
        r.z -= z_min

    dwg = Plotter(name, rdb, exits)
    dwg.plot()
