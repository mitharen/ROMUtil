import logging
import networkx as nx
import pyomo.opt
from pyomo.environ import ConcreteModel, RangeSet, Param, Var, Objective, ConstraintList, Binary, NonNegativeIntegers, SolverFactory

from romutil.models import Direction, Room, Exit, RoomDef
from romutil.plotter import Plotter
from romutil.renderers import SVGRenderer, render_map
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

def _has_feasible_coordinates(model, rdb) -> bool:
    """Check if model has valid coordinates populated for all rooms in rdb."""
    if model is None:
        return False
    if not hasattr(model, 'x') or not hasattr(model, 'y') or not hasattr(model, 'z'):
        return False
    try:
        return bool(rdb) and all(
            v in model.x and model.x[v].value is not None
            and v in model.y and model.y[v].value is not None
            and v in model.z and model.z[v].value is not None
            for v in rdb
        )
    except (KeyError, AttributeError, TypeError):
        return False

def solve_layout(rdb, area=None, solver_timeout=None):
    """
    Solves 3D coordinates for rooms using Pyomo MILP optimization.
    Simplifies straight hallways, resolves boundary dummies, restores hallways,
    normalizes coordinates so minimums are zero, restores original exits on rooms,
    and calculates one-way exit statuses.
    Returns (rdb, exits).
    """
    import copy
    area_name = area.name if hasattr(area, 'name') else (area[1] if isinstance(area, (list, tuple)) and len(area) > 1 else str(area or 'Area'))

    # Save original exits for complete export preservation
    original_exits = {vnum: copy.deepcopy(r.exits) for vnum, r in rdb.items()}

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
        log.warning(f'Ignoring disconnected room: {list(rdb.keys())}')
        for r in rdb.values():
            if r.x is None: r.x = 0
            if r.y is None: r.y = 0
            if r.z is None: r.z = 0
        return rdb, exits

    log.info(f'{area_name} Solving for {len(exits)} exits...')
    if solver_timeout is not None:
        model, results = solve(rdb, exits, timeout=solver_timeout)
    else:
        model, results = solve(rdb, exits)

    tc = getattr(getattr(results, 'solver', None), 'termination_condition', None)
    has_valid_coords = _has_feasible_coordinates(model, rdb)

    is_optimal = (tc == pyomo.opt.TerminationCondition.optimal)
    is_time_limit_or_feasible = (
        tc in (
            pyomo.opt.TerminationCondition.maxTimeLimit,
            pyomo.opt.TerminationCondition.feasible,
        )
    )

    if is_optimal and has_valid_coords:
        log.info(f'{area_name} Solve completed.')
    elif is_time_limit_or_feasible and has_valid_coords:
        log.warning(f'{area_name} Solver reached time limit; using best feasible layout.')
    else:
        log.error(f'{area_name} Solver failed!')
        if results and hasattr(results, 'solver'):
            log.debug(f'{str(results.solver)}')
        for r in rdb.values():
            if r.x is None: r.x = 0
            if r.y is None: r.y = 0
            if r.z is None: r.z = 0
        return rdb, exits

    for vnum, room in list(rdb.items()):
        room.x = model.x[vnum].value if model.x[vnum].value is not None else 0
        room.y = model.y[vnum].value if model.y[vnum].value is not None else 0
        room.z = model.z[vnum].value if model.z[vnum].value is not None else 0
        for r in restore_rooms(room):
            rdb[r.vnum] = r

    valid_rooms = [r for r in rdb.values() if r.x is not None and r.y is not None and r.z is not None]
    if valid_rooms:
        x_min = min([r.x for r in valid_rooms])
        y_min = min([r.y for r in valid_rooms])
        z_min = min([r.z for r in valid_rooms])
        for r in valid_rooms:
            r.x -= x_min
            r.y -= y_min
            r.z -= z_min

    # Restore original exits for all non-dummy rooms
    for vnum, orig_ex in original_exits.items():
        if vnum in rdb:
            rdb[vnum].exits = orig_ex

    # Compute one_way status on all exits
    for r in rdb.values():
        if r.dummy:
            continue
        for e in r.exits:
            has_return = False
            if e.dst in rdb and not rdb[e.dst].dummy:
                inv = e.direction.invert()
                for re in rdb[e.dst].exits:
                    if re.dst == r.vnum and re.direction == inv:
                        has_return = True
                        break
            e.one_way = not has_return

    return rdb, exits


def graph(rdb, name, area, split_levels=False, outbase=None, solver_timeout=None):
    rdb, exits = solve_layout(rdb, area, solver_timeout=solver_timeout)
    if not len(exits) and not len(rdb):
        return

    render_map(
        rdb,
        name,
        fmt="svg",
        header=area,
        exits=exits,
        split_levels=split_levels,
        outbase=outbase,
    )
