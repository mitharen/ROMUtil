from __future__ import annotations

import copy
import logging
from typing import Iterable, Mapping, Optional, Sequence

import networkx as nx
import pyomo.opt
from pyomo.environ import ConcreteModel, RangeSet, Param, Var, Objective, ConstraintList, Binary, NonNegativeIntegers, SolverFactory

from romutil.models import Direction, Room, Exit, RoomDef
from romutil.plotter import Plotter
from romutil.renderers import SVGRenderer, render_map
from romutil.solver import position_dummy_rooms, solve

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
    """
    minimum feedback arc set for a graph
    """
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

def compute_bounding_box(rooms: Iterable[Room]) -> tuple[int, int, int, int, int, int]:
    """
    Calculate 3D axis-aligned bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
    for an iterable of rooms with valid integer coordinates.
    Returns (0, 0, 0, 0, 0, 0) if no valid coordinates exist.
    """
    xs = [r.x for r in rooms if r.x is not None]
    ys = [r.y for r in rooms if r.y is not None]
    zs = [r.z for r in rooms if r.z is not None]
    if not xs or not ys or not zs:
        return (0, 0, 0, 0, 0, 0)
    return (
        min(xs),
        max(xs),
        min(ys),
        max(ys),
        min(zs),
        max(zs),
    )

def decompose_components(rdb: Mapping[int, Room], exits: Sequence[Exit]) -> list[set[int]]:
    """
    Build an undirected graph of non-dummy rooms and exits using networkx.Graph
    and decompose into weakly connected components.
    Returns components sorted deterministically by min VNUM.
    """
    non_dummy_vnums = [v for v, r in rdb.items() if getattr(r, 'dummy', False) is not True]
    G = nx.Graph()
    for v in non_dummy_vnums:
        G.add_node(v)
    for e in exits:
        if e.src in G and e.dst in G and e.src != e.dst:
            G.add_edge(e.src, e.dst)

    comps = list(nx.connected_components(G))
    comps.sort(key=lambda c: min(c) if c else 0)
    return comps

def _has_feasible_coordinates(model, rdb) -> bool:
    """Check if model has valid coordinates populated for all non-dummy rooms in rdb."""
    if model is None:
        return False
    if not hasattr(model, 'x') or not hasattr(model, 'y') or not hasattr(model, 'z'):
        return False
    try:
        non_dummy = [v for v, r in rdb.items() if getattr(r, 'dummy', False) is not True]
        return bool(non_dummy) and all(
            v in model.x and model.x[v].value is not None
            and v in model.y and model.y[v].value is not None
            and v in model.z and model.z[v].value is not None
            for v in non_dummy
        )
    except (KeyError, AttributeError, TypeError):
        return False

def solve_layout(rdb, area=None, solver_timeout=None, component_padding: int = 2):
    """
    Solves 3D coordinates for rooms using Pyomo MILP optimization.
    Simplifies straight hallways, resolves boundary dummies, restores hallways,
    decomposes disconnected room graphs into independent weakly connected components,
    packs multiple components along the X-axis with non-overlapping bounding boxes,
    normalizes coordinates so minimums are zero, restores original exits on rooms,
    and calculates one-way exit statuses.
    Returns (rdb, exits).
    """
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

    components = decompose_components(rdb, exits)

    if len(components) <= 1:
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

        if hasattr(model, 'cut'):
            for i, ex in enumerate(exits):
                try:
                    val = model.cut[i].value if hasattr(model.cut[i], 'value') else model.cut[i]
                    ex.cut = bool(round(val or 0))
                except (KeyError, IndexError, AttributeError):
                    ex.cut = False

        for vnum, room in list(rdb.items()):
            if getattr(room, 'dummy', False) is True:
                continue
            room.x = model.x[vnum].value if model.x[vnum].value is not None else 0
            room.y = model.y[vnum].value if model.y[vnum].value is not None else 0
            room.z = model.z[vnum].value if model.z[vnum].value is not None else 0
            for r in restore_rooms(room):
                rdb[r.vnum] = r

        position_dummy_rooms(rdb, exits)

        valid_rooms = [r for r in rdb.values() if r.x is not None and r.y is not None and r.z is not None]
        if valid_rooms:
            x_min = min([r.x for r in valid_rooms])
            y_min = min([r.y for r in valid_rooms])
            z_min = min([r.z for r in valid_rooms])
            for r in valid_rooms:
                r.x -= x_min
                r.y -= y_min
                r.z -= z_min
    else:
        pad = int(component_padding) if component_padding is not None else 2
        if pad < 2:
            log.warning(f'Component padding {pad} is less than minimum 2; clamping to 2.')
            pad = 2

        log.info(f'{area_name} Decomposed into {len(components)} independent components.')

        current_x_offset = 0

        for i, comp_vnums in enumerate(components):
            exits_i = [
                e for e in exits
                if e.src in comp_vnums or (getattr(rdb.get(e.src), 'dummy', False) and e.dst in comp_vnums)
            ]

            dummy_vnums_i = {
                e.dst for e in exits_i
                if getattr(rdb.get(e.dst), 'dummy', False)
            } | {
                e.src for e in exits_i
                if getattr(rdb.get(e.src), 'dummy', False)
            }

            rdb_i = {v: rdb[v] for v in comp_vnums}
            for dv in dummy_vnums_i:
                if dv in rdb:
                    rdb_i[dv] = copy.copy(rdb[dv])

            if not exits_i:
                for r in rdb_i.values():
                    if r.x is None: r.x = 0
                    if r.y is None: r.y = 0
                    if r.z is None: r.z = 0
                model_i = None
                results_i = None
            else:
                log.info(f'{area_name} Component {i+1}/{len(components)}: solving for {len(exits_i)} exits across {len(comp_vnums)} rooms...')
                if solver_timeout is not None:
                    model_i, results_i = solve(rdb_i, exits_i, timeout=solver_timeout)
                else:
                    model_i, results_i = solve(rdb_i, exits_i)

            tc_i = getattr(getattr(results_i, 'solver', None), 'termination_condition', None) if results_i else None
            has_valid_coords_i = _has_feasible_coordinates(model_i, rdb_i) if model_i else False

            is_optimal_i = (tc_i == pyomo.opt.TerminationCondition.optimal)
            is_time_limit_or_feasible_i = (
                tc_i in (
                    pyomo.opt.TerminationCondition.maxTimeLimit,
                    pyomo.opt.TerminationCondition.feasible,
                )
            )

            restored_rooms_i = []

            if (is_optimal_i or is_time_limit_or_feasible_i) and has_valid_coords_i and model_i is not None:
                if hasattr(model_i, 'cut'):
                    for j, ex in enumerate(exits_i):
                        try:
                            val = model_i.cut[j].value if hasattr(model_i.cut[j], 'value') else model_i.cut[j]
                            ex.cut = bool(round(val or 0))
                        except (KeyError, IndexError, AttributeError):
                            ex.cut = False
                if is_optimal_i:
                    log.info(f'{area_name} Component {i+1} solve completed.')
                else:
                    log.warning(f'{area_name} Component {i+1} reached time limit; using best feasible layout.')
                for vnum in comp_vnums:
                    room = rdb[vnum]
                    room.x = model_i.x[vnum].value if model_i.x[vnum].value is not None else 0
                    room.y = model_i.y[vnum].value if model_i.y[vnum].value is not None else 0
                    room.z = model_i.z[vnum].value if model_i.z[vnum].value is not None else 0
                    for r in restore_rooms(room):
                        rdb[r.vnum] = r
                        restored_rooms_i.append(r)
            else:
                if exits_i:
                    log.error(f'{area_name} Component {i+1} solver failed!')
                    if results_i and hasattr(results_i, 'solver'):
                        log.debug(f'{str(results_i.solver)}')
                for vnum in comp_vnums:
                    room = rdb[vnum]
                    if room.x is None: room.x = 0
                    if room.y is None: room.y = 0
                    if room.z is None: room.z = 0
                    for r in restore_rooms(room):
                        rdb[r.vnum] = r
                        if r.x is None: r.x = 0
                        if r.y is None: r.y = 0
                        if r.z is None: r.z = 0
                        restored_rooms_i.append(r)

            comp_rooms = {v: rdb[v] for v in comp_vnums}
            for r in restored_rooms_i:
                comp_rooms[r.vnum] = r
            for dv, d_room in rdb_i.items():
                if getattr(d_room, 'dummy', False):
                    comp_rooms[dv] = d_room

            position_dummy_rooms(comp_rooms, exits_i)

            c_xmin, c_xmax, c_ymin, c_ymax, c_zmin, c_zmax = compute_bounding_box(comp_rooms.values())

            for r in comp_rooms.values():
                if r.x is not None:
                    r.x -= c_xmin
                if r.y is not None:
                    r.y -= c_ymin
                if r.z is not None:
                    r.z -= c_zmin

            width_i = c_xmax - c_xmin

            for r in comp_rooms.values():
                if r.x is not None:
                    r.x += current_x_offset

            current_x_offset += width_i + pad

            for dv, d_room in comp_rooms.items():
                if getattr(d_room, 'dummy', False) and dv in rdb:
                    if rdb[dv].x is None:
                        rdb[dv].x = d_room.x
                        rdb[dv].y = d_room.y
                        rdb[dv].z = d_room.z

        position_dummy_rooms(rdb, exits)

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

    cut_map = {(e.src, e.dst, e.direction): getattr(e, 'cut', False) for e in exits}
    for r in rdb.values():
        if getattr(r, 'dummy', False):
            continue
        for e in r.exits:
            e.cut = cut_map.get((e.src, e.dst, e.direction), False)

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

def graph(rdb, name, area, split_levels=False, outbase=None, solver_timeout=None, **kwargs):
    if kwargs:
        rdb, exits = solve_layout(rdb, area, solver_timeout=solver_timeout, **kwargs)
    else:
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
