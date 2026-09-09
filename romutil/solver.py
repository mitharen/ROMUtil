from __future__ import annotations

import itertools
import logging
from math import sqrt
from typing import Any, Mapping, Optional, Sequence, Set as TypingSet, Tuple

import pyomo.opt
from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Objective,
    ConstraintList, Integers, Binary, PositiveIntegers,
    NonNegativeIntegers, Boolean, VarList, SolverFactory
)
import tqdm

from romutil.models import Direction, Exit, ExitDef
from romutil.plotter import Plotter

log = logging.getLogger('Mapper.solver')


def _get_coords(coords: Any, vnum: int) -> Optional[Tuple[float, float, float]]:
    """Helper to resolve (x, y, z) coordinates for a given room vnum."""
    if coords is None:
        return None
    if hasattr(coords, 'x') and hasattr(coords, 'y') and hasattr(coords, 'z'):
        try:
            xv = coords.x[vnum].value
            yv = coords.y[vnum].value
            zv = coords.z[vnum].value
            if xv is None or yv is None or zv is None:
                return None
            return (float(xv), float(yv), float(zv))
        except (KeyError, TypeError, AttributeError):
            return None

    try:
        val = coords.get(vnum) if hasattr(coords, 'get') else coords[vnum]
    except (KeyError, IndexError, TypeError):
        return None

    if val is None:
        return None
    if hasattr(val, 'x') and hasattr(val, 'y') and hasattr(val, 'z'):
        if val.x is None or val.y is None or val.z is None:
            return None
        return (float(val.x), float(val.y), float(val.z))
    if isinstance(val, (tuple, list)) and len(val) >= 3:
        if val[0] is None or val[1] is None or val[2] is None:
            return None
        return (float(val[0]), float(val[1]), float(val[2]))
    return None


def find_overlap_candidates(
    exits: Sequence[Exit],
    coords: Any,
    cuts: Optional[Sequence[Any]] = None,
    candidate_pairs: Optional[TypingSet[Tuple[int, int]]] = None,
    aabb_3d: bool = True,
) -> list[tuple[int, int]]:
    """
    1D sweep-line spatial indexing along the X-axis for overlap candidate detection.

    Projects each exit segment into an interval [x_min, x_max] on the X-axis, sorts
    event points in O(E log E) time, and maintains an active interval set during the sweep.
    Segments overlapping along the X-axis are evaluated against 3D AABB bounding boxes
    and candidate filters in O(1) set lookups, avoiding O(E^2) pair evaluations.

    Args:
        exits: Sequence of Exit objects.
        coords: Mapping of room vnum to (x, y, z) coordinates, or room dict, or Pyomo model.
        cuts: Optional sequence of cut statuses (booleans or Pyomo binary variables).
        candidate_pairs: Optional set of allowed/unresolved (left, right) index pairs with left < right.
        aabb_3d: If True (default), verifies full 3D AABB overlap (Y and Z axes).

    Returns:
        List of (left, right) index pairs (left < right) that overlap in space.
    """
    if not exits or coords is None:
        return []

    boxes: dict[int, tuple[float, float, float, float, float, float]] = {}
    endpoints: dict[int, tuple[int, int]] = {}
    events: list[tuple[float, int, int]] = []
    START_EVENT = 0
    END_EVENT = 1

    for i, ex in enumerate(exits):
        if getattr(ex, 'one_way', False):
            continue
        if cuts is not None and i < len(cuts):
            c_val = cuts[i]
            if hasattr(c_val, 'value'):
                c_val = c_val.value
            if c_val:
                continue

        p1 = _get_coords(coords, ex.src)
        p2 = _get_coords(coords, ex.dst)
        if p1 is None or p2 is None:
            continue

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        z_min, z_max = (z1, z2) if z1 <= z2 else (z2, z1)

        boxes[i] = (x_min, x_max, y_min, y_max, z_min, z_max)
        endpoints[i] = (ex.src, ex.dst)
        events.append((x_min, START_EVENT, i))
        events.append((x_max, END_EVENT, i))

    if not events:
        return []

    # Sort events: primary x asc, secondary START (0) before END (1), tertiary index asc
    events.sort(key=lambda ev: (ev[0], ev[1], ev[2]))

    active_set: set[int] = set()
    overlaps: list[tuple[int, int]] = []

    for _, ev_type, idx in events:
        if ev_type == START_EVENT:
            b_idx = boxes[idx]
            s1, d1 = endpoints[idx]
            b_min_y, b_max_y, b_min_z, b_max_z = b_idx[2], b_idx[3], b_idx[4], b_idx[5]
            for active_idx in active_set:
                left, right = (active_idx, idx) if active_idx < idx else (idx, active_idx)
                if candidate_pairs is not None and (left, right) not in candidate_pairs:
                    continue
                s2, d2 = endpoints[active_idx]
                # Skip incident exits sharing any room endpoint
                if s1 == s2 or s1 == d2 or d1 == s2 or d1 == d2:
                    continue
                if aabb_3d:
                    b_act = boxes[active_idx]
                    # Y-axis overlap: max(y_min1, y_min2) <= min(y_max1, y_max2)
                    if b_min_y > b_act[3] or b_act[2] > b_max_y:
                        continue
                    # Z-axis overlap: max(z_min1, z_min2) <= min(z_max1, z_max2)
                    if b_min_z > b_act[5] or b_act[4] > b_max_z:
                        continue
                overlaps.append((left, right))
            active_set.add(idx)
        else:
            active_set.remove(idx)

    return overlaps


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
            e = Exit(ExitDef(direction=0, dst_vnum=room.vnum), source=room.vnum)
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

    non_incidents: set[tuple[int, int]] = {
        pair for pair in itertools.combinations(range(len(exits)), 2)
        if exits[pair[0]].src not in exits[pair[1]] and exits[pair[0]].dst not in exits[pair[1]]
        and not exits[pair[0]].one_way and not exits[pair[1]].one_way
    }

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
        batch_target = max(1, int(sqrt(len(non_incidents)))) if non_incidents else 0

        coords = {
            vnum: (m.x[vnum].value, m.y[vnum].value, m.z[vnum].value)
            for vnum in rdb
        }
        cut_values = [bool(m.cut[i].value) for i in range(len(exits))]

        overlapping_pairs = find_overlap_candidates(
            exits,
            coords,
            cuts=cut_values,
            candidate_pairs=non_incidents,
        )

        for left, right in tqdm.tqdm(overlapping_pairs, desc='Finding Overlaps'):
            ex, nx = (exits[left], exits[right])

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

            non_incidents.discard((left, right))
            added_constraints += 1
            if added_constraints == batch_target:
                break
        else:
            if not added_constraints:
                break

    log.debug(f'cut {sum([m.cut[i].value for i in range(len(exits))])} exits')
    log.debug(f'{relations}/{len(non_incidents)} overlaps converted into constraints.')
    return m, result
