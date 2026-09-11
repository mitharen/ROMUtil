from __future__ import annotations

import itertools
import logging
from math import sqrt
import os
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


def position_dummy_rooms(rdb: Mapping[int, Any], exits: Sequence[Exit]) -> None:
    """
    Position boundary dummy rooms exactly 1 unit distance in the nominal exit
    direction from their source room: x_dummy = x_src + d_exit.
    """
    # Reset coordinates on dummy rooms so they are anchored afresh to source rooms
    for room in rdb.values():
        if getattr(room, 'dummy', False) is True:
            room.x = None
            room.y = None
            room.z = None

    positioned: set[int] = set()

    # Pass 1: Forward exits from non-dummy source room to dummy destination room
    for ex in exits:
        if ex.src not in rdb or ex.dst not in rdb:
            continue
        src_room = rdb[ex.src]
        dst_room = rdb[ex.dst]
        if getattr(dst_room, 'dummy', False) is True and getattr(src_room, 'dummy', False) is not True:
            if dst_room.vnum in positioned:
                continue
            if src_room.x is not None and src_room.y is not None and src_room.z is not None:
                dx, dy, dz = (0, 0, 0)
                if ex.direction == Direction.north:
                    dy = 1
                elif ex.direction == Direction.east:
                    dx = 1
                elif ex.direction == Direction.south:
                    dy = -1
                elif ex.direction == Direction.west:
                    dx = -1
                elif ex.direction == Direction.up:
                    dz = 1
                elif ex.direction == Direction.down:
                    dz = -1
                dst_room.x = src_room.x + dx
                dst_room.y = src_room.y + dy
                dst_room.z = src_room.z + dz
                positioned.add(dst_room.vnum)

    # Pass 2: Reverse exits from dummy source room to non-dummy destination room
    for ex in exits:
        if ex.src not in rdb or ex.dst not in rdb:
            continue
        src_room = rdb[ex.src]
        dst_room = rdb[ex.dst]
        if getattr(src_room, 'dummy', False) is True and getattr(dst_room, 'dummy', False) is not True:
            if src_room.vnum in positioned:
                continue
            if dst_room.x is not None and dst_room.y is not None and dst_room.z is not None:
                dx, dy, dz = (0, 0, 0)
                if ex.direction == Direction.north:
                    dy = 1
                elif ex.direction == Direction.east:
                    dx = 1
                elif ex.direction == Direction.south:
                    dy = -1
                elif ex.direction == Direction.west:
                    dx = -1
                elif ex.direction == Direction.up:
                    dz = 1
                elif ex.direction == Direction.down:
                    dz = -1
                src_room.x = dst_room.x - dx
                src_room.y = dst_room.y - dy
                src_room.z = dst_room.z - dz
                positioned.add(src_room.vnum)

    # Pass 3: Fallback for any disconnected or unanchored dummy rooms
    for room in rdb.values():
        if getattr(room, 'dummy', False) is True:
            if room.x is None:
                room.x = 0
            if room.y is None:
                room.y = 0
            if room.z is None:
                room.z = 0


def compute_dimension_bounds(exits: Sequence[Exit]) -> tuple[int, int, int]:
    """
    Calculate tight dimension-specific bounds (Mx, My, Mz) based on exit lengths
    oriented along each respective axis, with safe lower bounds.

    Mx = max(10, sum(|dx|) + 1)
    My = max(10, sum(|dy|) + 1)
    Mz = max(5, sum(|dz|) + 1)

    Returns:
        tuple[int, int, int]: (Mx, My, Mz)
    """
    sum_x = sum(e.distance for e in exits if e.direction in (Direction.east, Direction.west))
    sum_y = sum(e.distance for e in exits if e.direction in (Direction.north, Direction.south))
    sum_z = sum(e.distance for e in exits if e.direction in (Direction.up, Direction.down))

    mx = max(10, sum_x + 1)
    my = max(10, sum_y + 1)
    mz = max(5, sum_z + 1)
    return mx, my, mz


def get_candidate_batch_cap(num_candidates: int) -> int:
    """
    Calculate the constraint batch size cap per solver iteration to prevent
    combinatorial explosion in CBC branch-and-cut:
    cap = min(15, max(5, int(sqrt(num_candidates)))) if num_candidates > 0 else 0
    """
    if num_candidates <= 0:
        return 0
    return min(15, max(5, int(sqrt(num_candidates))))


def add_overlap_constraint(
    m: ConcreteModel,
    ex: Exit,
    nx: Exit,
    left: int,
    right: int,
    relations: int,
    coords: Optional[Mapping[int, Any]] = None,
    has_vertical_exits: bool = True,
    dummy_anchors: Optional[Mapping[int, tuple[int, int, int, int]]] = None,
) -> int:
    """
    Add disjunctive spatial separation constraints between two overlapping exit segments.

    If vertical separation is inapplicable (e.g. no vertical exits exist in the area/subgraph,
    or both exits are horizontal and lie on the same horizontal plane), separation directions
    are reduced from 6 to 4 (North, South, East, West), eliminating 2 binary variables
    per overlap constraint (33% reduction in binary decision variables).

    Disjunctive Big-M constraints use dimension-specific bounds (2 * Mx, 2 * My, 2 * Mz)
    rather than monolithic M. Supports affine dummy room endpoints when dummy_anchors is provided.

    Returns:
        int: The next relation index (relations + 1).
    """
    z_match = False
    if coords is not None:
        p1 = _get_coords(coords, ex.src)
        p2 = _get_coords(coords, nx.src)
        if p1 is not None and p2 is not None:
            z_match = (p1[2] == p2[2])

    is_horizontal_pair = (
        ex.direction not in (Direction.up, Direction.down)
        and nx.direction not in (Direction.up, Direction.down)
    )

    is_planar = (not has_vertical_exits) or (is_horizontal_pair and (coords is None or z_match))

    if is_planar:
        active_dirs = [Direction.north, Direction.east, Direction.south, Direction.west]
    else:
        active_dirs = [
            Direction.north,
            Direction.east,
            Direction.up,
            Direction.south,
            Direction.west,
            Direction.down,
        ]

    relation = Var(active_dirs, within=Boolean)
    m.add_component(f'relation{relations}', relation)
    m.crossings.add(sum(relation[d] for d in active_dirs) >= 1)

    mx = getattr(m, 'Mx', getattr(m, 'M', 100))
    my = getattr(m, 'My', getattr(m, 'M', 100))
    mz = getattr(m, 'Mz', getattr(m, 'M', 100))
    d_min = getattr(m, 'd_min', 1)

    def _get_var_x(v: int):
        if v in m.Rooms:
            return m.x[v]
        if dummy_anchors and v in dummy_anchors:
            s, dx, _, _ = dummy_anchors[v]
            if s in m.Rooms:
                return m.x[s] + dx
        return 0

    def _get_var_y(v: int):
        if v in m.Rooms:
            return m.y[v]
        if dummy_anchors and v in dummy_anchors:
            s, _, dy, _ = dummy_anchors[v]
            if s in m.Rooms:
                return m.y[s] + dy
        return 0

    def _get_var_z(v: int):
        if v in m.Rooms:
            return m.z[v]
        if dummy_anchors and v in dummy_anchors:
            s, _, _, dz = dummy_anchors[v]
            if s in m.Rooms:
                return m.z[s] + dz
        return 0

    prod = list(itertools.product((ex.src, ex.dst), (nx.src, nx.dst)))

    # East: x[p] - x[q] >= d_min
    if Direction.east in active_dirs:
        for p, q in prod:
            m.crossings.add(
                _get_var_x(p) - _get_var_x(q) + (2 * mx + 4) * ((1 - relation[Direction.east]) + m.cut[left] + m.cut[right]) >= d_min
            )

    # West: x[q] - x[p] >= d_min
    if Direction.west in active_dirs:
        for p, q in prod:
            m.crossings.add(
                _get_var_x(q) - _get_var_x(p) + (2 * mx + 4) * ((1 - relation[Direction.west]) + m.cut[left] + m.cut[right]) >= d_min
            )

    # North: y[p] - y[q] >= d_min
    if Direction.north in active_dirs:
        for p, q in prod:
            m.crossings.add(
                _get_var_y(p) - _get_var_y(q) + (2 * my + 4) * ((1 - relation[Direction.north]) + m.cut[left] + m.cut[right]) >= d_min
            )

    # South: y[q] - y[p] >= d_min
    if Direction.south in active_dirs:
        for p, q in prod:
            m.crossings.add(
                _get_var_y(q) - _get_var_y(p) + (2 * my + 4) * ((1 - relation[Direction.south]) + m.cut[left] + m.cut[right]) >= d_min
            )

    # Up: z[p] - z[q] >= d_min
    if Direction.up in active_dirs:
        for p, q in prod:
            m.crossings.add(
                _get_var_z(p) - _get_var_z(q) + (2 * mz + 4) * ((1 - relation[Direction.up]) + m.cut[left] + m.cut[right]) >= d_min
            )

    # Down: z[q] - z[p] >= d_min
    if Direction.down in active_dirs:
        for p, q in prod:
            m.crossings.add(
                _get_var_z(q) - _get_var_z(p) + (2 * mz + 4) * ((1 - relation[Direction.down]) + m.cut[left] + m.cut[right]) >= d_min
            )

    return relations + 1


def add_dummy_separation_constraint(
    m: ConcreteModel,
    v: int,
    s: int,
    dx: int,
    dy: int,
    dz: int,
    relations: int,
    coords: Optional[Mapping[int, Any]] = None,
    has_vertical_exits: bool = True,
) -> int:
    """
    Add disjunctive spatial separation constraints between a core room v and
    an affine dummy stub coordinate anchored at (x_s + dx, y_s + dy, z_s + dz).

    Enforces that room v is separated from the dummy stub by at least 1 unit
    along at least one axis without introducing new integer decision variables:
      East:  x_v - x_s >= 1 + dx
      West:  x_s - x_v >= 1 - dx
      North: y_v - y_s >= 1 + dy
      South: y_s - y_v >= 1 - dy
      Up:    z_v - z_s >= 1 + dz
      Down:  z_s - z_v >= 1 - dz

    Returns:
        int: The next relation index (relations + 1).
    """
    z_match = False
    if coords is not None:
        p_v = _get_coords(coords, v)
        p_s = _get_coords(coords, s)
        if p_v is not None and p_s is not None:
            z_match = (p_v[2] == p_s[2] + dz)

    is_planar = (not has_vertical_exits) or (dz == 0 and (coords is None or z_match))

    if is_planar:
        active_dirs = [Direction.north, Direction.east, Direction.south, Direction.west]
    else:
        active_dirs = [
            Direction.north,
            Direction.east,
            Direction.up,
            Direction.south,
            Direction.west,
            Direction.down,
        ]

    relation = Var(active_dirs, within=Boolean)
    m.add_component(f'dummy_rel_{relations}', relation)
    m.crossings.add(sum(relation[d] for d in active_dirs) >= 1)

    mx = getattr(m, 'Mx', getattr(m, 'M', 100))
    my = getattr(m, 'My', getattr(m, 'M', 100))
    mz = getattr(m, 'Mz', getattr(m, 'M', 100))

    big_mx = 2 * mx + 4
    big_my = 2 * my + 4
    big_mz = 2 * mz + 4

    # East: x_v - x_s >= 1 + dx
    if Direction.east in active_dirs:
        m.crossings.add(
            m.x[v] - m.x[s] + big_mx * (1 - relation[Direction.east]) >= 1 + dx
        )

    # West: x_s - x_v >= 1 - dx
    if Direction.west in active_dirs:
        m.crossings.add(
            m.x[s] - m.x[v] + big_mx * (1 - relation[Direction.west]) >= 1 - dx
        )

    # North: y_v - y_s >= 1 + dy
    if Direction.north in active_dirs:
        m.crossings.add(
            m.y[v] - m.y[s] + big_my * (1 - relation[Direction.north]) >= 1 + dy
        )

    # South: y_s - y_v >= 1 - dy
    if Direction.south in active_dirs:
        m.crossings.add(
            m.y[s] - m.y[v] + big_my * (1 - relation[Direction.south]) >= 1 - dy
        )

    # Up: z_v - z_s >= 1 + dz
    if Direction.up in active_dirs:
        m.crossings.add(
            m.z[v] - m.z[s] + big_mz * (1 - relation[Direction.up]) >= 1 + dz
        )

    # Down: z_s - z_v >= 1 - dz
    if Direction.down in active_dirs:
        m.crossings.add(
            m.z[s] - m.z[v] + big_mz * (1 - relation[Direction.down]) >= 1 - dz
        )

    return relations + 1


def get_cbc_solver(timeout: Optional[int] = None, **custom_options: Any) -> Any:
    """
    Instantiate and configure the Coin-OR CBC MILP solver with tuned options.

    Configures:
      - threads: min(4, os.cpu_count() or 1) for parallel branch-and-bound.
      - ratioGap: 0.05 (5% relative MIP gap tolerance) to prevent branch-and-bound
        tailing off while guaranteeing visually indistinguishable layouts.
      - presolve: 'on' for aggressive preprocessing and problem reduction.
      - cuts: 'on' for cutting-plane generation.
      - heuristics: 'on' for primal integer heuristics.
      - seconds: execution timeout in seconds if specified (also sets 'sec' for
        Pyomo backward compatibility).
      - custom_options: any additional or overridden solver options.

    Args:
        timeout: Optional time limit in seconds.
        **custom_options: Additional solver options to set or override.

    Returns:
        Configured Pyomo solver plugin instance for Coin-OR CBC.
    """
    solver = SolverFactory('cbc', tee=False)
    cpu_count = os.cpu_count() or 1
    solver.options['threads'] = min(4, cpu_count)
    solver.options['timeM'] = 'elapsed'
    solver.options['ratioGap'] = 0.05
    solver.options['presolve'] = 'on'
    solver.options['cuts'] = 'on'
    solver.options['heuristics'] = 'on'
    if timeout is not None:
        timeout_sec = int(timeout)
        solver.options['seconds'] = timeout_sec
        solver.options['sec'] = timeout_sec
    for k, v in custom_options.items():
        solver.options[k] = v
    return solver


def non_euler(rdb, exits):
    m = ConcreteModel()

    non_dummy_rooms = [v for v, r in rdb.items() if not getattr(r, 'dummy', False)]
    m.Rooms = Set(initialize=non_dummy_rooms)
    m.Exits = RangeSet(0, len(exits) - 1)
    m.Directions = RangeSet(0, Direction.mod.value - 1)

    mx, my, mz = compute_dimension_bounds(exits)
    m.Mx = Param(initialize=mx)
    m.My = Param(initialize=my)
    m.Mz = Param(initialize=mz)
    m.M = Param(initialize=sum([e.distance for e in exits]))

    m.x = Var(m.Rooms, within=Integers, bounds=(-m.Mx, m.Mx))
    m.y = Var(m.Rooms, within=Integers, bounds=(-m.My, m.My))
    m.z = Var(m.Rooms, within=Integers, bounds=(-m.Mz, m.Mz))
    m.cut = Var(m.Exits, within=Binary)
    m.relative_pos = ConstraintList()

    one_way_exits = [
        e for e in exits
        if e.dst not in rdb
        or getattr(rdb.get(e.dst), 'dummy', False)
        or getattr(rdb.get(e.src), 'dummy', False)
        or e not in rdb[e.dst].exits
    ]

    for i, e in enumerate(exits):
        if e in one_way_exits:
            continue
        if e.src not in m.Rooms or e.dst not in m.Rooms:
            continue

        if e.dst != e.src:
            if e.direction not in (Direction.east, Direction.west):
                m.relative_pos.add(m.x[e.src] - m.x[e.dst] + 2 * m.Mx * m.cut[i] >= 0)
                m.relative_pos.add(m.x[e.dst] - m.x[e.src] + 2 * m.Mx * m.cut[i] >= 0)
            if e.direction not in (Direction.north, Direction.south):
                m.relative_pos.add(m.y[e.src] - m.y[e.dst] + 2 * m.My * m.cut[i] >= 0)
                m.relative_pos.add(m.y[e.dst] - m.y[e.src] + 2 * m.My * m.cut[i] >= 0)
            if e.direction not in (Direction.up, Direction.down):
                m.relative_pos.add(m.z[e.src] - m.z[e.dst] + 2 * m.Mz * m.cut[i] >= 0)
                m.relative_pos.add(m.z[e.dst] - m.z[e.src] + 2 * m.Mz * m.cut[i] >= 0)

            if e.direction == Direction.east:
                m.relative_pos.add(m.x[e.dst] - m.x[e.src] + 2 * m.Mx * m.cut[i] >= 1)
            elif e.direction == Direction.north:
                m.relative_pos.add(m.y[e.dst] - m.y[e.src] + 2 * m.My * m.cut[i] >= 1)
            elif e.direction == Direction.up:
                m.relative_pos.add(m.z[e.dst] - m.z[e.src] + 2 * m.Mz * m.cut[i] >= 1)
            elif e.direction == Direction.west:
                m.relative_pos.add(m.x[e.src] - m.x[e.dst] + 2 * m.Mx * m.cut[i] >= 1)
            elif e.direction == Direction.south:
                m.relative_pos.add(m.y[e.src] - m.y[e.dst] + 2 * m.My * m.cut[i] >= 1)
            elif e.direction == Direction.down:
                m.relative_pos.add(m.z[e.src] - m.z[e.dst] + 2 * m.Mz * m.cut[i] >= 1)

    m.obj = Objective(expr=sum(m.cut[i] for i in range(len(exits))))

    solver = get_cbc_solver(timeout=20)
    solver.solve(m, tee=False)
    return m

def solve(rdb, area_exits, timeout=None):
    exits = [e for e in area_exits if e.src != e.dst]

    m = non_euler(rdb, exits)
    for i in range(len(exits)):
        if m.cut[i].value:
            exits[i].one_way = True

    for room in rdb.values():
        if not getattr(room, 'dummy', False) and not len(room.exits):
            e = Exit(ExitDef(direction=0, dst_vnum=room.vnum), source=room.vnum)
            exits.append(e)

    dummy_anchors: dict[int, tuple[int, int, int, int]] = {}
    for e in exits:
        if e.src in rdb and e.dst in rdb:
            if getattr(rdb[e.dst], 'dummy', False) and not getattr(rdb[e.src], 'dummy', False):
                if e.dst not in dummy_anchors:
                    dx, dy, dz = (0, 0, 0)
                    if e.direction == Direction.north:
                        dy = 1
                    elif e.direction == Direction.east:
                        dx = 1
                    elif e.direction == Direction.south:
                        dy = -1
                    elif e.direction == Direction.west:
                        dx = -1
                    elif e.direction == Direction.up:
                        dz = 1
                    elif e.direction == Direction.down:
                        dz = -1
                    dummy_anchors[e.dst] = (e.src, dx, dy, dz)

    non_dummy_rooms = [v for v, r in rdb.items() if not getattr(r, 'dummy', False)]
    m = ConcreteModel()

    m.Rooms = Set(initialize=non_dummy_rooms)
    m.Exits = RangeSet(0, len(exits) - 1)
    m.Directions = RangeSet(0, Direction.mod.value - 1)

    mx, my, mz = compute_dimension_bounds(exits)
    m.Mx = Param(initialize=mx)
    m.My = Param(initialize=my)
    m.Mz = Param(initialize=mz)
    m.M = Param(initialize=sum([e.distance for e in exits]))
    m.d_min = Param(initialize=1)
    m.l_min = Param(m.Exits, initialize=lambda m, x: exits[x].distance)

    m.x = Var(m.Rooms, within=Integers, bounds=(-m.Mx, m.Mx))
    m.y = Var(m.Rooms, within=Integers, bounds=(-m.My, m.My))
    m.z = Var(m.Rooms, within=Integers, bounds=(-m.Mz, m.Mz))
    m.cut = Var(m.Exits, within=Binary)
    m.l_max = Var(m.Exits, within=PositiveIntegers)
    m.one_ways = VarList(within=NonNegativeIntegers, bounds=(0, 2 * m.M))

    m.relative_pos = ConstraintList()
    m.one_way_pos = ConstraintList()
    m.crossings = ConstraintList()

    def get_x(vnum: int):
        if vnum in m.Rooms:
            return m.x[vnum]
        if vnum in dummy_anchors:
            src, dx, _, _ = dummy_anchors[vnum]
            if src in m.Rooms:
                return m.x[src] + dx
        return 0

    def get_y(vnum: int):
        if vnum in m.Rooms:
            return m.y[vnum]
        if vnum in dummy_anchors:
            src, _, dy, _ = dummy_anchors[vnum]
            if src in m.Rooms:
                return m.y[src] + dy
        return 0

    def get_z(vnum: int):
        if vnum in m.Rooms:
            return m.z[vnum]
        if vnum in dummy_anchors:
            src, _, _, dz = dummy_anchors[vnum]
            if src in m.Rooms:
                return m.z[src] + dz
        return 0

    one_ways = []

    for e in exits:
        if (
            e.dst not in rdb
            or getattr(rdb.get(e.dst), 'dummy', False)
            or getattr(rdb.get(e.src), 'dummy', False)
            or e not in rdb[e.dst].exits
        ):
            e.one_way = True

    for i, x in enumerate(exits):
        if x.src not in m.Rooms and x.dst not in m.Rooms:
            continue
        if x.dst not in m.Rooms and dummy_anchors.get(x.dst, (None,))[0] == x.src:
            continue
        if x.src not in m.Rooms and dummy_anchors.get(x.src, (None,))[0] == x.dst:
            continue

        if x.one_way:
            x_off = m.d_min if x.direction == Direction.east else -m.d_min if x.direction == Direction.west else 0
            y_off = m.d_min if x.direction == Direction.north else -m.d_min if x.direction == Direction.south else 0
            z_off = m.d_min if x.direction == Direction.up else -m.d_min if x.direction == Direction.down else 0
            X = m.one_ways.add()
            Y = m.one_ways.add()
            Z = m.one_ways.add()
            x_diff = get_x(x.dst) - get_x(x.src) - x_off
            m.one_way_pos.add(x_diff <= X)
            m.one_way_pos.add(-x_diff <= X)
            y_diff = get_y(x.dst) - get_y(x.src) - y_off
            m.one_way_pos.add(y_diff <= Y)
            m.one_way_pos.add(-y_diff <= Y)
            z_diff = get_z(x.dst) - get_z(x.src) - z_off
            m.one_way_pos.add(z_diff <= Z)
            m.one_way_pos.add(-z_diff <= Z)
            one_ways.append(X + Y + Z)

            if x.direction == Direction.east:
                m.one_way_pos.add(get_x(x.dst) - get_x(x.src) + 2 * m.Mx * m.cut[i] >= m.d_min)
            elif x.direction == Direction.west:
                m.one_way_pos.add(get_x(x.src) - get_x(x.dst) + 2 * m.Mx * m.cut[i] >= m.d_min)
            elif x.direction == Direction.north:
                m.one_way_pos.add(get_y(x.dst) - get_y(x.src) + 2 * m.My * m.cut[i] >= m.d_min)
            elif x.direction == Direction.south:
                m.one_way_pos.add(get_y(x.src) - get_y(x.dst) + 2 * m.My * m.cut[i] >= m.d_min)
            elif x.direction == Direction.up:
                m.one_way_pos.add(get_z(x.dst) - get_z(x.src) + 2 * m.Mz * m.cut[i] >= m.d_min)
            elif x.direction == Direction.down:
                m.one_way_pos.add(get_z(x.src) - get_z(x.dst) + 2 * m.Mz * m.cut[i] >= m.d_min)

            continue

        if x.dst != x.src and x.src in m.Rooms and x.dst in m.Rooms:
            if x.direction not in (Direction.east, Direction.west):
                m.relative_pos.add(m.x[x.src] + 2 * m.Mx * m.cut[i] >= m.x[x.dst])
                m.relative_pos.add(m.x[x.dst] + 2 * m.Mx * m.cut[i] >= m.x[x.src])
            if x.direction not in (Direction.north, Direction.south):
                m.relative_pos.add(m.y[x.src] + 2 * m.My * m.cut[i] >= m.y[x.dst])
                m.relative_pos.add(m.y[x.dst] + 2 * m.My * m.cut[i] >= m.y[x.src])
            if x.direction not in (Direction.up, Direction.down):
                m.relative_pos.add(m.z[x.src] + 2 * m.Mz * m.cut[i] >= m.z[x.dst])
                m.relative_pos.add(m.z[x.dst] + 2 * m.Mz * m.cut[i] >= m.z[x.src])

            if x.direction == Direction.east:
                m.relative_pos.add(m.x[x.dst] - m.x[x.src] + 2 * m.Mx * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.x[x.dst] - m.x[x.src] - 2 * m.Mx * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.north:
                m.relative_pos.add(m.y[x.dst] - m.y[x.src] + 2 * m.My * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.y[x.dst] - m.y[x.src] - 2 * m.My * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.up:
                m.relative_pos.add(m.z[x.dst] - m.z[x.src] + 2 * m.Mz * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.z[x.dst] - m.z[x.src] - 2 * m.Mz * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.west:
                m.relative_pos.add(m.x[x.src] - m.x[x.dst] + 2 * m.Mx * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.x[x.src] - m.x[x.dst] - 2 * m.Mx * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.south:
                m.relative_pos.add(m.y[x.src] - m.y[x.dst] + 2 * m.My * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.y[x.src] - m.y[x.dst] - 2 * m.My * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.down:
                m.relative_pos.add(m.z[x.src] - m.z[x.dst] + 2 * m.Mz * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.z[x.src] - m.z[x.dst] - 2 * m.Mz * m.cut[i] <= m.l_max[i])

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

    timeout_sec = 30 if timeout is None else int(timeout)
    solver = get_cbc_solver(timeout=timeout_sec)
    constrained_dummy_collisions: set[tuple[int, int]] = set()

    while True:
        result = solver.solve(m, tee=False)
        if result.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
            return m, result

        if result.solver.termination_condition in (
            pyomo.opt.TerminationCondition.maxTimeLimit,
            pyomo.opt.TerminationCondition.feasible,
        ):
            if result.solver.termination_condition == pyomo.opt.TerminationCondition.maxTimeLimit:
                log.warning('Solver reached time limit.')
            has_feasible = bool(non_dummy_rooms) and all(
                hasattr(m, 'x') and hasattr(m, 'y') and hasattr(m, 'z')
                and vnum in m.x and m.x[vnum].value is not None
                and vnum in m.y and m.y[vnum].value is not None
                and vnum in m.z and m.z[vnum].value is not None
                for vnum in non_dummy_rooms
            )
            if has_feasible:
                for vnum in non_dummy_rooms:
                    room = rdb[vnum]
                    room.x = m.x[vnum].value if m.x[vnum].value is not None else 0
                    room.y = m.y[vnum].value if m.y[vnum].value is not None else 0
                    room.z = m.z[vnum].value if m.z[vnum].value is not None else 0
                position_dummy_rooms(rdb, exits)
                Plotter('progress.svg', rdb, exits).plot()
            return m, result

        for vnum in non_dummy_rooms:
            room = rdb[vnum]
            room.x = m.x[vnum].value if m.x[vnum].value is not None else 0
            room.y = m.y[vnum].value if m.y[vnum].value is not None else 0
            room.z = m.z[vnum].value if m.z[vnum].value is not None else 0
        position_dummy_rooms(rdb, exits)
        Plotter('progress.svg', rdb, exits).plot()

        has_vertical_exits = any(e.direction in (Direction.up, Direction.down) for e in exits)
        added_constraints = 0

        coords = {
            vnum: (room.x, room.y, room.z)
            for vnum, room in rdb.items()
            if room.x is not None and room.y is not None and room.z is not None
        }
        cut_values = [bool(m.cut[i].value) for i in range(len(exits))]

        # Check for room-to-dummy point collisions
        for dv, anchor in dummy_anchors.items():
            src_vnum, dx, dy, dz = anchor
            if src_vnum not in m.Rooms:
                continue
            d_pos = _get_coords(coords, dv)
            if d_pos is None:
                continue
            for vnum in non_dummy_rooms:
                if vnum == src_vnum:
                    continue
                v_pos = _get_coords(coords, vnum)
                if v_pos is not None and v_pos == d_pos:
                    if (vnum, dv) not in constrained_dummy_collisions:
                        constrained_dummy_collisions.add((vnum, dv))
                        relations = add_dummy_separation_constraint(
                            m,
                            vnum,
                            src_vnum,
                            dx,
                            dy,
                            dz,
                            relations,
                            coords=coords,
                            has_vertical_exits=has_vertical_exits,
                        )
                        added_constraints += 1
                        log.info(
                            f'Separated core room {vnum} from colliding dummy stub {dv} '
                            f'(anchored to {src_vnum} with offset {(dx, dy, dz)})'
                        )

        overlapping_pairs = find_overlap_candidates(
            exits,
            coords,
            cuts=cut_values,
            candidate_pairs=non_incidents,
        )

        batch_target = get_candidate_batch_cap(len(overlapping_pairs))

        for left, right in tqdm.tqdm(overlapping_pairs, desc='Finding Overlaps'):
            ex, nx = (exits[left], exits[right])
            def _endpoint_valid(v: int) -> bool:
                return v in m.Rooms or (v in dummy_anchors and dummy_anchors[v][0] in m.Rooms)

            if not (_endpoint_valid(ex.src) and _endpoint_valid(ex.dst) and _endpoint_valid(nx.src) and _endpoint_valid(nx.dst)):
                non_incidents.discard((left, right))
                continue

            relations = add_overlap_constraint(
                m,
                ex,
                nx,
                left,
                right,
                relations,
                coords=coords,
                has_vertical_exits=has_vertical_exits,
                dummy_anchors=dummy_anchors,
            )

            non_incidents.discard((left, right))
            added_constraints += 1
            if added_constraints >= batch_target:
                break
        else:
            if not added_constraints:
                break

    cut_sum = sum(m.cut[i].value for i in range(len(exits)) if m.cut[i].value is not None)
    log.debug(f'cut {cut_sum} exits')
    log.debug(f'{relations}/{len(non_incidents)} overlaps converted into constraints.')
    position_dummy_rooms(rdb, exits)
    return m, result
