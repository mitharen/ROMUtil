from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import networkx as nx
import pyomo.opt
from romutil.models import Direction, Room, Exit, RoomDef
from romutil.solver import compute_dynamic_solver_timeout, position_dummy_rooms, solve

log = logging.getLogger('Mapper.graph')

@dataclass
class HallwayCorridor:
    """Represents a straight bidirectional hallway corridor between endpoints u and v."""

    u_vnum: int
    v_vnum: int
    direction: Direction
    total_distance: int
    rooms: list[Room]
    cumulative_distances: list[int]


class CorridorFixup(tuple):
    """
    A 3-tuple (room, direction, distance) representing a collapsed hallway room,
    annotated with corridor endpoint metadata for even spacing interpolation.

    Subclasses tuple so it can be unpacked as (r, d, dist) or indexed by [0], [1], [2]
    for complete backward compatibility.
    """

    room: Room
    direction: Direction
    distance: int
    target_vnum: Optional[int]
    total_distance: Optional[int]

    def __new__(
        cls,
        room: Room,
        direction: Optional[Direction] = None,
        distance: Optional[int] = None,
        target_vnum: Optional[int] = None,
        total_distance: Optional[int] = None,
    ) -> CorridorFixup:
        if direction is None and isinstance(room, tuple) and len(room) >= 3:
            r, d, dist = room[:3]
            tv = getattr(room, "target_vnum", target_vnum)
            td = getattr(room, "total_distance", total_distance)
            inst = super().__new__(cls, (r, d, dist))
            inst.room = r
            inst.direction = d
            inst.distance = dist
            inst.target_vnum = tv
            inst.total_distance = td
            return inst
        if direction is None or distance is None:
            raise ValueError("CorridorFixup requires (room, direction, distance)")
        inst = super().__new__(cls, (room, direction, distance))
        inst.room = room
        inst.direction = direction
        inst.distance = distance
        inst.target_vnum = target_vnum
        inst.total_distance = total_distance
        return inst

    def __reduce__(self) -> Any:
        return (
            CorridorFixup,
            (self.room, self.direction, self.distance, self.target_vnum, self.total_distance),
        )

    def __repr__(self) -> str:
        return (
            f"CorridorFixup(vnum={self.room.vnum}, dir={self.direction.name}, "
            f"dist={self.distance}, target_vnum={self.target_vnum}, total_dist={self.total_distance})"
        )


def _apply_directional_offset(x: int, y: int, z: int, d: Direction, dist: int) -> tuple[int, int, int]:
    """Calculate nominal directional offset coordinates from (x, y, z) by stepping dist in direction d."""
    if d == Direction.north:
        return x, y + dist, z
    elif d == Direction.east:
        return x + dist, y, z
    elif d == Direction.south:
        return x, y - dist, z
    elif d == Direction.west:
        return x - dist, y, z
    elif d == Direction.up:
        return x, y, z + dist
    elif d == Direction.down:
        return x, y, z - dist
    elif d == Direction.northeast:
        return x + dist, y + dist, z
    elif d == Direction.northwest:
        return x - dist, y + dist, z
    elif d == Direction.southeast:
        return x + dist, y - dist, z
    elif d == Direction.southwest:
        return x - dist, y - dist, z
    return x, y, z


def restore_rooms(room: Room, rdb: Optional[Mapping[int, Room]] = None) -> list[Room]:
    """
    Restore collapsed hallway rooms attached to `room`.

    If `rdb` is provided and the fixup references a valid endpoint room `v` in `rdb`
    with distinct solved coordinates, intermediate rooms are interpolated evenly
    along the corridor between `room` and `v`:
        x_h = round(x_u + t * (x_v - x_u))
        y_h = round(y_u + t * (y_v - y_u))
        z_h = round(z_u + t * (z_v - z_u))
    where t = dist / total_distance.

    Otherwise (e.g. degenerate endpoints where u == v, missing rdb, missing endpoint,
    or None coordinates), safely falls back to nominal unit directional offsets:
        x_h = x_u + dist * d(D)
    preserving 100% backward compatibility.
    """
    rooms: list[Room] = []
    rx = room.x if room.x is not None else 0
    ry = room.y if room.y is not None else 0
    rz = room.z if room.z is not None else 0

    for fixup in getattr(room, "fixups", ()):
        r: Room = fixup[0]
        d: Direction = fixup[1]
        dist: int = fixup[2]

        target_vnum: Optional[int] = getattr(fixup, "target_vnum", None)
        total_dist: Optional[int] = getattr(fixup, "total_distance", None)
        if target_vnum is None and len(fixup) >= 5:
            target_vnum = fixup[3]
            total_dist = fixup[4]

        # If fixup doesn't have target metadata, check room attributes on r
        if target_vnum is None:
            target_vnum = getattr(r, "_corridor_target", None)
            if total_dist is None:
                total_dist = getattr(r, "_corridor_total_dist", None)

        # Fallback target resolution: inspect exits on room in direction d
        if target_vnum is None and rdb is not None:
            for ex in getattr(room, "exits", ()):
                if ex.direction == d and ex.dst in rdb and ex.dst != room.vnum:
                    target_vnum = ex.dst
                    if total_dist is None:
                        total_dist = ex.distance
                    break

        v_room = rdb.get(target_vnum) if (rdb is not None and target_vnum is not None) else None

        can_interpolate = False
        if (
            v_room is not None
            and v_room.x is not None
            and v_room.y is not None
            and v_room.z is not None
            and total_dist is not None
            and total_dist > 0
        ):
            # Degenerate & Coincident check: endpoints must not coincide
            if (v_room.x, v_room.y, v_room.z) != (rx, ry, rz):
                can_interpolate = True

        if (
            can_interpolate
            and v_room is not None
            and total_dist is not None
            and v_room.x is not None
            and v_room.y is not None
            and v_room.z is not None
        ):
            t = dist / total_dist
            r.x = round(rx + t * (v_room.x - rx))
            r.y = round(ry + t * (v_room.y - ry))
            r.z = round(rz + t * (v_room.z - rz))
        else:
            r.x, r.y, r.z = _apply_directional_offset(rx, ry, rz, d, dist)

        rooms.append(r)
        rooms += restore_rooms(r, rdb=rdb)

    return rooms


def _is_hallway_candidate(r: Room, rdb: Mapping[int, Room]) -> bool:
    if getattr(r, "dummy", False):
        return False
    if len(r.exits) != 2:
        return False
    e0, e1 = r.exits[0], r.exits[1]
    if e0.dst not in rdb or e1.dst not in rdb:
        return False
    if e0.dst == e1.dst:
        return False
    if e0.direction != e1.direction.invert():
        return False
    n0, n1 = rdb[e0.dst], rdb[e1.dst]
    if e0 not in n0.exits or e1 not in n1.exits:
        return False
    return True


def _collapse_hallways(rdb: dict[int, Room], area_name: str) -> list[HallwayCorridor]:
    """
    Find and collapse maximal straight bidirectional hallway corridors.
    Tracks each corridor as a HallwayCorridor and populates parent.fixups
    with CorridorFixup entries containing endpoint references and total distances.
    """
    corridors: list[HallwayCorridor] = []
    collapsed_rooms: set[int] = set()

    for vnum in list(rdb.keys()):
        if vnum in collapsed_rooms or vnum not in rdb:
            continue
        r = rdb[vnum]
        if not _is_hallway_candidate(r, rdb):
            continue

        e0, e1 = r.exits[0], r.exits[1]
        D = e1.direction
        D_rev = e0.direction

        # Trace backward in direction D_rev
        backward_rooms: list[Room] = []
        curr = r
        cycle_detected = False
        seen_in_chain = {r.vnum}

        while True:
            ex_rev = next((e for e in curr.exits if e.direction == D_rev), None)
            if ex_rev is None:
                break
            prev_vnum = ex_rev.dst
            if prev_vnum in seen_in_chain:
                cycle_detected = True
                break
            if prev_vnum not in rdb or prev_vnum in collapsed_rooms:
                break
            prev_room = rdb[prev_vnum]
            if not _is_hallway_candidate(prev_room, rdb):
                break
            dirs = {e.direction for e in prev_room.exits}
            if dirs != {D, D_rev}:
                break
            seen_in_chain.add(prev_vnum)
            backward_rooms.append(prev_room)
            curr = prev_room

        if cycle_detected:
            continue

        # Trace forward in direction D
        forward_rooms: list[Room] = []
        curr = r
        while True:
            ex_fwd = next((e for e in curr.exits if e.direction == D), None)
            if ex_fwd is None:
                break
            next_vnum = ex_fwd.dst
            if next_vnum in seen_in_chain:
                cycle_detected = True
                break
            if next_vnum not in rdb or next_vnum in collapsed_rooms:
                break
            next_room = rdb[next_vnum]
            if not _is_hallway_candidate(next_room, rdb):
                break
            dirs = {e.direction for e in next_room.exits}
            if dirs != {D, D_rev}:
                break
            seen_in_chain.add(next_vnum)
            forward_rooms.append(next_room)
            curr = next_room

        if cycle_detected:
            continue

        corridor_rooms = list(reversed(backward_rooms)) + [r] + forward_rooms
        first_room = corridor_rooms[0]
        last_room = corridor_rooms[-1]

        ex_u = next((e for e in first_room.exits if e.direction == D_rev), None)
        ex_v = next((e for e in last_room.exits if e.direction == D), None)
        if ex_u is None or ex_v is None:
            continue

        u_vnum = ex_u.dst
        v_vnum = ex_v.dst
        if u_vnum == v_vnum or u_vnum not in rdb or v_vnum not in rdb:
            continue

        u_room = rdb[u_vnum]
        v_room = rdb[v_vnum]

        # Calculate step weights
        ex_from_u = next((e for e in u_room.exits if e.dst == first_room.vnum and e.direction == D), None)
        w_first = ex_from_u.distance if ex_from_u is not None else 1

        step_weights = [w_first]
        for i in range(len(corridor_rooms) - 1):
            curr_rm = corridor_rooms[i]
            nxt_rm = corridor_rooms[i + 1]
            ex_step = next((e for e in curr_rm.exits if e.dst == nxt_rm.vnum and e.direction == D), None)
            step_weights.append(ex_step.distance if ex_step is not None else 1)

        ex_to_v = next((e for e in last_room.exits if e.dst == v_vnum and e.direction == D), None)
        w_last = ex_to_v.distance if ex_to_v is not None else 1
        step_weights.append(w_last)

        total_distance = sum(step_weights)

        cum_dists: list[int] = []
        cur_d = 0
        for i in range(len(corridor_rooms)):
            cur_d += step_weights[i]
            cum_dists.append(cur_d)

        u_room.replace_exit(first_room.vnum, v_vnum, total_distance - w_first)
        v_room.replace_exit(last_room.vnum, u_vnum, total_distance - w_last)

        for i, room_i in enumerate(corridor_rooms):
            cd = cum_dists[i]
            fixup = CorridorFixup(room_i, D, cd, target_vnum=v_vnum, total_distance=total_distance)
            u_room.fixups.append(fixup)
            setattr(room_i, "_corridor_src", u_vnum)
            setattr(room_i, "_corridor_target", v_vnum)
            setattr(room_i, "_corridor_total_dist", total_distance)
            setattr(room_i, "_corridor_dist", cd)
            setattr(room_i, "_corridor_dir", D)

        for room_i in corridor_rooms:
            collapsed_rooms.add(room_i.vnum)
            if room_i.vnum in rdb:
                del rdb[room_i.vnum]
            log.debug(f"{area_name} Trimmed hallway {room_i.vnum}.")

        corridors.append(
            HallwayCorridor(
                u_vnum=u_vnum,
                v_vnum=v_vnum,
                direction=D,
                total_distance=total_distance,
                rooms=corridor_rooms,
                cumulative_distances=cum_dists,
            )
        )

    return corridors


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

def solve_layout(rdb, area=None, solver_timeout=None, component_padding: int = 2, extra_constraints_hook: Optional[Callable[[Any], None]] = None, hierarchical: bool = True):
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

    setattr(solve_layout, 'last_timeout_collision_failure', False)
    setattr(solve_layout, 'last_unresolved_collisions', {})

    if hierarchical:
        non_dummy = [r for r in rdb.values() if not getattr(r, 'dummy', False)]
        distinct_areas = {
            getattr(r, 'area_name', None) or getattr(r, 'area_file', None)
            for r in non_dummy
        }
        distinct_areas.discard(None)
        if len(distinct_areas) > 1 and len(non_dummy) > 2:
            from romutil.decomposition import solve_hierarchical_layout
            return solve_hierarchical_layout(
                rdb,
                area=area,
                solver_timeout=solver_timeout,
                component_padding=component_padding,
            )


    # Save original exits for complete export preservation
    original_exits = {vnum: copy.deepcopy(r.exits) for vnum, r in rdb.items()}

    # collapse straight bidirectional hallways
    corridors = _collapse_hallways(rdb, area_name)
    setattr(solve_layout, 'last_corridors', corridors)

    exits = list(set([e for r in rdb.values() for e in r.exits]))

    # Decouple external exits and multi-source destinations
    claimed_targets: set[int] = set()
    next_synth_vnum = -1

    for e in exits:
        orig_dst = getattr(e, 'target_vnum', None)
        if orig_dst is None:
            orig_dst = e.dst

        # Check if destination is external (not a non-dummy core room in rdb)
        is_external = (
            e.dst == -1
            or orig_dst == -1
            or e.dst not in rdb
            or getattr(rdb.get(e.dst), 'dummy', False)
        )
        if not is_external:
            continue

        e.target_vnum = orig_dst

        # If it is a valid non-negative target VNUM not yet claimed by another exit
        if orig_dst != -1 and orig_dst not in claimed_targets and orig_dst not in rdb:
            claimed_targets.add(orig_dst)
            dummy_room = Room(RoomDef(vnum=orig_dst, name=f'External {orig_dst}', description='', exits=()), target_vnum=orig_dst)
            dummy_room.dummy = True
            dummy_room.exits.append(e)
            rdb[orig_dst] = dummy_room
            e.dst = orig_dst
            if e.src in original_exits:
                for oe in original_exits[e.src]:
                    if oe.direction == e.direction and oe.dst == orig_dst:
                        oe.target_vnum = orig_dst
                        break
        else:
            # Multi-source external exit or unresolved dst == -1: decouple into dedicated synthetic dummy room
            while next_synth_vnum in rdb:
                next_synth_vnum -= 1
            synth_vnum = next_synth_vnum
            next_synth_vnum -= 1

            dummy_name = f'External {orig_dst}' if orig_dst != -1 else 'Unresolved'
            dummy_room = Room(RoomDef(vnum=synth_vnum, name=dummy_name, description='', exits=()), target_vnum=orig_dst)
            dummy_room.dummy = True
            dummy_room.exits.append(e)
            rdb[synth_vnum] = dummy_room

            # Update the exit to point to the dedicated dummy room
            e.dst = synth_vnum

            # Update matching exit in original_exits so that exit restoration retains decoupled stub
            if e.src in original_exits:
                for oe in original_exits[e.src]:
                    if oe.direction == e.direction and (oe.dst == orig_dst or oe.dst == -1):
                        oe.dst = synth_vnum
                        oe.target_vnum = orig_dst
                        break

    if not len(exits):
        log.warning(f'Ignoring disconnected room: {list(rdb.keys())}')
        for r in rdb.values():
            if r.x is None: r.x = 0
            if r.y is None: r.y = 0
            if r.z is None: r.z = 0
        return rdb, exits

    components = decompose_components(rdb, exits)

    if len(components) <= 1:
        non_dummy_vnums = [v for v, r in rdb.items() if not getattr(r, 'dummy', False)] or list(rdb.keys())
        if solver_timeout is not None:
            log.info(f'{area_name} Solving for {len(exits)} exits across {len(non_dummy_vnums)} rooms (fixed timeout {solver_timeout}s)...')
            if extra_constraints_hook is not None:
                model, results = solve(rdb, exits, timeout=solver_timeout, extra_constraints_hook=extra_constraints_hook)
            else:
                model, results = solve(rdb, exits, timeout=solver_timeout)
        else:
            comp_timeout = compute_dynamic_solver_timeout(len(non_dummy_vnums), len(exits))
            density = (len(exits) / len(non_dummy_vnums)) if non_dummy_vnums else 0.0
            log.info(f'{area_name} Solving for {len(exits)} exits across {len(non_dummy_vnums)} rooms (density {density:.2f}, dynamic timeout {comp_timeout}s)...')
            if extra_constraints_hook is not None:
                model, results = solve(rdb, exits, extra_constraints_hook=extra_constraints_hook)
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

        has_timeout_collision = bool(
            getattr(model, 'timeout_collision_failure', False)
            or getattr(results, 'timeout_collision_failure', False)
        )

        if has_timeout_collision:
            details = getattr(model, 'unresolved_collisions', {}) or getattr(results, 'unresolved_collisions', {})
            r_c = details.get('room_collisions', 0)
            c_p = details.get('collinear_penetrations', 0)
            o_p = details.get('overlapping_pairs', 0)
            log.error(
                f"{area_name} Layout solve failed to find a collision-free layout due to solver timeout: "
                f"unresolved collisions (room collisions: {r_c}, collinear penetrations: {c_p}, overlapping pairs: {o_p})"
            )

        setattr(solve_layout, 'last_timeout_collision_failure', has_timeout_collision)
        setattr(solve_layout, 'last_unresolved_collisions', getattr(model, 'unresolved_collisions', {}))


        if is_optimal and has_valid_coords:
            log.info(f'{area_name} Solve completed.')
        elif is_time_limit_or_feasible and has_valid_coords:
            log.warning(f'{area_name} Solver reached time limit; using best feasible layout.')
        else:
            log.error(f'{area_name} Solver failed!')
            if results and hasattr(results, 'solver'):
                log.debug(f'{str(results.solver)}')
            for r in list(rdb.values()):
                if r.x is None: r.x = 0
                if r.y is None: r.y = 0
                if r.z is None: r.z = 0
                for restored in restore_rooms(r, rdb=rdb):
                    rdb[restored.vnum] = restored
                    if restored.x is None: restored.x = 0
                    if restored.y is None: restored.y = 0
                    if restored.z is None: restored.z = 0
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

        for vnum, room in list(rdb.items()):
            if getattr(room, 'dummy', False) is True:
                continue
            for r in restore_rooms(room, rdb=rdb):
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
        any_timeout_collision = False
        all_unresolved_details = []

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
                if solver_timeout is not None:
                    log.info(f'{area_name} Component {i+1}/{len(components)}: solving for {len(exits_i)} exits across {len(comp_vnums)} rooms (fixed timeout {solver_timeout}s)...')
                    if extra_constraints_hook is not None:
                        model_i, results_i = solve(rdb_i, exits_i, timeout=solver_timeout, extra_constraints_hook=extra_constraints_hook)
                    else:
                        model_i, results_i = solve(rdb_i, exits_i, timeout=solver_timeout)
                else:
                    comp_timeout = compute_dynamic_solver_timeout(len(comp_vnums), len(exits_i))
                    density = (len(exits_i) / len(comp_vnums)) if comp_vnums else 0.0
                    log.info(f'{area_name} Component {i+1}/{len(components)}: solving for {len(exits_i)} exits across {len(comp_vnums)} rooms (density {density:.2f}, dynamic timeout {comp_timeout}s)...')
                    if extra_constraints_hook is not None:
                        model_i, results_i = solve(rdb_i, exits_i, extra_constraints_hook=extra_constraints_hook)
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

            has_comp_timeout_collision = bool(
                getattr(model_i, 'timeout_collision_failure', False)
                or getattr(results_i, 'timeout_collision_failure', False)
            )
            if has_comp_timeout_collision:
                any_timeout_collision = True
                details_i = getattr(model_i, 'unresolved_collisions', {}) or getattr(results_i, 'unresolved_collisions', {})
                all_unresolved_details.append(details_i)
                r_c = details_i.get('room_collisions', 0)
                c_p = details_i.get('collinear_penetrations', 0)
                o_p = details_i.get('overlapping_pairs', 0)
                log.error(
                    f"{area_name} Component {i+1} layout solve failed to find a collision-free layout due to solver timeout: "
                    f"unresolved collisions (room collisions: {r_c}, collinear penetrations: {c_p}, overlapping pairs: {o_p})"
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

                for vnum in comp_vnums:
                    room = rdb[vnum]
                    for r in restore_rooms(room, rdb=rdb):
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

                for vnum in comp_vnums:
                    room = rdb[vnum]
                    for r in restore_rooms(room, rdb=rdb):
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

        setattr(solve_layout, 'last_timeout_collision_failure', any_timeout_collision)
        setattr(
            solve_layout,
            'last_unresolved_collisions',
            {
                'components': all_unresolved_details,
                'total': sum(d.get('total', 0) for d in all_unresolved_details),
            },
        )

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
