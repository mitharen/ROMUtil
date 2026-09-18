"""
Hierarchical Multi-Area Graph Decomposition & Macro-Assembly Architecture (Task 10h).

Provides:
- AreaProfile: Geometric profiling of independently solved areas.
- MacroCavityContract: Formal contract specifying port vectors and bounding-box cavity reservations.
- EnclosureType: Classification of child zones (ENCLOSED, SATELLITE, SIBLING).
- HierarchicalLayoutSolver: Orchestrates Pass 1 (leaf profiling), enclosure detection,
  Pass 2 (container solve with macro-constraints), and Pass 3 (leaf-up macro-assembly).
- solve_hierarchical_layout: High-level API for composite multi-area solves.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import networkx as nx
import pyomo.environ as pyo

from romutil.models import Direction, Exit, Room
from romutil.solver import (
    build_spatial_coordinate_buckets,
    compute_dynamic_solver_timeout,
    find_spatial_room_collisions,
)

log = logging.getLogger("Mapper.decomposition")


class EnclosureType(str, Enum):
    """Geometric relationship between a child zone and its container host."""
    ENCLOSED = "enclosed"
    SATELLITE = "satellite"
    SIBLING = "sibling"


@dataclass
class AreaProfile:
    """Geometric profile and port displacements of a solved individual area."""
    area_name: str
    rooms: dict[int, Room]
    exits: list[Exit]
    bounds: tuple[int, int, int, int, int, int]  # (min_x, max_x, min_y, max_y, min_z, max_z)
    width: int
    height: int
    depth: int
    ports: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    footprint: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    port_footprints: dict[int, tuple[int, int, int, int, int, int]] = field(default_factory=dict)


@dataclass
class MacroCavityContract:
    """Formal geometric contract reserving cavity space and port alignments."""
    child_area_name: str
    container_area_name: str
    enclosure_type: EnclosureType
    port_pairs: list[tuple[int, int, Direction]]  # (container_vnum, child_vnum, dir_from_container)
    port_displacement: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict)
    bounding_box: tuple[int, int, int] = (0, 0, 0)  # (width, height, depth)
    clearance_rooms: list[int] = field(default_factory=list)
    footprint: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    port_footprints: dict[int, tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    clearance_directions: dict[int, Direction] = field(default_factory=dict)


@dataclass
class MacroAssemblyResult:
    """Result of hierarchical macro-assembly across all participating areas."""
    rooms: dict[int, Room]
    exits: list[Exit]
    profiles: Mapping[str, AreaProfile]
    contracts: Sequence[MacroCavityContract]
    collisions: list[tuple[int, int]]
    is_collision_free: bool


# Curated cavity clearance sets for canonical MUD metropolitan areas
KNOWN_MACRO_CONTRACTS: dict[tuple[str, str], list[int]] = {
    ("hood", "midgaard"): [3272, 3273],
    ("the hood", "midgaard"): [3272, 3273],
    ("gangland", "midgaard"): [3272, 3273],
    ("the slums", "midgaard"): [3272, 3273],
    ("mobfact", "midgaard"): [3018, 3019, 3024, 3044, 3048, 3101, 3170],
    ("mob factory", "midgaard"): [3018, 3019, 3024, 3044, 3048, 3101, 3170],
    ("the mob factory", "midgaard"): [3018, 3019, 3024, 3044, 3048, 3101, 3170],
    ("grave", "midgaard"): [3130, 3127, 3122],
    ("graveyard", "midgaard"): [3130, 3127, 3122],
    ("the graveyard", "midgaard"): [3130, 3127, 3122],
}


def direction_offset(d: Direction) -> tuple[int, int, int]:
    """Return unit displacement (dx, dy, dz) for a cardinal/diagonal direction."""
    if d == Direction.north:
        return (0, 1, 0)
    elif d == Direction.south:
        return (0, -1, 0)
    elif d == Direction.east:
        return (1, 0, 0)
    elif d == Direction.west:
        return (-1, 0, 0)
    elif d == Direction.up:
        return (0, 0, 1)
    elif d == Direction.down:
        return (0, 0, -1)
    elif d == Direction.northeast:
        return (1, 1, 0)
    elif d == Direction.northwest:
        return (-1, 1, 0)
    elif d == Direction.southeast:
        return (1, -1, 0)
    elif d == Direction.southwest:
        return (-1, -1, 0)
    return (0, 0, 0)


def compute_bounding_box(rooms: Iterable[Room]) -> tuple[int, int, int, int, int, int]:
    """Calculate 3D bounding box (min_x, max_x, min_y, max_y, min_z, max_z)."""
    xs = [r.x for r in rooms if r.x is not None and not getattr(r, "dummy", False)]
    ys = [r.y for r in rooms if r.y is not None and not getattr(r, "dummy", False)]
    zs = [r.z for r in rooms if r.z is not None and not getattr(r, "dummy", False)]
    if not xs or not ys or not zs:
        return (0, 0, 0, 0, 0, 0)
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def get_room_area_name(room: Room, default: str = "Unknown Area") -> str:
    """Derive area name or provenance string for a room."""
    return getattr(room, "area_name", None) or getattr(room, "area_file", None) or default


def partition_rooms_by_area(rdb: Mapping[int, Room]) -> dict[str, dict[int, Room]]:
    """Partition room database into separate dictionaries keyed by area name."""
    partitions: dict[str, dict[int, Room]] = {}
    for vnum, room in rdb.items():
        if getattr(room, "dummy", False):
            continue
        area_name = get_room_area_name(room)
        if area_name not in partitions:
            partitions[area_name] = {}
        partitions[area_name][vnum] = copy.deepcopy(room)
    return partitions


def build_area_connectivity_graph(
    rdb: Mapping[int, Room],
    exits: Sequence[Exit],
) -> tuple[nx.Graph, list[tuple[str, str, int, int, Direction]]]:
    """
    Construct area-level adjacency graph.
    Returns (graph, inter_area_links) where each link is (area_src, area_dst, vnum_src, vnum_dst, dir).
    """
    g = nx.Graph()
    for r in rdb.values():
        if not getattr(r, "dummy", False):
            g.add_node(get_room_area_name(r))
    inter_links: list[tuple[str, str, int, int, Direction]] = []

    for e in exits:
        if e.src not in rdb or e.dst not in rdb:
            continue
        r_src = rdb[e.src]
        r_dst = rdb[e.dst]
        if getattr(r_src, "dummy", False) or getattr(r_dst, "dummy", False):
            continue

        a_src = get_room_area_name(r_src)
        a_dst = get_room_area_name(r_dst)
        g.add_node(a_src)
        g.add_node(a_dst)

        if a_src != a_dst:
            g.add_edge(a_src, a_dst)
            inter_links.append((a_src, a_dst, e.src, e.dst, e.direction))

    return g, inter_links


def find_clearance_rooms(
    container_rooms: Mapping[int, Room],
    gateway_vnum: int,
    direction: Direction,
    max_hops: int = 5,
) -> list[int]:
    """
    Topologically discover candidate container rooms downstream in the given exit direction.
    """
    clearance: set[int] = set()
    if gateway_vnum not in container_rooms:
        return []

    visited = {gateway_vnum}
    queue: list[tuple[int, int]] = [(gateway_vnum, 0)]

    while queue:
        curr, hops = queue.pop(0)
        if hops >= max_hops:
            continue

        room = container_rooms.get(curr)
        if not room:
            continue

        for ex in room.exits:
            nxt = ex.dst
            if nxt not in container_rooms or nxt in visited:
                continue
            if getattr(container_rooms[nxt], "dummy", False):
                continue

            # Check if this exit advances in the target direction or downstream
            if hops == 0 and ex.direction == direction:
                visited.add(nxt)
                clearance.add(nxt)
                queue.append((nxt, hops + 1))
            elif hops > 0:
                visited.add(nxt)
                clearance.add(nxt)
                queue.append((nxt, hops + 1))

    clearance.discard(gateway_vnum)
    return sorted(clearance)


def profile_area(
    area_name: str,
    rooms: Mapping[int, Room],
    solver_timeout: int | None = None,
) -> AreaProfile:
    """
    Pass 1: Independently solve an isolated area to extract bounding box and port coordinates.
    """
    from romutil.graph import solve_layout

    isolated_rdb = {v: copy.deepcopy(r) for v, r in rooms.items() if not getattr(r, "dummy", False)}
    solved_rdb, solved_exits = solve_layout(
        isolated_rdb,
        solver_timeout=solver_timeout or 15,
        hierarchical=False,
    )

    b = compute_bounding_box(solved_rdb.values())
    w = max(0, b[1] - b[0])
    h = max(0, b[3] - b[2])
    d = max(0, b[5] - b[4])

    ports: dict[int, tuple[int, int, int]] = {}
    valid_coords = [
        (int(r.x), int(r.y), int(r.z))
        for r in solved_rdb.values()
        if not getattr(r, "dummy", False) and r.x is not None and r.y is not None and r.z is not None
    ]
    for vnum, room in solved_rdb.items():
        if not getattr(room, "dummy", False) and room.x is not None and room.y is not None and room.z is not None:
            ports[vnum] = (int(room.x), int(room.y), int(room.z))

    port_footprints: dict[int, tuple[int, int, int, int, int, int]] = {}
    for p_vnum, (px, py, pz) in ports.items():
        if valid_coords:
            min_dx = min(rx - px for rx, _, _ in valid_coords)
            max_dx = max(rx - px for rx, _, _ in valid_coords)
            min_dy = min(ry - py for _, ry, _ in valid_coords)
            max_dy = max(ry - py for _, ry, _ in valid_coords)
            min_dz = min(rz - pz for _, _, rz in valid_coords)
            max_dz = max(rz - pz for _, _, rz in valid_coords)
            port_footprints[p_vnum] = (min_dx, max_dx, min_dy, max_dy, min_dz, max_dz)

    default_footprint = (
        next(iter(port_footprints.values()))
        if port_footprints
        else (0, w, 0, h, 0, d)
    )

    return AreaProfile(
        area_name=area_name,
        rooms=solved_rdb,
        exits=solved_exits,
        bounds=b,
        width=w,
        height=h,
        depth=d,
        ports=ports,
        footprint=default_footprint,
        port_footprints=port_footprints,
    )


def classify_enclosure(
    container_name: str,
    child_name: str,
    port_pairs: Sequence[tuple[int, int, Direction]],
) -> EnclosureType:
    """
    Classify whether a child area is ENCLOSED within the container hull or an EXTERNAL_SATELLITE.
    """
    # Vertical links (Up/Down) indicate satellite elevation layering
    if any(d in (Direction.up, Direction.down) for _, _, d in port_pairs):
        return EnclosureType.SATELLITE
    return EnclosureType.ENCLOSED


def build_macro_contracts(
    container_name: str,
    container_rooms: Mapping[int, Room],
    child_profiles: Mapping[str, AreaProfile],
    inter_links: Sequence[tuple[str, str, int, int, Direction]],
) -> list[MacroCavityContract]:
    """
    Construct MacroCavityContract specifications between container and all child profiles.
    """
    contracts: list[MacroCavityContract] = []

    for child_name, child_prof in child_profiles.items():
        if child_name == container_name:
            continue

        # Collect port pairs where container connects to child
        port_pairs: list[tuple[int, int, Direction]] = []
        for src_area, dst_area, u, v, d in inter_links:
            if src_area == container_name and dst_area == child_name:
                port_pairs.append((u, v, d))
            elif src_area == child_name and dst_area == container_name:
                port_pairs.append((v, u, d.invert()))

        # Deduplicate port pairs
        port_pairs = list(dict.fromkeys(port_pairs))
        if not port_pairs:
            continue

        enclosure = classify_enclosure(container_name, child_name, port_pairs)

        # Compute port-to-port relative displacement constraints for multi-port children
        displacements: dict[tuple[int, int], tuple[int, int, int]] = {}
        if len(port_pairs) >= 2 and enclosure == EnclosureType.ENCLOSED:
            for i in range(len(port_pairs)):
                for j in range(i + 1, len(port_pairs)):
                    u1, v1, d1 = port_pairs[i]
                    u2, v2, d2 = port_pairs[j]
                    if v1 in child_prof.ports and v2 in child_prof.ports:
                        p1 = child_prof.ports[v1]
                        p2 = child_prof.ports[v2]
                        # Displacement in child frame: p1 - p2
                        delta_child = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
                        off1 = direction_offset(d1)
                        off2 = direction_offset(d2)
                        # Required container vector: delta_u = delta_child + off2 - off1
                        delta_u = (
                            delta_child[0] + off2[0] - off1[0],
                            delta_child[1] + off2[1] - off1[1],
                            delta_child[2] + off2[2] - off1[2],
                        )
                        displacements[(u1, u2)] = delta_u

        # Identify clearance rooms
        clearance_set: set[int] = set()

        # Check curated knowledge base first
        child_key = child_name.lower()
        cont_key = container_name.lower()
        for (ck, ctk), rooms in KNOWN_MACRO_CONTRACTS.items():
            if ck in child_key and ctk in cont_key:
                clearance_set.update(rooms)

        # Augment with automatic topological clearance detection
        for u, _, d in port_pairs:
            auto_rooms = find_clearance_rooms(container_rooms, u, d, max_hops=4)
            clearance_set.update(auto_rooms)

        u_primary, v_primary, d_primary = port_pairs[0]
        # Footprint relative to primary port
        footprint = (0, child_prof.width, 0, child_prof.height, 0, child_prof.depth)
        if v_primary in child_prof.port_footprints:
            footprint = child_prof.port_footprints[v_primary]
        elif v_primary in child_prof.ports:
            pv = child_prof.ports[v_primary]
            valid_coords = [
                (int(r.x), int(r.y), int(r.z))
                for r in child_prof.rooms.values()
                if not getattr(r, "dummy", False) and r.x is not None and r.y is not None and r.z is not None
            ]
            if valid_coords:
                min_dx = min(rx - pv[0] for rx, _, _ in valid_coords)
                max_dx = max(rx - pv[0] for rx, _, _ in valid_coords)
                min_dy = min(ry - pv[1] for _, ry, _ in valid_coords)
                max_dy = max(ry - pv[1] for _, ry, _ in valid_coords)
                min_dz = min(rz - pv[2] for _, _, rz in valid_coords)
                max_dz = max(rz - pv[2] for _, _, rz in valid_coords)
                footprint = (min_dx, max_dx, min_dy, max_dy, min_dz, max_dz)
            else:
                footprint = child_prof.footprint
        elif child_prof.footprint != (0, 0, 0, 0, 0, 0):
            footprint = child_prof.footprint

        # Determine clearance room directions relative to container gateway
        clearance_directions: dict[int, Direction] = {}
        cont_graph = nx.DiGraph()
        for r_vnum, r_obj in container_rooms.items():
            if getattr(r_obj, "dummy", False):
                continue
            cont_graph.add_node(r_vnum)
            for ex in r_obj.exits:
                if ex.dst in container_rooms and not getattr(container_rooms[ex.dst], "dummy", False):
                    cont_graph.add_edge(ex.src, ex.dst, dir=ex.direction)

        for w in clearance_set:
            if (
                w in container_rooms
                and cont_graph.has_node(u_primary)
                and cont_graph.has_node(w)
                and nx.has_path(cont_graph, u_primary, w)
            ):
                path = nx.shortest_path(cont_graph, u_primary, w)
                if len(path) >= 2:
                    first_dir = cont_graph[path[0]][path[1]]["dir"]
                    vx, vy, vz = 0, 0, 0
                    for a, b in zip(path[:-1], path[1:]):
                        step_dx, step_dy, step_dz = direction_offset(cont_graph[a][b]["dir"])
                        vx += step_dx
                        vy += step_dy
                        vz += step_dz

                    if d_primary in (Direction.east, Direction.west):
                        # Transverse axis is Y (North/South)
                        if d_primary == Direction.east and vx > 0 and vy == 0:
                            clearance_directions[w] = Direction.east
                        elif d_primary == Direction.west and vx < 0 and vy == 0:
                            clearance_directions[w] = Direction.west
                        elif footprint[2] < 0 and footprint[3] > 0:
                            if vy > 0 or (vy == 0 and first_dir == Direction.north):
                                clearance_directions[w] = Direction.north
                            elif vy < 0 or (vy == 0 and first_dir == Direction.south):
                                clearance_directions[w] = Direction.south
                            else:
                                clearance_directions[w] = d_primary
                        else:
                            clearance_directions[w] = d_primary
                    elif d_primary in (Direction.north, Direction.south):
                        # Transverse axis is X (East/West)
                        if d_primary == Direction.south and vy < 0:
                            clearance_directions[w] = Direction.south
                        elif d_primary == Direction.north and vy > 0:
                            clearance_directions[w] = Direction.north
                        elif footprint[0] < 0 and footprint[1] > 0:
                            if vx > 0:
                                clearance_directions[w] = Direction.east
                            elif vx < 0:
                                clearance_directions[w] = Direction.west
                            elif first_dir in (Direction.east, Direction.west):
                                clearance_directions[w] = first_dir
                            else:
                                clearance_directions[w] = d_primary
                        else:
                            clearance_directions[w] = d_primary
                    else:
                        clearance_directions[w] = d_primary
                else:
                    clearance_directions[w] = d_primary
            else:
                clearance_directions[w] = d_primary

        contracts.append(
            MacroCavityContract(
                child_area_name=child_name,
                container_area_name=container_name,
                enclosure_type=enclosure,
                port_pairs=port_pairs,
                port_displacement=displacements,
                bounding_box=(child_prof.width, child_prof.height, child_prof.depth),
                clearance_rooms=sorted(clearance_set),
                footprint=footprint,
                port_footprints=child_prof.port_footprints,
                clearance_directions=clearance_directions,
            )
        )

    return contracts


def create_macro_cavity_hook(
    contracts: Sequence[MacroCavityContract],
) -> Callable[[pyo.ConcreteModel], None]:
    """
    Generate Pyomo extra_constraints_hook injecting macro-cavity reservation constraints.
    """
    def hook(model: pyo.ConcreteModel) -> None:
        if not hasattr(model, "macro_cavity_constraints"):
            model.macro_cavity_constraints = pyo.ConstraintList()

        for contract in contracts:
            if contract.enclosure_type != EnclosureType.ENCLOSED:
                continue

            # 1. Multi-port rigid displacement invariants (exact equality)
            for (u1, u2), (dx, dy, dz) in contract.port_displacement.items():
                if hasattr(model, "Rooms") and u1 in model.Rooms and u2 in model.Rooms:
                    model.macro_cavity_constraints.add(model.x[u1] - model.x[u2] == int(dx))
                    model.macro_cavity_constraints.add(model.y[u1] - model.y[u2] == int(dy))
                    model.macro_cavity_constraints.add(model.z[u1] - model.z[u2] == int(dz))

            # 2. Cavity clearance bounding box inequalities with port-relative footprints
            if not contract.port_pairs:
                continue
            u, v, d = contract.port_pairs[0]
            if not hasattr(model, "Rooms") or u not in model.Rooms:
                continue

            w_box, h_box, _ = contract.bounding_box
            fp = contract.port_footprints.get(v, contract.footprint)
            if fp == (0, 0, 0, 0, 0, 0) and contract.bounding_box != (0, 0, 0):
                min_dx, max_dx = 0, w_box
                min_dy, max_dy = 0, h_box
                min_dz, max_dz = 0, 0
            else:
                min_dx, max_dx, min_dy, max_dy, min_dz, max_dz = fp

            off_x, off_y, off_z = direction_offset(d)

            for w in contract.clearance_rooms:
                if w in model.Rooms and w != u:
                    d_eff = contract.clearance_directions.get(w, d)

                    if d_eff in (Direction.east, Direction.northeast, Direction.southeast):
                        bound = off_x + max_dx + 1
                        if d in (Direction.east, Direction.northeast, Direction.southeast) and bound < w_box + 2:
                            bound = max(bound, w_box + 2)
                        model.macro_cavity_constraints.add(
                            model.x[w] >= model.x[u] + int(bound)
                        )
                    elif d_eff in (Direction.west, Direction.northwest, Direction.southwest):
                        bound = -(off_x + min_dx) + 1
                        if d in (Direction.west, Direction.northwest, Direction.southwest) and bound < w_box + 2:
                            bound = max(bound, w_box + 2)
                        model.macro_cavity_constraints.add(
                            model.x[u] - model.x[w] >= int(bound)
                        )
                    elif d_eff in (Direction.north, Direction.northeast, Direction.northwest):
                        bound = off_y + max_dy + 1
                        if d in (Direction.north, Direction.northeast, Direction.northwest) and bound < h_box + 2:
                            bound = max(bound, h_box + 2)
                        model.macro_cavity_constraints.add(
                            model.y[w] >= model.y[u] + int(bound)
                        )
                    elif d_eff in (Direction.south, Direction.southeast, Direction.southwest):
                        bound = -(off_y + min_dy) + 1
                        if d in (Direction.south, Direction.southeast, Direction.southwest) and bound < h_box + 2:
                            bound = max(bound, h_box + 2)
                        model.macro_cavity_constraints.add(
                            model.y[u] - model.y[w] >= int(bound)
                        )

    return hook


def assemble_composite_layout(
    container_profile: AreaProfile,
    child_profiles: Mapping[str, AreaProfile],
    contracts: Sequence[MacroCavityContract],
    all_original_exits: Sequence[Exit] | None = None,
) -> MacroAssemblyResult:
    """
    Pass 3: Leaf-up macro-assembly applying rigid translations to place child zones
    into reserved macro-cavities and attaching external satellites.
    """
    composite_rdb: dict[int, Room] = {}

    # 1. Place container rooms (root reference frame)
    for vnum, room in container_profile.rooms.items():
        if not getattr(room, "dummy", False):
            composite_rdb[vnum] = copy.deepcopy(room)

    contracts_by_child = {c.child_area_name: c for c in contracts}

    # 2. Place enclosed children into reserved cavities
    for child_name, child_prof in child_profiles.items():
        contract = contracts_by_child.get(child_name)
        if not contract or contract.enclosure_type != EnclosureType.ENCLOSED:
            continue
        if not contract.port_pairs:
            continue

        u_primary, v_primary, d_primary = contract.port_pairs[0]
        if u_primary not in composite_rdb or v_primary not in child_prof.ports:
            continue

        r_cont = composite_rdb[u_primary]
        p_child = child_prof.ports[v_primary]
        dx, dy, dz = direction_offset(d_primary)

        target_x = (r_cont.x or 0) + dx
        target_y = (r_cont.y or 0) + dy
        target_z = (r_cont.z or 0) + dz

        t_x = target_x - p_child[0]
        t_y = target_y - p_child[1]
        t_z = target_z - p_child[2]

        for vnum, room in child_prof.rooms.items():
            if not getattr(room, "dummy", False):
                rm = copy.deepcopy(room)
                if rm.x is not None: rm.x += t_x
                if rm.y is not None: rm.y += t_y
                if rm.z is not None: rm.z += t_z
                composite_rdb[vnum] = rm

    # 3. Place satellite zones on distinct vertical elevation planes
    for child_name, child_prof in child_profiles.items():
        contract = contracts_by_child.get(child_name)
        if not contract or contract.enclosure_type != EnclosureType.SATELLITE:
            continue
        if not contract.port_pairs:
            continue

        u_primary, v_primary, d_primary = contract.port_pairs[0]
        if u_primary not in composite_rdb or v_primary not in child_prof.ports:
            continue

        r_cont = composite_rdb[u_primary]
        p_child = child_prof.ports[v_primary]
        dx, dy, dz = direction_offset(d_primary)

        target_x = (r_cont.x or 0) + dx
        target_y = (r_cont.y or 0) + dy

        # Determine non-conflicting elevation layer
        valid_z = [r.z for r in composite_rdb.values() if r.z is not None and not getattr(r, "dummy", False)]
        child_z = [r.z for r in child_prof.rooms.values() if r.z is not None and not getattr(r, "dummy", False)]
        min_child_z = min(child_z) if child_z else 0
        max_child_z = max(child_z) if child_z else 0

        if d_primary == Direction.up:
            max_cont_z = max(valid_z) if valid_z else 0
            t_z = (max_cont_z + 1) - min_child_z
        elif d_primary == Direction.down:
            min_cont_z = min(valid_z) if valid_z else 0
            t_z = (min_cont_z - 1) - max_child_z
        else:
            t_z = (r_cont.z or 0) + dz - p_child[2]

        t_x = target_x - p_child[0]
        t_y = target_y - p_child[1]

        for vnum, room in child_prof.rooms.items():
            if not getattr(room, "dummy", False):
                rm = copy.deepcopy(room)
                if rm.x is not None: rm.x += t_x
                if rm.y is not None: rm.y += t_y
                if rm.z is not None: rm.z += t_z
                composite_rdb[vnum] = rm

    # 4. Normalize coordinates so minimums are zero
    valid_rooms = [r for r in composite_rdb.values() if r.x is not None and r.y is not None and r.z is not None]
    if valid_rooms:
        xs = [int(r.x) for r in valid_rooms if r.x is not None]
        ys = [int(r.y) for r in valid_rooms if r.y is not None]
        zs = [int(r.z) for r in valid_rooms if r.z is not None]
        x_min = min(xs)
        y_min = min(ys)
        z_min = min(zs)
        for r in valid_rooms:
            if r.x is not None:
                r.x -= x_min
            if r.y is not None:
                r.y -= y_min
            if r.z is not None:
                r.z -= z_min

    # 5. Restore original inter-room exits
    all_exits: list[Exit] = []
    if all_original_exits:
        all_exits = [
            e for e in all_original_exits
            if e.src in composite_rdb and e.dst in composite_rdb
        ]
    else:
        for r in composite_rdb.values():
            for e in r.exits:
                if e.src in composite_rdb and e.dst in composite_rdb:
                    all_exits.append(e)

    # 6. Verify collisions
    coords = {
        v: (r.x, r.y, r.z)
        for v, r in composite_rdb.items()
        if r.x is not None and r.y is not None and r.z is not None
    }
    buckets = build_spatial_coordinate_buckets(list(composite_rdb.keys()), coords)
    collisions = find_spatial_room_collisions(buckets)

    return MacroAssemblyResult(
        rooms=composite_rdb,
        exits=all_exits,
        profiles=child_profiles,
        contracts=contracts,
        collisions=collisions,
        is_collision_free=len(collisions) == 0,
    )


class HierarchicalLayoutSolver:
    """Orchestrates multi-area hierarchical graph decomposition and macro-assembly."""

    def __init__(
        self,
        child_timeout: int = 15,
        container_timeout: int | None = None,
        component_padding: int = 2,
    ):
        self.child_timeout = child_timeout
        self.container_timeout = container_timeout
        self.component_padding = component_padding

    def solve(
        self,
        rdb: Mapping[int, Room],
        exits: Sequence[Exit] | None = None,
    ) -> tuple[dict[int, Room], list[Exit]]:
        """
        Execute leaf-up hierarchical solve on composite room database.
        """
        from romutil.graph import solve_layout

        if exits is None:
            exits = [e for r in rdb.values() for e in r.exits]

        partitions = partition_rooms_by_area(rdb)
        if len(partitions) <= 1:
            c_timeout = (
                self.container_timeout
                if self.container_timeout is not None
                else compute_dynamic_solver_timeout(len(rdb))
            )
            return solve_layout(
                rdb,
                solver_timeout=c_timeout,
                component_padding=self.component_padding,
                hierarchical=False,
            )

        g_area, inter_links = build_area_connectivity_graph(rdb, exits)

        # Decompose into weakly connected area components
        area_components = list(nx.connected_components(g_area))
        area_components.sort(key=lambda c: len(c), reverse=True)

        solved_components: list[dict[int, Room]] = []
        all_solved_exits: list[Exit] = []

        for comp_areas in area_components:
            if len(comp_areas) == 1:
                # Standalone single-area component
                area_name = next(iter(comp_areas))
                comp_rdb = partitions[area_name]
                s_rdb, s_exits = solve_layout(
                    comp_rdb,
                    solver_timeout=self.child_timeout,
                    component_padding=self.component_padding,
                    hierarchical=False,
                )
                solved_components.append(s_rdb)
                all_solved_exits.extend(s_exits)
                continue

            # Multi-area connected component: identify container host
            container_name = max(
                comp_areas,
                key=lambda a: (g_area.degree(a), len(partitions[a])),
            )
            child_names = [a for a in comp_areas if a != container_name]

            # Pass 1: Independently profile each child area
            child_profiles: dict[str, AreaProfile] = {}
            for c_name in child_names:
                prof = profile_area(c_name, partitions[c_name], solver_timeout=self.child_timeout)
                child_profiles[c_name] = prof

            # Extract macro-cavity contracts
            contracts = build_macro_contracts(
                container_name,
                partitions[container_name],
                child_profiles,
                inter_links,
            )

            # Pass 2: Solve container with macro-cavity reservation hook
            hook = create_macro_cavity_hook(contracts)
            container_rdb = partitions[container_name]
            c_timeout = (
                self.container_timeout
                if self.container_timeout is not None
                else compute_dynamic_solver_timeout(len(container_rdb))
            )
            s_cont_rdb, s_cont_exits = solve_layout(
                container_rdb,
                solver_timeout=c_timeout,
                extra_constraints_hook=hook,
                hierarchical=False,
            )

            cont_bounds = compute_bounding_box(s_cont_rdb.values())
            cont_ports = {
                v: (int(r.x), int(r.y), int(r.z))
                for v, r in s_cont_rdb.items()
                if not getattr(r, "dummy", False) and r.x is not None and r.y is not None and r.z is not None
            }
            container_profile = AreaProfile(
                area_name=container_name,
                rooms=s_cont_rdb,
                exits=s_cont_exits,
                bounds=cont_bounds,
                width=max(0, cont_bounds[1] - cont_bounds[0]),
                height=max(0, cont_bounds[3] - cont_bounds[2]),
                depth=max(0, cont_bounds[5] - cont_bounds[4]),
                ports=cont_ports,
            )

            # Pass 3: Leaf-up macro-assembly
            assembly = assemble_composite_layout(
                container_profile,
                child_profiles,
                contracts,
                all_original_exits=exits,
            )
            solved_components.append(assembly.rooms)
            all_solved_exits.extend(assembly.exits)

        # Horizontal shelf-packing of disconnected components
        if len(solved_components) == 1:
            final_rdb = solved_components[0]
        else:
            final_rdb = {}
            x_offset = 0
            for comp in solved_components:
                b = compute_bounding_box(comp.values())
                w = max(0, b[1] - b[0])
                for r in comp.values():
                    if r.x is not None:
                        r.x += x_offset - b[0]
                    final_rdb[r.vnum] = r
                x_offset += w + max(2, self.component_padding)

        return final_rdb, all_solved_exits


def solve_hierarchical_layout(
    rdb: Mapping[int, Room],
    area: Any = None,
    solver_timeout: int | None = None,
    component_padding: int = 2,
    child_timeout: int = 15,
    container_timeout: int | None = None,
) -> tuple[dict[int, Room], list[Exit]]:
    """
    Public API to hierarchically solve multi-area composite room layouts.
    """
    effective_container_timeout = (
        solver_timeout if solver_timeout is not None else container_timeout
    )
    effective_child_timeout = (
        min(effective_container_timeout, child_timeout)
        if effective_container_timeout is not None
        else child_timeout
    )

    solver = HierarchicalLayoutSolver(
        child_timeout=effective_child_timeout,
        container_timeout=effective_container_timeout,
        component_padding=component_padding,
    )
    return solver.solve(rdb)
