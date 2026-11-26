"""Bottom-up tree traversal and MILP solving for hierarchical layout trees."""

from __future__ import annotations

import collections
from dataclasses import replace
from typing import Any, Collection, Mapping, Sequence

from romutil.layout.geometry import BoundingBox3D, decompose_into_orthogonal_sub_boxes
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeNodeType,
    TreeSolution,
)
from romutil.layout.milp.cluster import (
    UnresolvableCollisionError,
    compute_adaptive_cluster_timeout,
    solve_cluster,
    solve_cluster_with_cutting_planes,
)
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.formulation import formulate_cluster_model
from romutil.layout.milp.inventory import (
    _collect_extended_footprints,
    _expand_room_cells,
    get_child_boundary_ports,
    get_child_coordinate,
    get_child_strictly_interior_rooms,
    inventory_cluster,
)
from romutil.layout.milp.normalization import normalize_cluster_solution
from romutil.layout.milp.reactive_split import (
    compute_split_lp_signature,
    compute_total_edge_length,
    detect_planar_transverse_shear,
    diagnose_transverse_shear,
    diagnose_unresolvable_collision,
    generate_coordinate_threshold_partitions,
    get_child_interface_rooms,
    is_split_pair_origin,
    is_splittable_child,
    split_child_along_direction,
)
from romutil.layout.protocols import TreeSolver
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Direction, Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus

_MAX_DYNAMIC_BRIDGE_SPLIT_DEPTH = 4
_MAX_REACTIVE_SPLIT_ROUNDS = 8
_MAX_DIAGNOSTIC_RETRIES = 4


def _safe_displacement(direction: int | Direction) -> tuple[int, int, int] | None:
    """Return the 3D unit displacement for direction, or None if unsupported."""
    try:
        return direction_to_displacement(direction)
    except (ValueError, KeyError):
        return None


def _as_coord3d(raw: Any) -> Coord3D:
    """Normalize a Coord3D or 3-tuple offset into a Coord3D instance."""
    if isinstance(raw, Coord3D):
        return raw
    if isinstance(raw, (tuple, list)) and len(raw) == 3:
        return Coord3D(int(raw[0]), int(raw[1]), int(raw[2]))
    return Coord3D(0, 0, 0)


def _collect_subtree_rooms(node: SolveTreeNode) -> set[int]:
    """Return all room IDs owned by node and its descendants."""
    rooms = set(node.rooms)
    for child in node.children:
        rooms |= _collect_subtree_rooms(child)
    return rooms


def _ray_hits_box(
    origin: Coord3D,
    direction: tuple[int, int, int],
    box: tuple[int, int, int, int, int, int],
) -> bool:
    """Return True if the open planar cardinal ray from origin intersects box."""
    rx, ry, _rz = direction
    min_x, max_x, min_y, max_y, min_z, max_z = box
    if not (min_z <= origin.z <= max_z):
        return False
    if rx > 0:
        return max_x > origin.x and min_y <= origin.y <= max_y
    if rx < 0:
        return min_x < origin.x and min_y <= origin.y <= max_y
    if ry > 0:
        return max_y > origin.y and min_x <= origin.x <= max_x
    if ry < 0:
        return min_y < origin.y and min_x <= origin.x <= max_x
    return False


def _ray_hits_box_with_extent(
    origin: Coord3D,
    ext: tuple[int, int],
    direction: tuple[int, int, int],
    box: tuple[int, int, int, int, int, int],
) -> bool:
    """Return True if the open planar cardinal ray past anchor extent intersects box."""
    rx, ry, _rz = direction
    lo, hi = ext
    min_x, max_x, min_y, max_y, min_z, max_z = box
    if not (min_z <= origin.z <= max_z):
        return False
    if rx > 0:
        return max_x > origin.x + hi and max_y >= origin.y - lo and min_y <= origin.y + hi
    if rx < 0:
        return min_x < origin.x - lo and max_y >= origin.y - lo and min_y <= origin.y + hi
    if ry > 0:
        return max_y > origin.y + hi and max_x >= origin.x - lo and min_x <= origin.x + hi
    if ry < 0:
        return min_y < origin.y - lo and max_x >= origin.x - lo and min_x <= origin.x + hi
    return False


def _ray_continues_beyond_neighbor(
    anchor_vnum: int,
    direction: tuple[int, int, int],
    subtree_rooms: set[int],
    graph: LayoutGraph,
) -> bool:
    """Return True if an outward neighbor along direction continues further along the same ray."""
    for edge in graph.edges:
        if edge.cut:
            continue
        nbr: int | None = None
        if edge.src == anchor_vnum and edge.dst not in subtree_rooms:
            if _safe_displacement(edge.direction) == direction:
                nbr = edge.dst
        elif edge.dst == anchor_vnum and edge.src not in subtree_rooms:
            ed = _safe_displacement(edge.direction)
            if ed is not None and (-ed[0], -ed[1], -ed[2]) == direction:
                nbr = edge.src
        if nbr is not None and _has_uncut_outward_edge_in_direction(
            nbr, direction, subtree_rooms | {anchor_vnum}, graph
        ):
            return True
    return False


def _child_attachment_rooms(
    sub_child: SolveTreeNode,
    parent_rooms: Collection[int],
    graph: LayoutGraph,
) -> set[int]:
    """Return parent room IDs to which a nested child cluster attaches."""
    ports = get_child_boundary_ports(sub_child) & set(parent_rooms)
    if ports:
        return ports
    sub_rooms = _collect_subtree_rooms(sub_child)
    for edge in graph.edges:
        if edge.cut:
            continue
        if edge.src in parent_rooms and edge.dst in sub_rooms:
            ports.add(edge.src)
        elif edge.dst in parent_rooms and edge.src in sub_rooms:
            ports.add(edge.dst)
    return ports


def _has_uncut_outward_edge_in_direction(
    anchor_vnum: int,
    direction: tuple[int, int, int],
    subtree_rooms: set[int],
    graph: LayoutGraph,
) -> bool:
    """Return True if anchor_vnum has an uncut outward graph edge along direction outside subtree_rooms."""
    for edge in graph.edges:
        if edge.cut:
            continue
        if edge.src == anchor_vnum and edge.dst not in subtree_rooms:
            if _safe_displacement(edge.direction) == direction:
                return True
        elif edge.dst == anchor_vnum and edge.src not in subtree_rooms:
            ed = _safe_displacement(edge.direction)
            if ed is not None and (-ed[0], -ed[1], -ed[2]) == direction:
                return True
    return False


def _has_outward_exit_in_direction(
    anchor_vnum: int,
    direction: tuple[int, int, int],
    subtree_rooms: set[int],
    graph: LayoutGraph,
) -> bool:
    """Return True if anchor_vnum has an outward exit outside subtree_rooms along direction."""
    if _has_uncut_outward_edge_in_direction(anchor_vnum, direction, subtree_rooms, graph):
        return True

    if anchor_vnum in graph.rooms:
        for exit_def in graph.rooms[anchor_vnum].exits:
            if exit_def.dst_vnum not in subtree_rooms and _safe_displacement(exit_def.direction) == direction:
                return True
    return False


def _find_bridge_ray_blockers(
    node: SolveTreeNode,
    node_sol: NodeSolution,
    graph: LayoutGraph,
) -> tuple[int, int, tuple[int, int, int], set[int]] | None:
    """Detect whether a solved BridgeCut child's local geometry blocks its outward anchor ray."""
    contract = node.cut_contract
    if not isinstance(contract, BridgeCut) or contract.anchor_vnum is None:
        return None
    u, v = contract.bridge_edge
    if contract.anchor_vnum != u or v not in node.rooms:
        return None
    disp = _safe_displacement(contract.direction)
    if disp is None or disp[2] != 0 or sum(abs(c) for c in disp) != 1:
        return None

    rev_d = (-disp[0], -disp[1], -disp[2])
    if not _has_outward_exit_in_direction(u, rev_d, _collect_subtree_rooms(node), graph):
        return None

    origin = get_child_coordinate(node_sol, u)
    if origin is None:
        return None

    blockers: set[int] = set()
    for room_vnum in node.rooms:
        pt = node_sol.local_coordinates.get(room_vnum)
        if pt is not None and _ray_hits_box(origin, rev_d, (pt.x, pt.x, pt.y, pt.y, pt.z, pt.z)):
            blockers.add(room_vnum)

    for edge in graph.edges:
        if edge.cut:
            continue
        if edge.src in node.rooms and edge.dst in node.rooms:
            p1 = node_sol.local_coordinates.get(edge.src)
            p2 = node_sol.local_coordinates.get(edge.dst)
            if p1 is not None and p2 is not None:
                box = (
                    min(p1.x, p2.x),
                    max(p1.x, p2.x),
                    min(p1.y, p2.y),
                    max(p1.y, p2.y),
                    min(p1.z, p2.z),
                    max(p1.z, p2.z),
                )
                if _ray_hits_box(origin, rev_d, box):
                    blockers.add(edge.src)
                    blockers.add(edge.dst)

    child_offsets = node_sol.metadata.get("child_offsets", {})
    child_sub_boxes = node_sol.metadata.get("child_sub_boxes", {})
    for sc in node.children:
        off = _as_coord3d(child_offsets.get(sc.node_id))
        for sb in child_sub_boxes.get(sc.node_id, ()):
            box = (
                off.x + sb.min_x,
                off.x + sb.max_x,
                off.y + sb.min_y,
                off.y + sb.max_y,
                off.z + sb.min_z,
                off.z + sb.max_z,
            )
            if _ray_hits_box(origin, rev_d, box):
                blockers.update(_child_attachment_rooms(sc, node.rooms, graph))
                break

    return (u, v, rev_d, blockers) if blockers else None


def _try_detach_articulation_ray_blocking_subchild(
    child: SolveTreeNode,
    child_sol: NodeSolution,
    graph: LayoutGraph,
) -> tuple[SolveTreeNode, SolveTreeNode] | None:
    """Detach a nested BridgeCut sub-child whose solved sub-box blocks an anchor outward ray."""
    contract = child.cut_contract
    if not isinstance(contract, ArticulationPointCut) or not child.children:
        return None
    if len(child.children) == 1 and not child.rooms:
        return None

    anchor_vnum = contract.anchor_vnum
    origin = get_child_coordinate(child_sol, anchor_vnum)
    if origin is None:
        return None

    subtree_rooms = _collect_subtree_rooms(child)
    out_dirs = [
        direction
        for direction in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
        if _has_uncut_outward_edge_in_direction(anchor_vnum, direction, subtree_rooms, graph)
    ]
    if not out_dirs:
        return None

    sc_rooms = {sc.node_id: _collect_subtree_rooms(sc) for sc in child.children}
    room_to_sc: dict[int, str] = {}
    for sc in child.children:
        for room_vnum in sc_rooms[sc.node_id]:
            room_to_sc[room_vnum] = sc.node_id
    sibling_connected: set[str] = set()
    for edge in graph.edges:
        if edge.cut:
            continue
        c1 = room_to_sc.get(edge.src)
        c2 = room_to_sc.get(edge.dst)
        if c1 is not None and c2 is not None and c1 != c2:
            sibling_connected.add(c1)
            sibling_connected.add(c2)

    child_offsets = child_sol.metadata.get("child_offsets", {})
    child_sub_boxes = child_sol.metadata.get("child_sub_boxes", {})
    for sc in child.children:
        if not isinstance(sc.cut_contract, BridgeCut) or sc.cut_contract.anchor_vnum is None:
            continue
        if sc.node_id in sibling_connected:
            continue
        sc_disp = _safe_displacement(sc.cut_contract.direction)
        off = _as_coord3d(child_offsets.get(sc.node_id))
        for sb in child_sub_boxes.get(sc.node_id, ()):
            box = (
                off.x + sb.min_x,
                off.x + sb.max_x,
                off.y + sb.min_y,
                off.y + sb.max_y,
                off.z + sb.min_z,
                off.z + sb.max_z,
            )
            if any(
                _ray_hits_box(origin, direction, box)
                and not (
                    sc_disp is not None
                    and sc_disp[2] == 0
                    and (sc_disp[0] * direction[0] + sc_disp[1] * direction[1]) != 0
                    and _ray_continues_beyond_neighbor(anchor_vnum, direction, subtree_rooms, graph)
                )
                for direction in out_dirs
            ):
                base_node = SolveTreeNode(
                    node_id=f"{child.node_id}__base",
                    node_type=child.node_type,
                    rooms=child.rooms,
                    children=tuple(c for c in child.children if c.node_id != sc.node_id),
                    metadata=dict(child.metadata),
                )
                return base_node, sc
    return None


_RAY_ALLOWED_DIRS: dict[tuple[int, int, int], tuple[str, ...]] = {
    (1, 0, 0): ("l", "a", "b", "u", "d"),
    (-1, 0, 0): ("r", "a", "b", "u", "d"),
    (0, 1, 0): ("r", "l", "b", "u", "d"),
    (0, -1, 0): ("r", "l", "a", "u", "d"),
}


def _try_resolve_articulation_ray_blockage(
    node: SolveTreeNode,
    node_sol: NodeSolution,
    child_solutions_map: Mapping[str, NodeSolution],
    graph: LayoutGraph,
    options: LayoutOptions | None,
    solver_name: str = "cbc",
) -> NodeSolution | None:
    """Re-solve an ArticulationPointCut cluster with semi-infinite open-ray clearance."""
    contract = node.cut_contract
    if not isinstance(contract, ArticulationPointCut):
        return None
    if _try_detach_articulation_ray_blocking_subchild(node, node_sol, graph) is not None:
        return None

    anchor = contract.anchor_vnum
    origin = get_child_coordinate(node_sol, anchor)
    if origin is None:
        return None

    subtree_rooms = _collect_subtree_rooms(node)
    out_dirs = [
        d
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
        if _has_uncut_outward_edge_in_direction(anchor, d, subtree_rooms, graph)
    ]
    if not out_dirs:
        return None

    _cu, _rs, room_extents = _collect_extended_footprints(graph, options)
    anchor_ext = room_extents.get(anchor, (0, 0))
    child_offsets = node_sol.metadata.get("child_offsets", {})
    child_sub_boxes = node_sol.metadata.get("child_sub_boxes", {})

    blocked_dirs: list[tuple[int, int, int]] = []
    for d in out_dirs:
        is_blocked = False
        for r_vnum, pt in node_sol.local_coordinates.items():
            if r_vnum == anchor:
                continue
            r_lo, r_hi = room_extents.get(r_vnum, (0, 0))
            r_box = (pt.x - r_lo, pt.x + r_hi, pt.y - r_lo, pt.y + r_hi, pt.z, pt.z)
            if _ray_hits_box_with_extent(origin, anchor_ext, d, r_box):
                is_blocked = True
                break

        if not is_blocked:
            for edge in graph.edges:
                if edge.cut:
                    continue
                if (
                    edge.src in node.rooms
                    and edge.dst in node.rooms
                    and edge.src != anchor
                    and edge.dst != anchor
                ):
                    p1 = node_sol.local_coordinates.get(edge.src)
                    p2 = node_sol.local_coordinates.get(edge.dst)
                    if p1 is not None and p2 is not None:
                        e_box = (
                            min(p1.x, p2.x),
                            max(p1.x, p2.x),
                            min(p1.y, p2.y),
                            max(p1.y, p2.y),
                            min(p1.z, p2.z),
                            max(p1.z, p2.z),
                        )
                        if _ray_hits_box_with_extent(origin, anchor_ext, d, e_box):
                            is_blocked = True
                            break

        if not is_blocked:
            for cid, sbs in child_sub_boxes.items():
                off = _as_coord3d(child_offsets.get(cid))
                for sb in sbs:
                    sb_box = (
                        off.x + sb.min_x,
                        off.x + sb.max_x,
                        off.y + sb.min_y,
                        off.y + sb.max_y,
                        off.z + sb.min_z,
                        off.z + sb.max_z,
                    )
                    if _ray_hits_box_with_extent(origin, anchor_ext, d, sb_box):
                        is_blocked = True
                        break
                if is_blocked:
                    break

        if is_blocked:
            blocked_dirs.append(d)

    if not blocked_dirs:
        return None

    inv = inventory_cluster(
        node=node,
        child_solutions=child_solutions_map,
        graph=graph,
        options=options,
        boundary_ports=frozenset(get_child_boundary_ports(node)),
    )
    n_sub_boxes = sum(len(sbs) for sbs in inv.child_sub_boxes.values())
    t_cluster = compute_adaptive_cluster_timeout(node, options, n_sub_boxes=n_sub_boxes)
    form = formulate_cluster_model(inv, options)
    if anchor not in form.local_rooms:
        return None

    ax, ay, az = form.get_point(anchor)
    lo, hi = inv.room_extents.get(anchor, (0, 0))

    def get_ray_box(d: tuple[int, int, int]) -> tuple[Any, Any, Any, Any, Any, Any]:
        if d == (1, 0, 0):
            return (ax + hi + 1, form.m_coord, ay - lo, ay + hi, az, az)
        if d == (-1, 0, 0):
            return (-form.m_coord, ax - lo - 1, ay - lo, ay + hi, az, az)
        if d == (0, 1, 0):
            return (ax - lo, ax + hi, ay + hi + 1, form.m_coord, az, az)
        return (ax - lo, ax + hi, -form.m_coord, ay - lo - 1, az, az)

    room_pairs = [(u, d) for u in form.local_rooms if u != anchor for d in blocked_dirs]
    sb_pairs = [
        ((sc.node_id, k), d)
        for sc in inv.children
        for k in range(len(inv.child_sub_boxes.get(sc.node_id, ())))
        for d in blocked_dirs
    ]
    corr_pairs = [
        (idx, d)
        for idx, ae in enumerate(inv.active_edges)
        if anchor not in ae.associated_rooms
        and ae.src != anchor
        and ae.dst != anchor
        and sum(1 for c in ae.direction if c != 0) == 1
        for d in blocked_dirs
    ]

    add_disjunctive_box_separation(
        model=form.model,
        pairs=room_pairs,
        get_box_a=form.get_room_box,
        get_box_b=get_ray_box,
        prefix="ray_room",
        m_big=form.m_big,
        allowed_directions={pair: _RAY_ALLOWED_DIRS[pair[1]] for pair in room_pairs},
    )
    add_disjunctive_box_separation(
        model=form.model,
        pairs=sb_pairs,
        get_box_a=form.get_child_sub_box,
        get_box_b=get_ray_box,
        prefix="ray_sb",
        m_big=form.m_big,
        allowed_directions={pair: _RAY_ALLOWED_DIRS[pair[1]] for pair in sb_pairs},
    )
    add_disjunctive_box_separation(
        model=form.model,
        pairs=corr_pairs,
        get_box_a=lambda idx: form.get_corridor_envelope_box(inv.active_edges[idx], interior=False),
        get_box_b=get_ray_box,
        prefix="ray_corr",
        m_big=form.m_big,
        allowed_directions={pair: _RAY_ALLOWED_DIRS[pair[1]] for pair in corr_pairs},
    )

    try:
        extraction = solve_cluster_with_cutting_planes(
            formulation=form,
            node=node,
            options=options,
            solver_name=solver_name,
            total_cluster_budget=float(t_cluster),
        )
    except LayoutSolverError:
        return None

    return normalize_cluster_solution(
        extraction=extraction,
        node=node,
        children=inv.children,
        child_solutions=child_solutions_map,
        child_sub_boxes=inv.child_sub_boxes,
        active_edges=inv.active_edges,
        graph=graph,
        room_extents=inv.room_extents,
    )


def _try_split_ray_blocked_child(
    child: SolveTreeNode,
    child_sol: NodeSolution,
    graph: LayoutGraph,
) -> tuple[SolveTreeNode, SolveTreeNode] | None:
    """Split a ray-blocked BridgeCut child across an internal orthogonal planar bridge edge."""
    blocker_info = _find_bridge_ray_blockers(child, child_sol, graph)
    if blocker_info is None:
        return _try_detach_articulation_ray_blocking_subchild(child, child_sol, graph)

    _u, v, rev_d, blockers = blocker_info
    adj: dict[int, list[int]] = collections.defaultdict(list)
    edge_info: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] = {}
    for edge in graph.edges:
        if edge.cut:
            continue
        if edge.src in child.rooms and edge.dst in child.rooms:
            ed = _safe_displacement(edge.direction)
            if ed is None:
                continue
            dir_int = int(edge.direction)
            rev_dir = int(Direction(edge.direction).invert())
            adj[edge.src].append(edge.dst)
            adj[edge.dst].append(edge.src)
            weight = max(1, edge.weight)
            edge_info[(edge.src, edge.dst)] = (dir_int, ed, weight)
            edge_info[(edge.dst, edge.src)] = (rev_dir, (-ed[0], -ed[1], -ed[2]), weight)

    sc_attach = {
        sc.node_id: _child_attachment_rooms(sc, child.rooms, graph)
        for sc in child.children
    }
    sc_rooms = {sc.node_id: _collect_subtree_rooms(sc) for sc in child.children}
    sc_adj: dict[str, set[str]] = {sc.node_id: set() for sc in child.children}
    room_to_sc: dict[int, str] = {}
    for sc in child.children:
        for room_vnum in sc_rooms[sc.node_id]:
            room_to_sc[room_vnum] = sc.node_id
    for edge in graph.edges:
        if edge.cut:
            continue
        c1 = room_to_sc.get(edge.src)
        c2 = room_to_sc.get(edge.dst)
        if c1 is not None and c2 is not None and c1 != c2:
            sc_adj[c1].add(c2)
            sc_adj[c2].add(c1)
    sc_groups: list[tuple[list[SolveTreeNode], set[int]]] = []
    sc_by_id = {sc.node_id: sc for sc in child.children}
    seen_sc: set[str] = set()
    for sc in child.children:
        if sc.node_id in seen_sc:
            continue
        grp_ids: list[str] = []
        q_grp: collections.deque[str] = collections.deque([sc.node_id])
        seen_sc.add(sc.node_id)
        while q_grp:
            cur_id = q_grp.popleft()
            grp_ids.append(cur_id)
            for nxt_id in sorted(sc_adj[cur_id]):
                if nxt_id not in seen_sc:
                    seen_sc.add(nxt_id)
                    q_grp.append(nxt_id)
        grp_nodes = [sc_by_id[gid] for gid in grp_ids]
        grp_pts: set[int] = set()
        for gid in grp_ids:
            grp_pts.update(sc_attach[gid])
        sc_groups.append((grp_nodes, grp_pts))

    dist: dict[int, int] = {v: 0}
    queue: collections.deque[int] = collections.deque([v])
    while queue:
        curr = queue.popleft()
        for nxt in adj[curr]:
            if nxt not in dist:
                dist[nxt] = dist[curr] + 1
                queue.append(nxt)

    best: tuple[
        tuple[int, int, int, int],
        int,
        int,
        int,
        int,
        set[int],
        set[int],
        list[SolveTreeNode],
        list[SolveTreeNode],
    ] | None = None
    seen_undirected: set[tuple[int, int]] = set()

    for (raw_a, raw_b), _info in edge_info.items():
        undir_key = (min(raw_a, raw_b), max(raw_a, raw_b))
        if undir_key in seen_undirected:
            continue
        seen_undirected.add(undir_key)

        if raw_a not in dist or raw_b not in dist:
            continue
        a, b = (raw_a, raw_b) if dist[raw_a] <= dist[raw_b] else (raw_b, raw_a)
        dir_int, ed, weight = edge_info[(a, b)]
        # Candidate cut must be planar cardinal and orthogonal to the blocked ray
        if ed[2] != 0 or sum(abs(c) for c in ed) != 1 or (ed[0] * rev_d[0] + ed[1] * rev_d[1]) != 0:
            continue

        side_v: set[int] = {v}
        q_bridge: collections.deque[int] = collections.deque([v])
        while q_bridge:
            curr = q_bridge.popleft()
            for nxt in adj[curr]:
                if (curr == a and nxt == b) or (curr == b and nxt == a):
                    continue
                if nxt not in side_v:
                    side_v.add(nxt)
                    q_bridge.append(nxt)
        if b in side_v:
            continue

        side_other = set(child.rooms) - side_v
        valid_partition = True
        base_children: list[SolveTreeNode] = []
        wing_children: list[SolveTreeNode] = []
        for grp_nodes, pts in sc_groups:
            if (pts & side_v) and (pts & side_other):
                valid_partition = False
                break
            if pts & side_other:
                wing_children.extend(grp_nodes)
            else:
                base_children.extend(grp_nodes)
        if not valid_partition:
            continue

        separated_blockers = len(blockers & side_other)
        if separated_blockers == 0:
            continue

        score = (-separated_blockers, dist[a], a, b)
        if best is None or score < best[0]:
            best = (score, a, b, dir_int, weight, side_v, side_other, base_children, wing_children)

    if best is None:
        return None

    _score, a, b, dir_int, weight, side_v, side_other, base_children, wing_children = best
    base_node = SolveTreeNode(
        node_id=f"{child.node_id}__base",
        node_type=child.node_type,
        rooms=frozenset(side_v),
        children=tuple(base_children),
        metadata=dict(child.metadata),
    )
    wing_contract = BridgeCut(bridge_edge=(a, b), direction=dir_int, weight=weight, anchor_vnum=a)
    wing_meta: dict[str, Any] = {
        "cut_type": CutType.BRIDGE,
        "anchor_vnum": a,
        "bridge_edge": (a, b),
        "direction": dir_int,
        "weight": weight,
        "cut_contract": wing_contract,
    }
    wing_severed = sorted(
        {
            tuple(edge)
            for wc in wing_children
            for edge in wc.metadata.get("severed_bridge_edges", ())
        }
    )
    if wing_severed:
        wing_meta["severed_bridge_edges"] = tuple(wing_severed)
    wing_node = SolveTreeNode(
        node_id=f"{child.node_id}__wing",
        node_type=child.node_type,
        rooms=frozenset(side_other),
        children=tuple(wing_children),
        metadata=wing_meta,
    )
    return base_node, wing_node


def _extract_status(node_sol: NodeSolution) -> SolverStatus:
    """Normalize a NodeSolution status metadata entry to SolverStatus."""
    raw = node_sol.metadata.get("status", SolverStatus.FEASIBLE)
    if isinstance(raw, SolverStatus):
        return raw
    if isinstance(raw, str):
        try:
            return SolverStatus(raw.lower())
        except ValueError:
            return SolverStatus.FEASIBLE
    return SolverStatus.FEASIBLE


def _resynthesize_split_child_solution(
    orig_child: SolveTreeNode,
    pieces: Sequence[tuple[SolveTreeNode, NodeSolution]],
    parent_sol: NodeSolution,
    solutions: Mapping[str, NodeSolution],
    room_extents: Mapping[int, tuple[int, int]] | None = None,
    *,
    unpacked_ids: Collection[str] = (),
    graph: LayoutGraph | None = None,
    descendant_solutions_out: dict[str, NodeSolution] | None = None,
) -> tuple[NodeSolution, Coord3D, tuple[BoundingBox3D, ...]]:
    """Recombine dynamically split sub-pieces back into orig_child and unpacked descendants."""
    parent_offsets = parent_sol.metadata.get("child_offsets", {})

    abs_room_coords: dict[int, Coord3D] = {}
    sub_offsets_in_parent: dict[str, Coord3D] = {}
    sub_boxes_map: dict[str, tuple[BoundingBox3D, ...]] = {}
    all_coords_in_parent: dict[Any, Coord3D] = {}
    port_in_parent: dict[int, Coord3D] = {}

    orig_child_ids = {c.node_id for c in orig_child.children}
    orig_subtree_rooms = _collect_subtree_rooms(orig_child)
    parent_sub_boxes = parent_sol.metadata.get("child_sub_boxes", {})
    unpacked_set = set(unpacked_ids) | {orig_child.node_id}

    for p_node, p_sol in pieces:
        p_off = _as_coord3d(parent_offsets.get(p_node.node_id))
        if (p_node.node_id in orig_child_ids and not unpacked_ids) or (
            p_node.node_id in solutions and p_node.node_id not in unpacked_set
        ):
            sub_offsets_in_parent[p_node.node_id] = p_off
            if p_node.node_id in parent_sub_boxes:
                sub_boxes_map[p_node.node_id] = tuple(parent_sub_boxes[p_node.node_id])
            for room_vnum, pt in p_sol.local_coordinates.items():
                translated = Coord3D(p_off.x + pt.x, p_off.y + pt.y, p_off.z + pt.z)
                if room_vnum in orig_subtree_rooms:
                    abs_room_coords.setdefault(room_vnum, translated)
                if room_vnum not in orig_child.rooms:
                    all_coords_in_parent.setdefault(room_vnum, translated)
            for key, pt in p_sol.metadata.get("all_room_coordinates", {}).items():
                if key not in orig_child.rooms:
                    all_coords_in_parent.setdefault(
                        key, Coord3D(p_off.x + pt.x, p_off.y + pt.y, p_off.z + pt.z)
                    )
            continue

        for port_vnum, pt in p_sol.port_coordinates.items():
            port_in_parent[port_vnum] = Coord3D(p_off.x + pt.x, p_off.y + pt.y, p_off.z + pt.z)

        p_ports = set(p_sol.port_coordinates.keys())
        for key, pt in p_sol.metadata.get("all_room_coordinates", {}).items():
            if key in p_ports and key not in p_sol.local_coordinates:
                continue
            all_coords_in_parent[key] = Coord3D(p_off.x + pt.x, p_off.y + pt.y, p_off.z + pt.z)

        for room_vnum, pt in p_sol.local_coordinates.items():
            translated = Coord3D(p_off.x + pt.x, p_off.y + pt.y, p_off.z + pt.z)
            abs_room_coords[room_vnum] = translated
            all_coords_in_parent[room_vnum] = translated

        p_child_offsets = p_sol.metadata.get("child_offsets", {})
        p_child_sbs = p_sol.metadata.get("child_sub_boxes", {})
        for sc in p_node.children:
            sc_off = _as_coord3d(p_child_offsets.get(sc.node_id))
            sc_abs = Coord3D(p_off.x + sc_off.x, p_off.y + sc_off.y, p_off.z + sc_off.z)
            sub_offsets_in_parent[sc.node_id] = sc_abs
            if sc.node_id in p_child_sbs:
                sub_boxes_map[sc.node_id] = tuple(p_child_sbs[sc.node_id])
            sc_sol = solutions.get(sc.node_id)
            if sc_sol is not None:
                for rv, pt in sc_sol.local_coordinates.items():
                    abs_room_coords.setdefault(rv, Coord3D(sc_abs.x + pt.x, sc_abs.y + pt.y, sc_abs.z + pt.z))

    def _add_corridor_cells(a: int, b: int, disp: tuple[int, int, int]) -> None:
        pa = abs_room_coords.get(a) or all_coords_in_parent.get(a)
        pb = abs_room_coords.get(b) or all_coords_in_parent.get(b)
        if pa is None or pb is None:
            return
        dx, dy, dz = disp
        dist = abs(pb.x - pa.x) + abs(pb.y - pa.y) + abs(pb.z - pa.z)
        if (
            dist > 1
            and pb.x - pa.x == dist * dx
            and pb.y - pa.y == dist * dy
            and pb.z - pa.z == dist * dz
        ):
            if dz == 0:
                ext_a = (room_extents or {}).get(a, (0, 0))
                ext_b = (room_extents or {}).get(b, (0, 0))
                pos_dir = (dx > 0) or (dy > 0)
                t_start = (ext_a[1] + 1) if pos_dir else (ext_a[0] + 1)
                t_stop = dist - (ext_b[0] if pos_dir else ext_b[1])
            else:
                t_start, t_stop = 1, dist
            for step in range(t_start, t_stop):
                all_coords_in_parent.setdefault(
                    f"corridor_{a}_{b}_{step}",
                    Coord3D(pa.x + step * dx, pa.y + step * dy, pa.z + step * dz),
                )

    # Include stretched corridor cells along inter-piece split bridges
    for p_node, _p_sol in pieces:
        contract = p_node.cut_contract
        if not isinstance(contract, BridgeCut):
            continue
        disp = _safe_displacement(contract.direction)
        if disp is not None:
            _add_corridor_cells(contract.bridge_edge[0], contract.bridge_edge[1], disp)

    # Include stretched corridor cells along severed inter-piece edges (e.g. directed parallel cuts)
    if graph is not None:
        severed_pairs: set[frozenset[int]] = set()
        for p_node, _p_sol in pieces:
            for e_raw in p_node.metadata.get("severed_bridge_edges", ()):
                severed_pairs.add(frozenset((int(e_raw[0]), int(e_raw[1]))))
        if severed_pairs:
            for edge in graph.edges:
                if edge.cut:
                    continue
                if frozenset({edge.src, edge.dst}) in severed_pairs:
                    if edge.src in orig_subtree_rooms and edge.dst in orig_subtree_rooms:
                        disp = _safe_displacement(edge.direction)
                        if disp is not None:
                            _add_corridor_cells(edge.src, edge.dst, disp)

    statuses = [_extract_status(p_sol) for _p_node, p_sol in pieces]
    statuses.append(_extract_status(parent_sol))
    if any(s == SolverStatus.TIMEOUT for s in statuses):
        merged_status = SolverStatus.TIMEOUT
    elif all(s == SolverStatus.OPTIMAL for s in statuses):
        merged_status = SolverStatus.OPTIMAL
    else:
        merged_status = SolverStatus.FEASIBLE

    extents = room_extents or {}

    def _project_node(
        cur_node: SolveTreeNode,
    ) -> tuple[NodeSolution, Coord3D, tuple[BoundingBox3D, ...]]:
        if cur_node.node_id not in unpacked_set:
            c_sol = solutions[cur_node.node_id]
            c_off = sub_offsets_in_parent.get(cur_node.node_id, Coord3D(0, 0, 0))
            c_sbs = sub_boxes_map.get(cur_node.node_id, c_sol.sub_boxes)
            return c_sol, c_off, c_sbs

        cur_local_in_parent = {
            r: abs_room_coords[r]
            for r in cur_node.rooms
            if r in abs_room_coords
        }
        cur_all_in_parent: dict[Any, Coord3D] = dict(cur_local_in_parent)
        cur_sub_offsets_in_parent: dict[str, Coord3D] = {}
        cur_sub_boxes_map: dict[str, tuple[BoundingBox3D, ...]] = {}
        child_projected: list[tuple[SolveTreeNode, NodeSolution, Coord3D]] = []

        for ch in cur_node.children:
            ch_sol, ch_off, ch_sbs = _project_node(ch)
            child_projected.append((ch, ch_sol, ch_off))
            cur_sub_offsets_in_parent[ch.node_id] = ch_off
            if ch_sbs:
                cur_sub_boxes_map[ch.node_id] = tuple(ch_sbs)
            elif ch.node_id in sub_boxes_map:
                cur_sub_boxes_map[ch.node_id] = sub_boxes_map[ch.node_id]
            for rv, pt in ch_sol.local_coordinates.items():
                if rv not in cur_node.rooms:
                    cur_all_in_parent.setdefault(
                        rv, Coord3D(ch_off.x + pt.x, ch_off.y + pt.y, ch_off.z + pt.z)
                    )
            for key, pt in ch_sol.metadata.get("all_room_coordinates", {}).items():
                if key not in cur_node.rooms:
                    cur_all_in_parent.setdefault(
                        key, Coord3D(ch_off.x + pt.x, ch_off.y + pt.y, ch_off.z + pt.z)
                    )

        if cur_node.node_id == orig_child.node_id:
            for key, pt in all_coords_in_parent.items():
                cur_all_in_parent.setdefault(key, pt)
        else:
            cur_sub_and_ports = _collect_subtree_rooms(cur_node) | get_child_boundary_ports(cur_node)
            for key, pt in all_coords_in_parent.items():
                if isinstance(key, int) and key in cur_sub_and_ports:
                    cur_all_in_parent.setdefault(key, pt)

        bound_pts: list[Coord3D] = list(cur_local_in_parent.values()) + list(cur_all_in_parent.values())
        for _ch, ch_sol, ch_off in child_projected:
            if ch_sol.bounding_box is not None:
                bb = ch_sol.bounding_box
                bound_pts.append(Coord3D(ch_off.x + bb.min_x, ch_off.y + bb.min_y, ch_off.z + bb.min_z))
                bound_pts.append(Coord3D(ch_off.x + bb.max_x, ch_off.y + bb.max_y, ch_off.z + bb.max_z))

        if bound_pts:
            node_origin = Coord3D(
                min(pt.x for pt in bound_pts),
                min(pt.y for pt in bound_pts),
                min(pt.z for pt in bound_pts),
            )
        else:
            node_origin = Coord3D(0, 0, 0)

        def _shift(pt: Coord3D) -> Coord3D:
            return Coord3D(pt.x - node_origin.x, pt.y - node_origin.y, pt.z - node_origin.z)

        merged_local = {r: _shift(pt) for r, pt in cur_local_in_parent.items()}
        merged_sub_offsets = {cid: _shift(pt) for cid, pt in cur_sub_offsets_in_parent.items()}
        merged_all = {k: _shift(pt) for k, pt in cur_all_in_parent.items()}

        merged_ports: dict[int, Coord3D] = {}
        for bp in get_child_boundary_ports(cur_node):
            raw_pt = (
                abs_room_coords.get(bp)
                or parent_sol.local_coordinates.get(bp)
                or parent_sol.port_coordinates.get(bp)
                or port_in_parent.get(bp)
                or cur_all_in_parent.get(bp)
                or all_coords_in_parent.get(bp)
            )
            if raw_pt is not None:
                merged_ports[bp] = _shift(raw_pt)

        bp_coord_set: set[Coord3D] = set()
        for bp, pt in merged_ports.items():
            bp_coord_set.update(_expand_room_cells(pt, extents.get(bp, (0, 0))))

        interior_cells: set[Coord3D] = set()
        for key, pt in merged_all.items():
            if key in merged_ports:
                continue
            ext = extents.get(key, (0, 0)) if isinstance(key, int) else (0, 0)
            for cell in _expand_room_cells(pt, ext):
                if cell not in bp_coord_set:
                    interior_cells.add(cell)
        if interior_cells:
            node_sub_boxes = decompose_into_orthogonal_sub_boxes(interior_cells)
        elif merged_all:
            node_sub_boxes = (BoundingBox3D.from_coordinates(merged_all.values()),)
        else:
            node_sub_boxes = ()

        node_sol = NodeSolution(
            node_id=cur_node.node_id,
            local_coordinates=merged_local,
            port_coordinates=merged_ports,
            sub_boxes=(),
            metadata={
                "child_offsets": merged_sub_offsets,
                "status": merged_status,
                "child_sub_boxes": cur_sub_boxes_map,
                "all_room_coordinates": merged_all,
                "port_coordinates": merged_ports,
            },
        )
        if cur_node.node_id != orig_child.node_id and descendant_solutions_out is not None:
            descendant_solutions_out[cur_node.node_id] = node_sol
        return node_sol, node_origin, node_sub_boxes

    return _project_node(orig_child)


class HierarchicalTreeSolver(TreeSolver):
    """Hierarchical tree solver that solves tree nodes bottom-up via MILP."""

    def __init__(
        self,
        graph: LayoutGraph | None = None,
        solver_name: str = "cbc",
    ) -> None:
        self.graph = graph
        self.solver_name = solver_name

    def solve_tree(
        self,
        tree: SolveTree,
        options: LayoutOptions | None = None,
    ) -> TreeSolution:
        """Recursively solve tree nodes bottom-up (post-order traversal)."""
        layout_graph = self.graph if self.graph is not None else tree.graph
        if layout_graph is None:
            all_rooms: set[int] = set()

            def _collect_rooms(n: SolveTreeNode) -> None:
                all_rooms.update(n.rooms)
                for ch in n.children:
                    _collect_rooms(ch)

            _collect_rooms(tree.root)
            for comp in tree.components:
                _collect_rooms(comp)

            layout_graph = LayoutGraph(
                rooms={v: Room(vnum=v, name=f"Room {v}", description="", exits=()) for v in all_rooms},
                edges=(),
            )
        if layout_graph.coord_unit is None and layout_graph.rooms:
            resolved_unit, _scales, _extents = _collect_extended_footprints(layout_graph, options)
            layout_graph = replace(layout_graph, coord_unit=resolved_unit)
        _cu, _rs, resolved_extents = _collect_extended_footprints(layout_graph, options)

        solutions: dict[str, NodeSolution] = {}

        def _expand_child(
            child_node: SolveTreeNode,
            child_sol: NodeSolution,
            g: LayoutGraph,
            depth: int = 0,
        ) -> list[tuple[SolveTreeNode, NodeSolution]]:
            if depth >= _MAX_DYNAMIC_BRIDGE_SPLIT_DEPTH:
                return [(child_node, child_sol)]
            split_pair = _try_split_ray_blocked_child(child_node, child_sol, g)
            if split_pair is None:
                return [(child_node, child_sol)]
            base_node, wing_node = split_pair
            base_sol = solve_cluster(
                node=base_node,
                child_solutions={
                    sc.node_id: solutions[sc.node_id]
                    for sc in base_node.children
                    if sc.node_id in solutions
                },
                graph=g,
                options=options,
                solver_name=self.solver_name,
            )
            if wing_node.node_id in solutions:
                return _expand_child(base_node, base_sol, g, depth + 1) + [
                    (wing_node, solutions[wing_node.node_id])
                ]
            wing_sol = solve_cluster(
                node=wing_node,
                child_solutions={
                    sc.node_id: solutions[sc.node_id]
                    for sc in wing_node.children
                    if sc.node_id in solutions
                },
                graph=g,
                options=options,
                solver_name=self.solver_name,
            )
            return _expand_child(base_node, base_sol, g, depth + 1) + _expand_child(
                wing_node, wing_sol, g, depth + 1
            )

        def _solve_node(
            node: SolveTreeNode,
            g: LayoutGraph,
            *,
            cache_solution: bool = True,
        ) -> NodeSolution:
            if node.node_id in solutions:
                return solutions[node.node_id]

            # Post-order traversal: solve all children first
            child_solutions_map: dict[str, NodeSolution] = {}
            for child in node.children:
                child_sol = _solve_node(child, g)
                child_solutions_map[child.node_id] = child_sol

            def _maybe_resolve_rays(
                target_node: SolveTreeNode,
                sol: NodeSolution,
                c_sols: Mapping[str, NodeSolution],
            ) -> NodeSolution:
                if isinstance(target_node.cut_contract, ArticulationPointCut):
                    resolved = _try_resolve_articulation_ray_blockage(
                        node=target_node,
                        node_sol=sol,
                        child_solutions_map=c_sols,
                        graph=g,
                        options=options,
                        solver_name=self.solver_name,
                    )
                    if resolved is not None:
                        return resolved
                return sol

            if node.node_type == TreeNodeType.WORLD_ROOT or not node.children:
                cluster_sol = solve_cluster(
                    node=node,
                    child_solutions=child_solutions_map,
                    graph=g,
                    options=options,
                    solver_name=self.solver_name,
                )
                cluster_sol = _maybe_resolve_rays(node, cluster_sol, child_solutions_map)
                if cache_solution:
                    solutions[node.node_id] = cluster_sol
                return cluster_sol

            effective_children: list[SolveTreeNode] = []
            effective_child_solutions: dict[str, NodeSolution] = {}
            split_map: dict[str, tuple[SolveTreeNode, list[tuple[SolveTreeNode, NodeSolution]]]] = {}

            for child in node.children:
                pieces = _expand_child(child, child_solutions_map[child.node_id], g)
                if len(pieces) > 1:
                    split_map[child.node_id] = (child, pieces)
                for p_node, p_sol in pieces:
                    effective_children.append(p_node)
                    effective_child_solutions[p_node.node_id] = p_sol

            def _has_any_splittable(
                inv: Any,
                cur_children: Sequence[SolveTreeNode],
                cur_child_sols: Mapping[str, NodeSolution],
            ) -> bool:
                return any(
                    is_splittable_child(c, inv, cur_child_sols)
                    for c in cur_children
                )

            has_split_candidate = any(is_split_pair_origin(c) for c in effective_children)
            reactive_unpacked_ids: dict[str, set[str]] = collections.defaultdict(set)
            init_inv = (
                inventory_cluster(
                    node=replace(node, children=tuple(effective_children)) if split_map else node,
                    child_solutions=effective_child_solutions,
                    graph=g,
                    options=options,
                )
                if has_split_candidate
                else None
            )

            if init_inv is None or not _has_any_splittable(
                init_inv, effective_children, effective_child_solutions
            ):
                target_node = (
                    replace(node, children=tuple(effective_children)) if split_map else node
                )
                target_sols = effective_child_solutions if split_map else child_solutions_map
                cluster_sol = solve_cluster(
                    node=target_node,
                    child_solutions=target_sols,
                    graph=g,
                    options=options,
                    solver_name=self.solver_name,
                )
                cluster_sol = _maybe_resolve_rays(target_node, cluster_sol, target_sols)
                if not split_map:
                    if cache_solution:
                        solutions[node.node_id] = cluster_sol
                    return cluster_sol
            else:
                orig_child_by_id = {c.node_id: c for c in node.children}
                split_seq = 0
                attempted_splits: set[
                    tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, int, int]]
                ] = set()
                failed_splits: set[tuple[str, int, int]] = set()

                def _compute_parent_attach_rooms(
                    cur_children: Sequence[SolveTreeNode],
                    cur_child_sols: Mapping[str, NodeSolution],
                ) -> set[int]:
                    b_ports = frozenset(get_child_boundary_ports(node))
                    strictly_int: set[int] = set()
                    for c in cur_children:
                        strictly_int.update(
                            get_child_strictly_interior_rooms(c, cur_child_sols.get(c.node_id))
                        )
                    ext_ports = {p for p in b_ports if p not in strictly_int}
                    return set(node.rooms) | ext_ports

                def _apply_reactive_split(
                    target_cid: str,
                    s_rooms: set[int],
                    t_rooms: set[int],
                    direction: tuple[int, int, int],
                ) -> bool:
                    nonlocal split_seq
                    sig = (
                        target_cid,
                        tuple(sorted(s_rooms)),
                        tuple(sorted(t_rooms)),
                        direction,
                    )
                    if sig in attempted_splits:
                        return False
                    attempted_splits.add(sig)

                    target_idx = next(
                        (i for i, c in enumerate(effective_children) if c.node_id == target_cid),
                        None,
                    )
                    if target_idx is None:
                        return False
                    target_child = effective_children[target_idx]
                    parent_attach = _compute_parent_attach_rooms(
                        effective_children, effective_child_solutions
                    )
                    split_res = split_child_along_direction(
                        child=target_child,
                        s_rooms=s_rooms,
                        t_rooms=t_rooms,
                        direction=direction,
                        parent_attach_rooms=parent_attach,
                        graph=g,
                        split_tag=f"rs{split_seq}",
                    )
                    if split_res is None:
                        return False
                    split_seq += 1
                    piece_s, piece_t, unpacked_nodes = split_res

                    try:
                        s_sol = _solve_node(piece_s, g, cache_solution=False)
                        t_sol = _solve_node(piece_t, g, cache_solution=False)
                    except LayoutSolverError:
                        return False

                    orig_root_id = str(
                        target_child.metadata.get("reactive_split_origin")
                        or target_child.node_id
                    )
                    new_pieces: list[tuple[SolveTreeNode, NodeSolution]] = []
                    for p_n, p_s in ((piece_s, s_sol), (piece_t, t_sol)):
                        expanded = _expand_child(p_n, p_s, g)
                        for ep_n, ep_s in expanded:
                            if ep_n.metadata.get("reactive_split_origin") != orig_root_id:
                                ep_meta = dict(ep_n.metadata)
                                ep_meta["reactive_split_origin"] = orig_root_id
                                ep_n = replace(ep_n, metadata=ep_meta)
                            new_pieces.append((ep_n, ep_s))

                    effective_child_solutions.pop(target_cid, None)
                    updated_list = (
                        effective_children[:target_idx]
                        + [pn for pn, _ in new_pieces]
                        + effective_children[target_idx + 1 :]
                    )
                    effective_children[:] = updated_list
                    for pn, ps in new_pieces:
                        effective_child_solutions[pn.node_id] = ps

                    for desc in unpacked_nodes:
                        reactive_unpacked_ids[orig_root_id].add(desc.node_id)
                    return True

                def _record_failed_split(
                    target_cid: str,
                    s_rooms: set[int],
                    t_rooms: set[int],
                    u_dir: tuple[int, int, int],
                    inv_ctx: Any,
                ) -> None:
                    target_child = next(
                        (c for c in effective_children if c.node_id == target_cid),
                        None,
                    )
                    if target_child is None:
                        return
                    iface, _ = get_child_interface_rooms(
                        target_child, inv_ctx, effective_child_solutions
                    )
                    lp_sig = compute_split_lp_signature(iface, s_rooms, t_rooms, u_dir)
                    if lp_sig is not None:
                        failed_splits.add((target_cid, lp_sig[0], lp_sig[1]))

                def _try_apply_split_request(
                    target_cid: str,
                    s_rooms: set[int],
                    t_rooms: set[int],
                    u_dir: tuple[int, int, int],
                    inv_ctx: Any,
                    *,
                    preserve_lp_sign: bool = True,
                ) -> bool:
                    if _apply_reactive_split(target_cid, s_rooms, t_rooms, u_dir):
                        return True
                    target_child = next(
                        (c for c in effective_children if c.node_id == target_cid),
                        None,
                    )
                    if target_child is not None:
                        iface, _ = get_child_interface_rooms(
                            target_child, inv_ctx, effective_child_solutions
                        )
                        orig_sig = compute_split_lp_signature(iface, s_rooms, t_rooms, u_dir)
                        alt_parts = generate_coordinate_threshold_partitions(
                            effective_child_solutions.get(target_cid),
                            s_rooms,
                            t_rooms,
                            u_dir,
                        )
                        for cand_s, cand_t, cand_dir in alt_parts:
                            if preserve_lp_sign:
                                if cand_dir != u_dir:
                                    continue
                                if (
                                    orig_sig is not None
                                    and compute_split_lp_signature(
                                        iface, cand_s, cand_t, cand_dir
                                    )
                                    != orig_sig
                                ):
                                    continue
                            if _apply_reactive_split(target_cid, cand_s, cand_t, cand_dir):
                                return True
                    _record_failed_split(target_cid, s_rooms, t_rooms, u_dir, inv_ctx)
                    return False

                best_valid_state: tuple[
                    NodeSolution,
                    list[SolveTreeNode],
                    dict[str, NodeSolution],
                    dict[str, set[str]],
                    tuple[int, int, int],
                ] | None = None
                cluster_sol_opt: NodeSolution | None = None

                def _restore_best_valid_state() -> NodeSolution:
                    assert best_valid_state is not None
                    (
                        restored_sol,
                        restored_children,
                        restored_child_sols,
                        restored_unpacked,
                        _,
                    ) = best_valid_state
                    effective_children[:] = restored_children
                    effective_child_solutions.clear()
                    effective_child_solutions.update(restored_child_sols)
                    reactive_unpacked_ids.clear()
                    for orig_k, desc_set in restored_unpacked.items():
                        reactive_unpacked_ids[orig_k] = set(desc_set)
                    return restored_sol

                for round_idx in range(_MAX_REACTIVE_SPLIT_ROUNDS + 1):
                    effective_node = replace(node, children=tuple(effective_children))
                    inv_round = (
                        init_inv
                        if round_idx == 0
                        else inventory_cluster(
                            node=effective_node,
                            child_solutions=effective_child_solutions,
                            graph=g,
                            options=options,
                        )
                    )
                    can_split_more = (
                        round_idx < _MAX_REACTIVE_SPLIT_ROUNDS
                        and _has_any_splittable(
                            inv_round, effective_children, effective_child_solutions
                        )
                    )

                    try:
                        solved_this_round = solve_cluster(
                            node=effective_node,
                            child_solutions=effective_child_solutions,
                            graph=g,
                            options=options,
                            solver_name=self.solver_name,
                        )
                    except UnresolvableCollisionError as exc:
                        if can_split_more:
                            inv_err = exc.inventory or inv_round
                            candidates = diagnose_unresolvable_collision(
                                exc,
                                effective_children,
                                inv_err,
                                effective_child_solutions,
                                g,
                            )
                            if any(
                                _apply_reactive_split(cid, s_rooms, t_rooms, u_dir)
                                for cid, s_rooms, t_rooms, u_dir in candidates
                            ):
                                continue
                        if best_valid_state is not None:
                            cluster_sol_opt = _restore_best_valid_state()
                            break
                        raise
                    except LayoutSolverError:
                        if best_valid_state is not None:
                            cluster_sol_opt = _restore_best_valid_state()
                            break
                        raise

                    inv_post = inventory_cluster(
                        node=effective_node,
                        child_solutions=effective_child_solutions,
                        graph=g,
                        options=options,
                    )
                    sheared = detect_planar_transverse_shear(
                        cluster_sol=solved_this_round,
                        inventory=inv_post,
                        effective_children=effective_children,
                        child_solutions=effective_child_solutions,
                    )
                    score = (
                        len(sheared),
                        sum(abs(v) for _ae, _ax, v in sheared),
                        compute_total_edge_length(
                            solved_this_round, inv_post, effective_child_solutions
                        ),
                    )
                    if best_valid_state is None or score < best_valid_state[4]:
                        best_valid_state = (
                            solved_this_round,
                            list(effective_children),
                            dict(effective_child_solutions),
                            {k: set(v) for k, v in reactive_unpacked_ids.items()},
                            score,
                        )

                    if not sheared or round_idx >= _MAX_REACTIVE_SPLIT_ROUNDS:
                        if best_valid_state is not None and best_valid_state[4] <= score:
                            cluster_sol_opt = _restore_best_valid_state()
                        else:
                            cluster_sol_opt = solved_this_round
                        break

                    split_ok = False
                    for _diag_attempt in range(_MAX_DIAGNOSTIC_RETRIES):
                        shear_candidates = diagnose_transverse_shear(
                            cluster_sol=solved_this_round,
                            inventory=inv_post,
                            effective_children=effective_children,
                            child_solutions=effective_child_solutions,
                            solver_name=self.solver_name,
                            graph=g,
                            failed_splits=failed_splits,
                        )
                        if not shear_candidates:
                            break
                        for cid, s_rooms, t_rooms, u_dir in shear_candidates:
                            if _try_apply_split_request(
                                cid,
                                s_rooms,
                                t_rooms,
                                u_dir,
                                inv_post,
                                preserve_lp_sign=True,
                            ):
                                split_ok = True
                                break
                        if split_ok:
                            break
                    if not split_ok:
                        cluster_sol_opt = _restore_best_valid_state()
                        break

                assert cluster_sol_opt is not None
                effective_node = replace(node, children=tuple(effective_children))
                cluster_sol = _maybe_resolve_rays(
                    effective_node, cluster_sol_opt, effective_child_solutions
                )

                reactive_pieces_by_orig: dict[str, list[tuple[SolveTreeNode, NodeSolution]]] = (
                    collections.defaultdict(list)
                )
                for p_n in effective_children:
                    orig_root_id = p_n.metadata.get("reactive_split_origin")
                    if orig_root_id is not None and orig_root_id in orig_child_by_id:
                        reactive_pieces_by_orig[str(orig_root_id)].append(
                            (p_n, effective_child_solutions[p_n.node_id])
                        )

                for orig_cid, r_pieces in reactive_pieces_by_orig.items():
                    split_map[orig_cid] = (orig_child_by_id[orig_cid], r_pieces)

                if not split_map:
                    if cache_solution:
                        solutions[node.node_id] = cluster_sol
                    return cluster_sol

            updated_meta = dict(cluster_sol.metadata)
            updated_offsets = dict(updated_meta.get("child_offsets", {}))
            updated_sub_boxes = dict(updated_meta.get("child_sub_boxes", {}))

            for orig_cid, (orig_child, pieces) in split_map.items():
                desc_out: dict[str, NodeSolution] = {}
                merged_sol, orig_offset, orig_sub_boxes = _resynthesize_split_child_solution(
                    orig_child=orig_child,
                    pieces=pieces,
                    parent_sol=cluster_sol,
                    solutions=solutions,
                    room_extents=resolved_extents,
                    unpacked_ids=reactive_unpacked_ids.get(orig_cid, ()),
                    graph=g,
                    descendant_solutions_out=desc_out,
                )
                solutions[orig_cid] = merged_sol
                solutions.update(desc_out)
                for p_node, _p_sol in pieces:
                    updated_offsets.pop(p_node.node_id, None)
                    updated_sub_boxes.pop(p_node.node_id, None)
                updated_offsets[orig_cid] = orig_offset
                updated_sub_boxes[orig_cid] = orig_sub_boxes

            updated_meta["child_offsets"] = updated_offsets
            updated_meta["child_sub_boxes"] = updated_sub_boxes
            resynthesized_parent = replace(cluster_sol, metadata=updated_meta)
            if cache_solution:
                solutions[node.node_id] = resynthesized_parent
            return resynthesized_parent

        _solve_node(tree.root, layout_graph)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp, layout_graph)

        if options and options.debug_solve_dir:
            from romutil.layout.milp.debug import render_debug_solve_tree_index

            render_debug_solve_tree_index(tree, options.debug_solve_dir)

        return TreeSolution(node_solutions=solutions)


__all__ = ["HierarchicalTreeSolver"]
