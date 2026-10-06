"""MILP formulation and solver for parent container clusters and gateway alignment."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.geometry import (
    BoundingBox3D,
    Segment3D,
    point_in_segment_interior,
    segment_box_penetration,
)
from romutil.layout.ir import (
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.milp.leaf import solve_leaf_cluster
from romutil.layout.protocols import TreeSolver
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Room
from romutil.types import Coord3D, LayoutOptions


def _get_child_bbox_corners(
    bbox: tuple[Coord3D, Coord3D] | BoundingBox3D | None,
) -> tuple[int, int, int, int, int, int] | None:
    """Extract (min_x, max_x, min_y, max_y, min_z, max_z) integer boundaries from a bounding box."""
    if bbox is None:
        return None
    if isinstance(bbox, BoundingBox3D):
        return (bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, bbox.min_z, bbox.max_z)
    min_pt, max_pt = bbox
    return (min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z)


def _boxes_intersect(
    b1: tuple[int, int, int, int, int, int],
    b2: tuple[int, int, int, int, int, int],
) -> bool:
    """Check if two axis-aligned 3D boxes intersect (closed boxes)."""
    return not (
        b1[1] < b2[0]
        or b1[0] > b2[1]
        or b1[3] < b2[2]
        or b1[2] > b2[3]
        or b1[5] < b2[4]
        or b1[4] > b2[5]
    )


def _point_in_box_bounds(
    pt: tuple[int, int, int],
    box: tuple[int, int, int, int, int, int],
) -> bool:
    """Check if a 3D point is inside an axis-aligned bounding box."""
    return (
        box[0] <= pt[0] <= box[1]
        and box[2] <= pt[1] <= box[3]
        and box[4] <= pt[2] <= box[5]
    )


def _solve_parent_fallback(
    node: SolveTreeNode,
    child_solutions: Mapping[str, NodeSolution],
    graph: LayoutGraph,
) -> NodeSolution:
    """Deterministic fallback placement for parent container and child macro-boxes."""
    rooms = sorted(node.rooms)
    children = list(node.children)

    if not rooms and not children:
        return NodeSolution(
            node_id=node.node_id,
            local_coordinates={},
            bounding_box=None,
            metadata={"child_offsets": {}},
        )

    # 1. Place parent rooms if present
    parent_coords: dict[int, tuple[int, int, int]] = {}
    if rooms:
        room_set = set(rooms)
        parent_edges: list[tuple[LayoutEdge, tuple[int, int, int]]] = []
        for e in graph.edges:
            if not e.cut and e.src in room_set and e.dst in room_set:
                try:
                    disp = direction_to_displacement(e.direction)
                    parent_edges.append((e, disp))
                except ValueError:
                    continue

        adj: dict[int, list[tuple[int, tuple[int, int, int], int, bool]]] = {u: [] for u in rooms}
        for e, disp in parent_edges:
            w = max(1, e.weight)
            adj[e.src].append((e.dst, disp, w, False))
            adj[e.dst].append((e.src, disp, w, True))

        occupied_rooms: set[tuple[int, int, int]] = set()
        for r in rooms:
            if r in parent_coords:
                continue
            if not parent_coords:
                curr_pos = (0, 0, 0)
            else:
                max_x = max(p[0] for p in occupied_rooms)
                curr_pos = (max_x + 2, 0, 0)

            parent_coords[r] = curr_pos
            occupied_rooms.add(curr_pos)

            queue: deque[int] = deque([r])
            while queue:
                curr = queue.popleft()
                cx, cy, cz = parent_coords[curr]
                for nbr, (dx, dy, dz), w, is_rev in adj[curr]:
                    if nbr not in parent_coords:
                        step_x, step_y, step_z = (-dx, -dy, -dz) if is_rev else (dx, dy, dz)
                        cand = (cx + step_x * w, cy + step_y * w, cz + step_z * w)
                        while cand in occupied_rooms:
                            cand = (cand[0] + step_x, cand[1] + step_y, cand[2] + step_z)
                        parent_coords[nbr] = cand
                        occupied_rooms.add(cand)
                        queue.append(nbr)

    # 2. Place child macro-boxes
    placed_boxes: list[tuple[str, tuple[int, int, int, int, int, int]]] = []
    raw_child_offsets: dict[str, tuple[int, int, int]] = {}

    cut_type = node.metadata.get("cut_type")
    poles = tuple(node.metadata.get("poles", ()))

    for child in children:
        c_sol = child_solutions.get(child.node_id)
        if c_sol is None:
            raw_child_offsets[child.node_id] = (0, 0, 0)
            continue

        c_coords = c_sol.local_coordinates
        c_bbox = _get_child_bbox_corners(c_sol.bounding_box)

        step_dir = (1, 0, 0)
        target_pos: tuple[int, int, int] | None = None

        # Check 2-cut split pair alignment
        if (
            cut_type == "split_pair"
            and len(poles) == 2
            and all(p in parent_coords and p in c_coords for p in poles)
        ):
            p0 = poles[0]
            target_pos = (
                parent_coords[p0][0] - c_coords[p0].x,
                parent_coords[p0][1] - c_coords[p0].y,
                parent_coords[p0][2] - c_coords[p0].z,
            )

        # Check 1-cut articulation anchor alignment
        if target_pos is None:
            a = child.metadata.get("anchor_vnum")
            if a is None:
                common = set(parent_coords.keys()) & set(c_coords.keys())
                if len(common) == 1:
                    a = next(iter(common))
            if a is not None and a in parent_coords and a in c_coords:
                target_pos = (
                    parent_coords[a][0] - c_coords[a].x,
                    parent_coords[a][1] - c_coords[a].y,
                    parent_coords[a][2] - c_coords[a].z,
                )

        doorway_info: tuple[int, int, tuple[int, int, int]] | None = None

        # Check doorway / bridge edge alignment
        if target_pos is None and parent_coords:
            for edge in graph.edges:
                if edge.cut:
                    continue
                if edge.src in parent_coords and edge.dst in c_coords:
                    try:
                        dx, dy, dz = direction_to_displacement(edge.direction)
                        w = max(1, edge.weight)
                        pu = parent_coords[edge.src]
                        cv = c_coords[edge.dst]
                        target_pos = (pu[0] + w * dx - cv.x, pu[1] + w * dy - cv.y, pu[2] + w * dz - cv.z)
                        step_dir = (dx, dy, dz) if (dx, dy, dz) != (0, 0, 0) else (1, 0, 0)
                        doorway_info = (edge.src, edge.dst, (dx, dy, dz))
                        break
                    except ValueError:
                        pass
                elif edge.dst in parent_coords and edge.src in c_coords:
                    try:
                        dx, dy, dz = direction_to_displacement(edge.direction)
                        w = max(1, edge.weight)
                        pu = parent_coords[edge.dst]
                        cv = c_coords[edge.src]
                        target_pos = (pu[0] - w * dx - cv.x, pu[1] - w * dy - cv.y, pu[2] - w * dz - cv.z)
                        step_dir = (-dx, -dy, -dz) if (-dx, -dy, -dz) != (0, 0, 0) else (1, 0, 0)
                        doorway_info = (edge.dst, edge.src, (-dx, -dy, -dz))
                        break
                    except ValueError:
                        pass

        # Unconnected child or parent with 0 rooms
        if target_pos is None:
            if not placed_boxes:
                target_pos = (0, 0, 0)
            else:
                max_x = max(b[1] for _, b in placed_boxes)
                c_min_x = c_bbox[0] if c_bbox else 0
                target_pos = (max_x + 2 - c_min_x, 0, 0)

        cx, cy, cz = target_pos

        # Ensure macro-box separation and corridor clearance against already placed disjoint children
        if c_bbox is not None:
            curr_box = (
                cx + c_bbox[0],
                cx + c_bbox[1],
                cy + c_bbox[2],
                cy + c_bbox[3],
                cz + c_bbox[4],
                cz + c_bbox[5],
            )
            collided = True
            step_count = 0
            while collided:
                collided = False
                for prev_id, prev_box in placed_boxes:
                    prev_sol = child_solutions.get(prev_id)
                    prev_coords = prev_sol.local_coordinates if prev_sol else {}
                    shared = set(c_coords.keys()) & set(prev_coords.keys())
                    if not shared and _boxes_intersect(curr_box, prev_box):
                        collided = True
                        break

                if not collided and doorway_info is not None:
                    u_src, v_dst, _ = doorway_info
                    pu = parent_coords[u_src]
                    cv = c_coords[v_dst]
                    pv = (cx + cv.x, cy + cv.y, cz + cv.z)
                    seg = Segment3D(Coord3D(*pu), Coord3D(*pv))

                    for pr, pr_pos in parent_coords.items():
                        if pr != u_src and pr not in c_coords:
                            if point_in_segment_interior(Coord3D(*pr_pos), seg):
                                collided = True
                                break

                    if not collided:
                        for prev_id, prev_box in placed_boxes:
                            prev_sol = child_solutions.get(prev_id)
                            prev_coords = prev_sol.local_coordinates if prev_sol else {}
                            shared = set(c_coords.keys()) & set(prev_coords.keys())
                            if not shared and u_src not in prev_coords:
                                pbox = BoundingBox3D(
                                    prev_box[0], prev_box[1], prev_box[2], prev_box[3], prev_box[4], prev_box[5]
                                )
                                if segment_box_penetration(seg, pbox):
                                    collided = True
                                    break

                if collided:
                    step_count += 1
                    if step_count > 50:
                        step_x = 0 if step_dir[0] != 0 else 1
                        step_y = 1 if step_dir[1] == 0 else 0
                        step_z = 0
                    else:
                        step_x = step_dir[0] if step_dir[0] != 0 else 1
                        step_y = step_dir[1]
                        step_z = step_dir[2]
                    cx += step_x
                    cy += step_y
                    cz += step_z
                    curr_box = (
                        cx + c_bbox[0],
                        cx + c_bbox[1],
                        cy + c_bbox[2],
                        cy + c_bbox[3],
                        cz + c_bbox[4],
                        cz + c_bbox[5],
                    )
            placed_boxes.append((child.node_id, curr_box))

        raw_child_offsets[child.node_id] = (cx, cy, cz)

    # 3. Normalization: shift all coordinates and offsets so min corner is (0, 0, 0)
    all_pts: list[tuple[int, int, int]] = list(parent_coords.values())
    for child in children:
        c_id = child.node_id
        cx, cy, cz = raw_child_offsets.get(c_id, (0, 0, 0))
        c_sol = child_solutions.get(c_id)
        c_bbox = _get_child_bbox_corners(c_sol.bounding_box) if c_sol else None
        if c_bbox:
            all_pts.append((cx + c_bbox[0], cy + c_bbox[2], cz + c_bbox[4]))
            all_pts.append((cx + c_bbox[1], cy + c_bbox[3], cz + c_bbox[5]))
        else:
            all_pts.append((cx, cy, cz))

    min_x = min(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    min_z = min(p[2] for p in all_pts)

    norm_parent_coords = {
        u: Coord3D(x - min_x, y - min_y, z - min_z)
        for u, (x, y, z) in parent_coords.items()
    }
    norm_child_offsets = {
        c_id: Coord3D(x - min_x, y - min_y, z - min_z)
        for c_id, (x, y, z) in raw_child_offsets.items()
    }

    shifted_pts = [Coord3D(p[0] - min_x, p[1] - min_y, p[2] - min_z) for p in all_pts]
    enc = BoundingBox3D.from_coordinates(shifted_pts)
    parent_bbox = (
        Coord3D(enc.min_x, enc.min_y, enc.min_z),
        Coord3D(enc.max_x, enc.max_y, enc.max_z),
    )

    return NodeSolution(
        node_id=node.node_id,
        local_coordinates=norm_parent_coords,
        bounding_box=parent_bbox,
        metadata={"child_offsets": norm_child_offsets},
    )


def solve_parent_cluster(
    node: SolveTreeNode,
    child_solutions: Mapping[str, NodeSolution],
    graph: LayoutGraph,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
) -> NodeSolution:
    """Solve parent container cluster embedding child macro-boxes via MILP.

    Args:
        node: Parent SolveTreeNode containing rooms and children.
        child_solutions: Mapping of child node_id to already-solved NodeSolution.
        graph: Underlying LayoutGraph with active edges.
        options: Optional layout configuration parameters.
        solver_name: Pyomo MILP solver name (defaults to 'cbc').

    Returns:
        NodeSolution containing parent local coordinates, child offsets in metadata, and bounding box.
    """
    rooms = sorted(node.rooms)
    children = list(node.children)

    if not rooms and not children:
        return NodeSolution(
            node_id=node.node_id,
            local_coordinates={},
            bounding_box=None,
            metadata={"child_offsets": {}},
        )

    if not children:
        sol = solve_leaf_cluster(node, graph, options, solver_name=solver_name)
        return NodeSolution(
            node_id=sol.node_id,
            local_coordinates=sol.local_coordinates,
            bounding_box=sol.bounding_box,
            metadata={"child_offsets": {}},
        )

    try:
        solver = pyo.SolverFactory(solver_name)
        if not solver.available():
            return _solve_parent_fallback(node, child_solutions, graph)
    except Exception:
        return _solve_parent_fallback(node, child_solutions, graph)

    try:
        model = pyo.ConcreteModel()
        m_coord = max(1000, 100 * (len(rooms) + len(children) * 10))
        m_big = 2 * m_coord

        # Variables: parent room coordinates
        if rooms:
            model.x = pyo.Var(rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))
            model.y = pyo.Var(rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))
            model.z = pyo.Var(rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))

            anchor = rooms[0]
            model.x[anchor].fix(0)
            model.y[anchor].fix(0)
            model.z[anchor].fix(0)

        # Variables: child macro-box origins
        child_ids = [c.node_id for c in children]
        model.cx = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.cy = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.cz = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord))

        if not rooms and child_ids:
            first_c = child_ids[0]
            model.cx[first_c].fix(0)
            model.cy[first_c].fix(0)
            model.cz[first_c].fix(0)

        # Internal parent edges
        room_set = set(rooms)
        internal_edges: list[tuple[LayoutEdge, tuple[int, int, int]]] = []
        if rooms:
            for e in graph.edges:
                if not e.cut and e.src in room_set and e.dst in room_set:
                    try:
                        disp = direction_to_displacement(e.direction)
                        internal_edges.append((e, disp))
                    except ValueError:
                        continue

        int_edge_indices = list(range(len(internal_edges)))
        if int_edge_indices:
            model.L_int = pyo.Var(
                int_edge_indices,
                domain=pyo.Integers,
                bounds=lambda m, i: (max(1, internal_edges[i][0].weight), m_coord),
            )
            model.int_x = pyo.ConstraintList()
            model.int_y = pyo.ConstraintList()
            model.int_z = pyo.ConstraintList()
            for i, (e, (dx, dy, dz)) in enumerate(internal_edges):
                model.int_x.add(model.x[e.dst] - model.x[e.src] == model.L_int[i] * dx)
                model.int_y.add(model.y[e.dst] - model.y[e.src] == model.L_int[i] * dy)
                model.int_z.add(model.z[e.dst] - model.z[e.src] == model.L_int[i] * dz)

        # Parent room non-overlap separation
        if len(rooms) > 1:
            parent_pairs = [
                (rooms[i], rooms[j])
                for i in range(len(rooms))
                for j in range(i + 1, len(rooms))
            ]
            model.pr_bl = pyo.Var(parent_pairs, domain=pyo.Binary)
            model.pr_br = pyo.Var(parent_pairs, domain=pyo.Binary)
            model.pr_bb = pyo.Var(parent_pairs, domain=pyo.Binary)
            model.pr_ba = pyo.Var(parent_pairs, domain=pyo.Binary)
            model.pr_bd = pyo.Var(parent_pairs, domain=pyo.Binary)
            model.pr_bu = pyo.Var(parent_pairs, domain=pyo.Binary)

            model.pr_sep_sum = pyo.ConstraintList()
            model.pr_sep_r = pyo.ConstraintList()
            model.pr_sep_l = pyo.ConstraintList()
            model.pr_sep_a = pyo.ConstraintList()
            model.pr_sep_b = pyo.ConstraintList()
            model.pr_sep_u = pyo.ConstraintList()
            model.pr_sep_d = pyo.ConstraintList()

            for u, v in parent_pairs:
                model.pr_sep_sum.add(
                    model.pr_bl[u, v]
                    + model.pr_br[u, v]
                    + model.pr_bb[u, v]
                    + model.pr_ba[u, v]
                    + model.pr_bd[u, v]
                    + model.pr_bu[u, v]
                    >= 1
                )
                model.pr_sep_r.add(model.x[u] - model.x[v] >= 1 - 2 * m_big * (1 - model.pr_br[u, v]))
                model.pr_sep_l.add(model.x[v] - model.x[u] >= 1 - 2 * m_big * (1 - model.pr_bl[u, v]))
                model.pr_sep_a.add(model.y[u] - model.y[v] >= 1 - 2 * m_big * (1 - model.pr_ba[u, v]))
                model.pr_sep_b.add(model.y[v] - model.y[u] >= 1 - 2 * m_big * (1 - model.pr_bb[u, v]))
                model.pr_sep_u.add(model.z[u] - model.z[v] >= 1 - 2 * m_big * (1 - model.pr_bu[u, v]))
                model.pr_sep_d.add(model.z[v] - model.z[u] >= 1 - 2 * m_big * (1 - model.pr_bd[u, v]))

        # Gateway alignments (1-cut and 2-cut)
        gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in children}
        model.align_x = pyo.ConstraintList()
        model.align_y = pyo.ConstraintList()
        model.align_z = pyo.ConstraintList()

        cut_type = node.metadata.get("cut_type")
        poles = tuple(node.metadata.get("poles", ()))
        is_split_pair = (
            cut_type == "split_pair"
            and len(poles) == 2
            and all(p in room_set for p in poles)
        )

        aligned_children: set[str] = set()

        if is_split_pair:
            for child in children:
                c_sol = child_solutions.get(child.node_id)
                if c_sol and all(p in c_sol.local_coordinates for p in poles):
                    for p in poles:
                        pc = c_sol.local_coordinates[p]
                        model.align_x.add(model.x[p] == model.cx[child.node_id] + pc.x)
                        model.align_y.add(model.y[p] == model.cy[child.node_id] + pc.y)
                        model.align_z.add(model.z[p] == model.cz[child.node_id] + pc.z)
                        gateway_anchors[child.node_id].add(p)
                    aligned_children.add(child.node_id)

        # 1-cut articulation point alignment
        for child in children:
            if child.node_id in aligned_children:
                continue
            c_sol = child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates
            a = child.metadata.get("anchor_vnum")
            if a is None:
                common = room_set & set(c_coords.keys())
                if len(common) == 1:
                    a = next(iter(common))
            if a is not None and a in room_set and a in c_coords:
                pc = c_coords[a]
                model.align_x.add(model.x[a] == model.cx[child.node_id] + pc.x)
                model.align_y.add(model.y[a] == model.cy[child.node_id] + pc.y)
                model.align_z.add(model.z[a] == model.cz[child.node_id] + pc.z)
                gateway_anchors[child.node_id].add(a)
                aligned_children.add(child.node_id)

        # 1-cut doorway / bridge edge alignment
        doorway_edges: list[tuple[str, int, int, tuple[int, int, int], int]] = []
        for child in children:
            if child.node_id in aligned_children:
                continue
            c_sol = child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates

            for edge in graph.edges:
                if edge.cut:
                    continue
                if edge.src in room_set and edge.dst in c_coords:
                    try:
                        disp = direction_to_displacement(edge.direction)
                        doorway_edges.append((child.node_id, edge.src, edge.dst, disp, edge.weight))
                        gateway_anchors[child.node_id].add(edge.src)
                        aligned_children.add(child.node_id)
                        break
                    except ValueError:
                        pass
                elif edge.dst in room_set and edge.src in c_coords:
                    try:
                        disp = direction_to_displacement(edge.direction)
                        rev_disp = (-disp[0], -disp[1], -disp[2])
                        doorway_edges.append((child.node_id, edge.dst, edge.src, rev_disp, edge.weight))
                        gateway_anchors[child.node_id].add(edge.dst)
                        aligned_children.add(child.node_id)
                        break
                    except ValueError:
                        pass

        gw_indices = list(range(len(doorway_edges)))
        if gw_indices:
            model.L_gw = pyo.Var(
                gw_indices,
                domain=pyo.Integers,
                bounds=lambda m, i: (max(1, doorway_edges[i][4]), m_coord),
            )
            model.gw_x = pyo.ConstraintList()
            model.gw_y = pyo.ConstraintList()
            model.gw_z = pyo.ConstraintList()
            for j, (c_id, u, v, (dx, dy, dz), _) in enumerate(doorway_edges):
                c_sol = child_solutions[c_id]
                cv = c_sol.local_coordinates[v]
                model.gw_x.add((model.cx[c_id] + cv.x) - model.x[u] == model.L_gw[j] * dx)
                model.gw_y.add((model.cy[c_id] + cv.y) - model.y[u] == model.L_gw[j] * dy)
                model.gw_z.add((model.cz[c_id] + cv.z) - model.z[u] == model.L_gw[j] * dz)

        # Pairwise child-child macro-box separation
        child_pairs: list[tuple[SolveTreeNode, SolveTreeNode]] = []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1 = children[i]
                c2 = children[j]
                sol1 = child_solutions.get(c1.node_id)
                sol2 = child_solutions.get(c2.node_id)
                if (
                    sol1
                    and sol2
                    and sol1.bounding_box is not None
                    and sol2.bounding_box is not None
                ):
                    shared = set(sol1.local_coordinates.keys()) & set(sol2.local_coordinates.keys())
                    if not shared:
                        child_pairs.append((c1, c2))

        child_pair_keys = [(c1.node_id, c2.node_id) for c1, c2 in child_pairs]
        if child_pair_keys:
            model.cb_bl = pyo.Var(child_pair_keys, domain=pyo.Binary)
            model.cb_br = pyo.Var(child_pair_keys, domain=pyo.Binary)
            model.cb_bb = pyo.Var(child_pair_keys, domain=pyo.Binary)
            model.cb_ba = pyo.Var(child_pair_keys, domain=pyo.Binary)
            model.cb_bd = pyo.Var(child_pair_keys, domain=pyo.Binary)
            model.cb_bu = pyo.Var(child_pair_keys, domain=pyo.Binary)

            model.cb_sep_sum = pyo.ConstraintList()
            model.cb_sep_r = pyo.ConstraintList()
            model.cb_sep_l = pyo.ConstraintList()
            model.cb_sep_a = pyo.ConstraintList()
            model.cb_sep_b = pyo.ConstraintList()
            model.cb_sep_u = pyo.ConstraintList()
            model.cb_sep_d = pyo.ConstraintList()

            for c1, c2 in child_pairs:
                k1, k2 = c1.node_id, c2.node_id
                box1 = _get_child_bbox_corners(child_solutions[k1].bounding_box)
                box2 = _get_child_bbox_corners(child_solutions[k2].bounding_box)
                assert box1 is not None and box2 is not None

                model.cb_sep_sum.add(
                    model.cb_bl[k1, k2]
                    + model.cb_br[k1, k2]
                    + model.cb_bb[k1, k2]
                    + model.cb_ba[k1, k2]
                    + model.cb_bd[k1, k2]
                    + model.cb_bu[k1, k2]
                    >= 1
                )
                # Right: K1 min_x >= K2 max_x + 1
                model.cb_sep_r.add(
                    (model.cx[k1] + box1[0]) - (model.cx[k2] + box2[1])
                    >= 1 - 2 * m_big * (1 - model.cb_br[k1, k2])
                )
                # Left: K2 min_x >= K1 max_x + 1
                model.cb_sep_l.add(
                    (model.cx[k2] + box2[0]) - (model.cx[k1] + box1[1])
                    >= 1 - 2 * m_big * (1 - model.cb_bl[k1, k2])
                )
                # Ahead: K1 min_y >= K2 max_y + 1
                model.cb_sep_a.add(
                    (model.cy[k1] + box1[2]) - (model.cy[k2] + box2[3])
                    >= 1 - 2 * m_big * (1 - model.cb_ba[k1, k2])
                )
                # Behind: K2 min_y >= K1 max_y + 1
                model.cb_sep_b.add(
                    (model.cy[k2] + box2[2]) - (model.cy[k1] + box1[3])
                    >= 1 - 2 * m_big * (1 - model.cb_bb[k1, k2])
                )
                # Up: K1 min_z >= K2 max_z + 1
                model.cb_sep_u.add(
                    (model.cz[k1] + box1[4]) - (model.cz[k2] + box2[5])
                    >= 1 - 2 * m_big * (1 - model.cb_bu[k1, k2])
                )
                # Down: K2 min_z >= K1 max_z + 1
                model.cb_sep_d.add(
                    (model.cz[k2] + box2[4]) - (model.cz[k1] + box1[5])
                    >= 1 - 2 * m_big * (1 - model.cb_bd[k1, k2])
                )

        # Parent room vs child box clearance
        rc_pairs: list[tuple[int, SolveTreeNode]] = []
        if rooms:
            for child in children:
                c_sol = child_solutions.get(child.node_id)
                if c_sol and c_sol.bounding_box is not None:
                    anchors = gateway_anchors[child.node_id]
                    for u in rooms:
                        if u not in anchors and u not in c_sol.local_coordinates:
                            rc_pairs.append((u, child))

        rc_pair_keys = [(u, child.node_id) for u, child in rc_pairs]
        if rc_pair_keys:
            model.rc_bl = pyo.Var(rc_pair_keys, domain=pyo.Binary)
            model.rc_br = pyo.Var(rc_pair_keys, domain=pyo.Binary)
            model.rc_bb = pyo.Var(rc_pair_keys, domain=pyo.Binary)
            model.rc_ba = pyo.Var(rc_pair_keys, domain=pyo.Binary)
            model.rc_bd = pyo.Var(rc_pair_keys, domain=pyo.Binary)
            model.rc_bu = pyo.Var(rc_pair_keys, domain=pyo.Binary)

            model.rc_sep_sum = pyo.ConstraintList()
            model.rc_sep_r = pyo.ConstraintList()
            model.rc_sep_l = pyo.ConstraintList()
            model.rc_sep_a = pyo.ConstraintList()
            model.rc_sep_b = pyo.ConstraintList()
            model.rc_sep_u = pyo.ConstraintList()
            model.rc_sep_d = pyo.ConstraintList()

            for u, child in rc_pairs:
                cid = child.node_id
                box = _get_child_bbox_corners(child_solutions[cid].bounding_box)
                assert box is not None

                model.rc_sep_sum.add(
                    model.rc_bl[u, cid]
                    + model.rc_br[u, cid]
                    + model.rc_bb[u, cid]
                    + model.rc_ba[u, cid]
                    + model.rc_bd[u, cid]
                    + model.rc_bu[u, cid]
                    >= 1
                )
                # Right: xu >= child max_x + 1
                model.rc_sep_r.add(
                    model.x[u] - (model.cx[cid] + box[1])
                    >= 1 - 2 * m_big * (1 - model.rc_br[u, cid])
                )
                # Left: child min_x >= xu + 1
                model.rc_sep_l.add(
                    (model.cx[cid] + box[0]) - model.x[u]
                    >= 1 - 2 * m_big * (1 - model.rc_bl[u, cid])
                )
                # Ahead: yu >= child max_y + 1
                model.rc_sep_a.add(
                    model.y[u] - (model.cy[cid] + box[3])
                    >= 1 - 2 * m_big * (1 - model.rc_ba[u, cid])
                )
                # Behind: child min_y >= yu + 1
                model.rc_sep_b.add(
                    (model.cy[cid] + box[2]) - model.y[u]
                    >= 1 - 2 * m_big * (1 - model.rc_bb[u, cid])
                )
                # Up: zu >= child max_z + 1
                model.rc_sep_u.add(
                    model.z[u] - (model.cz[cid] + box[5])
                    >= 1 - 2 * m_big * (1 - model.rc_bu[u, cid])
                )
                # Down: child min_z >= zu + 1
                model.rc_sep_d.add(
                    (model.cz[cid] + box[4]) - model.z[u]
                    >= 1 - 2 * m_big * (1 - model.rc_bd[u, cid])
                )

        # Gateway corridor vs. third-party room non-penetration
        gw_room_pairs: list[tuple[int, int]] = []
        if rooms and doorway_edges:
            for j, (c_id, u, v, _, _) in enumerate(doorway_edges):
                c_sol = child_solutions.get(c_id)
                c_coords = c_sol.local_coordinates if c_sol else {}
                for w in rooms:
                    if w != u and w not in c_coords:
                        gw_room_pairs.append((j, w))

        if gw_room_pairs:
            model.gwr_b0 = pyo.Var(gw_room_pairs, domain=pyo.Binary)
            model.gwr_b1 = pyo.Var(gw_room_pairs, domain=pyo.Binary)
            model.gwr_b2 = pyo.Var(gw_room_pairs, domain=pyo.Binary)
            model.gwr_b3 = pyo.Var(gw_room_pairs, domain=pyo.Binary)
            model.gwr_b4 = pyo.Var(gw_room_pairs, domain=pyo.Binary)
            model.gwr_b5 = pyo.Var(gw_room_pairs, domain=pyo.Binary)

            model.gwr_sep = pyo.ConstraintList()

            for j, w in gw_room_pairs:
                c_id, u, v, (dx, dy, dz), _ = doorway_edges[j]
                cv = child_solutions[c_id].local_coordinates[v]

                model.gwr_sep.add(
                    model.gwr_b0[j, w]
                    + model.gwr_b1[j, w]
                    + model.gwr_b2[j, w]
                    + model.gwr_b3[j, w]
                    + model.gwr_b4[j, w]
                    + model.gwr_b5[j, w]
                    >= 1
                )

                if dx != 0:
                    model.gwr_sep.add(model.y[w] - model.y[u] >= 1 - 2 * m_big * (1 - model.gwr_b0[j, w]))
                    model.gwr_sep.add(model.y[u] - model.y[w] >= 1 - 2 * m_big * (1 - model.gwr_b1[j, w]))
                    model.gwr_sep.add(model.z[w] - model.z[u] >= 1 - 2 * m_big * (1 - model.gwr_b2[j, w]))
                    model.gwr_sep.add(model.z[u] - model.z[w] >= 1 - 2 * m_big * (1 - model.gwr_b3[j, w]))
                    if dx > 0:
                        model.gwr_sep.add(model.x[u] - model.x[w] >= 0 - 2 * m_big * (1 - model.gwr_b4[j, w]))
                        model.gwr_sep.add(model.x[w] - (model.cx[c_id] + cv.x) >= 0 - 2 * m_big * (1 - model.gwr_b5[j, w]))
                    else:
                        model.gwr_sep.add(model.x[w] - model.x[u] >= 0 - 2 * m_big * (1 - model.gwr_b4[j, w]))
                        model.gwr_sep.add((model.cx[c_id] + cv.x) - model.x[w] >= 0 - 2 * m_big * (1 - model.gwr_b5[j, w]))
                elif dy != 0:
                    model.gwr_sep.add(model.x[w] - model.x[u] >= 1 - 2 * m_big * (1 - model.gwr_b0[j, w]))
                    model.gwr_sep.add(model.x[u] - model.x[w] >= 1 - 2 * m_big * (1 - model.gwr_b1[j, w]))
                    model.gwr_sep.add(model.z[w] - model.z[u] >= 1 - 2 * m_big * (1 - model.gwr_b2[j, w]))
                    model.gwr_sep.add(model.z[u] - model.z[w] >= 1 - 2 * m_big * (1 - model.gwr_b3[j, w]))
                    if dy > 0:
                        model.gwr_sep.add(model.y[u] - model.y[w] >= 0 - 2 * m_big * (1 - model.gwr_b4[j, w]))
                        model.gwr_sep.add(model.y[w] - (model.cy[c_id] + cv.y) >= 0 - 2 * m_big * (1 - model.gwr_b5[j, w]))
                    else:
                        model.gwr_sep.add(model.y[w] - model.y[u] >= 0 - 2 * m_big * (1 - model.gwr_b4[j, w]))
                        model.gwr_sep.add((model.cy[c_id] + cv.y) - model.y[w] >= 0 - 2 * m_big * (1 - model.gwr_b5[j, w]))
                else:  # dz != 0
                    model.gwr_sep.add(model.x[w] - model.x[u] >= 1 - 2 * m_big * (1 - model.gwr_b0[j, w]))
                    model.gwr_sep.add(model.x[u] - model.x[w] >= 1 - 2 * m_big * (1 - model.gwr_b1[j, w]))
                    model.gwr_sep.add(model.y[w] - model.y[u] >= 1 - 2 * m_big * (1 - model.gwr_b2[j, w]))
                    model.gwr_sep.add(model.y[u] - model.y[w] >= 1 - 2 * m_big * (1 - model.gwr_b3[j, w]))
                    if dz > 0:
                        model.gwr_sep.add(model.z[u] - model.z[w] >= 0 - 2 * m_big * (1 - model.gwr_b4[j, w]))
                        model.gwr_sep.add(model.z[w] - (model.cz[c_id] + cv.z) >= 0 - 2 * m_big * (1 - model.gwr_b5[j, w]))
                    else:
                        model.gwr_sep.add(model.z[w] - model.z[u] >= 0 - 2 * m_big * (1 - model.gwr_b4[j, w]))
                        model.gwr_sep.add((model.cz[c_id] + cv.z) - model.z[w] >= 0 - 2 * m_big * (1 - model.gwr_b5[j, w]))

        # Gateway corridor vs. sibling child macro-box non-penetration
        gw_sibling_pairs: list[tuple[int, str]] = []
        if doorway_edges and len(children) > 1:
            for j, (c_id, u, v, _, _) in enumerate(doorway_edges):
                c_sol = child_solutions.get(c_id)
                for s_child in children:
                    s_id = s_child.node_id
                    if s_id == c_id:
                        continue
                    s_sol = child_solutions.get(s_id)
                    if not s_sol or s_sol.bounding_box is None:
                        continue
                    if c_sol:
                        shared = set(c_sol.local_coordinates.keys()) & set(s_sol.local_coordinates.keys())
                        if shared:
                            continue
                    if u in s_sol.local_coordinates:
                        continue
                    if _get_child_bbox_corners(s_sol.bounding_box) is not None:
                        gw_sibling_pairs.append((j, s_id))

        if gw_sibling_pairs:
            model.gws_br = pyo.Var(gw_sibling_pairs, domain=pyo.Binary)
            model.gws_bl = pyo.Var(gw_sibling_pairs, domain=pyo.Binary)
            model.gws_ba = pyo.Var(gw_sibling_pairs, domain=pyo.Binary)
            model.gws_bb = pyo.Var(gw_sibling_pairs, domain=pyo.Binary)
            model.gws_bu = pyo.Var(gw_sibling_pairs, domain=pyo.Binary)
            model.gws_bd = pyo.Var(gw_sibling_pairs, domain=pyo.Binary)

            model.gws_sep = pyo.ConstraintList()

            for j, s_id in gw_sibling_pairs:
                c_id, u, v, (dx, dy, dz), _ = doorway_edges[j]
                cv = child_solutions[c_id].local_coordinates[v]
                s_box = _get_child_bbox_corners(child_solutions[s_id].bounding_box)
                assert s_box is not None

                model.gws_sep.add(
                    model.gws_br[j, s_id]
                    + model.gws_bl[j, s_id]
                    + model.gws_ba[j, s_id]
                    + model.gws_bb[j, s_id]
                    + model.gws_bu[j, s_id]
                    + model.gws_bd[j, s_id]
                    >= 1
                )

                if dx == 1:
                    c_min_x, c_max_x = model.x[u], model.cx[c_id] + cv.x
                    c_min_y, c_max_y = model.y[u], model.y[u]
                    c_min_z, c_max_z = model.z[u], model.z[u]
                elif dx == -1:
                    c_min_x, c_max_x = model.cx[c_id] + cv.x, model.x[u]
                    c_min_y, c_max_y = model.y[u], model.y[u]
                    c_min_z, c_max_z = model.z[u], model.z[u]
                elif dy == 1:
                    c_min_x, c_max_x = model.x[u], model.x[u]
                    c_min_y, c_max_y = model.y[u], model.cy[c_id] + cv.y
                    c_min_z, c_max_z = model.z[u], model.z[u]
                elif dy == -1:
                    c_min_x, c_max_x = model.x[u], model.x[u]
                    c_min_y, c_max_y = model.cy[c_id] + cv.y, model.y[u]
                    c_min_z, c_max_z = model.z[u], model.z[u]
                elif dz == 1:
                    c_min_x, c_max_x = model.x[u], model.x[u]
                    c_min_y, c_max_y = model.y[u], model.y[u]
                    c_min_z, c_max_z = model.z[u], model.cz[c_id] + cv.z
                else:  # dz == -1
                    c_min_x, c_max_x = model.x[u], model.x[u]
                    c_min_y, c_max_y = model.y[u], model.y[u]
                    c_min_z, c_max_z = model.cz[c_id] + cv.z, model.z[u]

                s_min_x = model.cx[s_id] + s_box[0]
                s_max_x = model.cx[s_id] + s_box[1]
                s_min_y = model.cy[s_id] + s_box[2]
                s_max_y = model.cy[s_id] + s_box[3]
                s_min_z = model.cz[s_id] + s_box[4]
                s_max_z = model.cz[s_id] + s_box[5]

                # Right
                model.gws_sep.add(c_min_x - s_max_x >= 1 - 2 * m_big * (1 - model.gws_br[j, s_id]))
                # Left
                model.gws_sep.add(s_min_x - c_max_x >= 1 - 2 * m_big * (1 - model.gws_bl[j, s_id]))
                # Ahead
                model.gws_sep.add(c_min_y - s_max_y >= 1 - 2 * m_big * (1 - model.gws_ba[j, s_id]))
                # Behind
                model.gws_sep.add(s_min_y - c_max_y >= 1 - 2 * m_big * (1 - model.gws_bb[j, s_id]))
                # Up
                model.gws_sep.add(c_min_z - s_max_z >= 1 - 2 * m_big * (1 - model.gws_bu[j, s_id]))
                # Down
                model.gws_sep.add(s_min_z - c_max_z >= 1 - 2 * m_big * (1 - model.gws_bd[j, s_id]))

        # Corridor crossing separation (between perpendicular non-incident corridors)
        gw_corridor_descriptors = []
        for j, (c_id, u, v, (dx, dy, dz), _) in enumerate(doorway_edges):
            if sum(1 for d in (dx, dy, dz) if d != 0) != 1:
                continue
            cv = child_solutions[c_id].local_coordinates[v]
            if dx != 0:
                act = "x"
                c_min = model.x[u] if dx > 0 else model.cx[c_id] + cv.x
                c_max = model.cx[c_id] + cv.x if dx > 0 else model.x[u]
                coords = {"y": model.y[u], "z": model.z[u]}
            elif dy != 0:
                act = "y"
                c_min = model.y[u] if dy > 0 else model.cy[c_id] + cv.y
                c_max = model.cy[c_id] + cv.y if dy > 0 else model.y[u]
                coords = {"x": model.x[u], "z": model.z[u]}
            else:
                act = "z"
                c_min = model.z[u] if dz > 0 else model.cz[c_id] + cv.z
                c_max = model.cz[c_id] + cv.z if dz > 0 else model.z[u]
                coords = {"x": model.x[u], "y": model.y[u]}
            gw_corridor_descriptors.append({
                "type": "gw",
                "index": j,
                "u": u,
                "c_id": c_id,
                "v": v,
                "act": act,
                "min": c_min,
                "max": c_max,
                "coords": coords,
            })

        int_corridor_descriptors = []
        for i, (edge, (dx, dy, dz)) in enumerate(internal_edges):
            if sum(1 for d in (dx, dy, dz) if d != 0) != 1:
                continue
            if dx != 0:
                act = "x"
                c_min = model.x[edge.src] if dx > 0 else model.x[edge.dst]
                c_max = model.x[edge.dst] if dx > 0 else model.x[edge.src]
                coords = {"y": model.y[edge.src], "z": model.z[edge.src]}
            elif dy != 0:
                act = "y"
                c_min = model.y[edge.src] if dy > 0 else model.y[edge.dst]
                c_max = model.y[edge.dst] if dy > 0 else model.y[edge.src]
                coords = {"x": model.x[edge.src], "z": model.z[edge.src]}
            else:
                act = "z"
                c_min = model.z[edge.src] if dz > 0 else model.z[edge.dst]
                c_max = model.z[edge.dst] if dz > 0 else model.z[edge.src]
                coords = {"x": model.x[edge.src], "y": model.y[edge.src]}
            int_corridor_descriptors.append({
                "type": "int",
                "index": i,
                "src": edge.src,
                "dst": edge.dst,
                "act": act,
                "min": c_min,
                "max": c_max,
                "coords": coords,
            })

        crossing_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        # 1. Doorway vs. Internal parent streets
        for gw in gw_corridor_descriptors:
            for ic in int_corridor_descriptors:
                if gw["act"] != ic["act"] and gw["u"] not in (ic["src"], ic["dst"]):
                    pkey = f"gw{gw['index']}_int{ic['index']}"
                    crossing_pairs.append((pkey, gw, ic))

        # 2. Doorway vs. Doorway
        for j1 in range(len(gw_corridor_descriptors)):
            for j2 in range(j1 + 1, len(gw_corridor_descriptors)):
                gw1 = gw_corridor_descriptors[j1]
                gw2 = gw_corridor_descriptors[j2]
                if gw1["act"] != gw2["act"]:
                    incident = gw1["u"] == gw2["u"] or (gw1["c_id"] == gw2["c_id"] and gw1["v"] == gw2["v"])
                    if not incident:
                        pkey = f"gw{j1}_gw{j2}"
                        crossing_pairs.append((pkey, gw1, gw2))

        if crossing_pairs:
            crossing_keys = [p[0] for p in crossing_pairs]
            model.cc_b0 = pyo.Var(crossing_keys, domain=pyo.Binary)
            model.cc_b1 = pyo.Var(crossing_keys, domain=pyo.Binary)
            model.cc_b2 = pyo.Var(crossing_keys, domain=pyo.Binary)
            model.cc_b3 = pyo.Var(crossing_keys, domain=pyo.Binary)
            model.cc_b4 = pyo.Var(crossing_keys, domain=pyo.Binary)
            model.cc_b5 = pyo.Var(crossing_keys, domain=pyo.Binary)

            model.cc_sep = pyo.ConstraintList()

            for pkey, c1, c2 in crossing_pairs:
                a1 = c1["act"]
                a2 = c2["act"]
                a3 = next(iter({"x", "y", "z"} - {a1, a2}))

                model.cc_sep.add(
                    model.cc_b0[pkey]
                    + model.cc_b1[pkey]
                    + model.cc_b2[pkey]
                    + model.cc_b3[pkey]
                    + model.cc_b4[pkey]
                    + model.cc_b5[pkey]
                    >= 1
                )

                # Transverse/vertical separation along a3
                model.cc_sep.add(c1["coords"][a3] - c2["coords"][a3] >= 1 - 2 * m_big * (1 - model.cc_b0[pkey]))
                model.cc_sep.add(c2["coords"][a3] - c1["coords"][a3] >= 1 - 2 * m_big * (1 - model.cc_b1[pkey]))

                # Planar clearance along a1 (c2 is at constant coordinate c2["coords"][a1])
                model.cc_sep.add(c2["coords"][a1] - c1["max"] >= 1 - 2 * m_big * (1 - model.cc_b2[pkey]))
                model.cc_sep.add(c1["min"] - c2["coords"][a1] >= 1 - 2 * m_big * (1 - model.cc_b3[pkey]))

                # Planar clearance along a2 (c1 is at constant coordinate c1["coords"][a2])
                model.cc_sep.add(c1["coords"][a2] - c2["max"] >= 1 - 2 * m_big * (1 - model.cc_b4[pkey]))
                model.cc_sep.add(c2["min"] - c1["coords"][a2] >= 1 - 2 * m_big * (1 - model.cc_b5[pkey]))

        # Parent bounding box variables and constraints
        model.x_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.x_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))

        model.bbox_bounds = pyo.ConstraintList()
        for u in rooms:
            model.bbox_bounds.add(model.x_max >= model.x[u])
            model.bbox_bounds.add(model.x_min <= model.x[u])
            model.bbox_bounds.add(model.y_max >= model.y[u])
            model.bbox_bounds.add(model.y_min <= model.y[u])
            model.bbox_bounds.add(model.z_max >= model.z[u])
            model.bbox_bounds.add(model.z_min <= model.z[u])

        for child in children:
            cid = child.node_id
            c_sol = child_solutions.get(cid)
            box = _get_child_bbox_corners(c_sol.bounding_box) if c_sol else None
            if box:
                model.bbox_bounds.add(model.x_max >= model.cx[cid] + box[1])
                model.bbox_bounds.add(model.x_min <= model.cx[cid] + box[0])
                model.bbox_bounds.add(model.y_max >= model.cy[cid] + box[3])
                model.bbox_bounds.add(model.y_min <= model.cy[cid] + box[2])
                model.bbox_bounds.add(model.z_max >= model.cz[cid] + box[5])
                model.bbox_bounds.add(model.z_min <= model.cz[cid] + box[4])
            else:
                model.bbox_bounds.add(model.x_max >= model.cx[cid])
                model.bbox_bounds.add(model.x_min <= model.cx[cid])
                model.bbox_bounds.add(model.y_max >= model.cy[cid])
                model.bbox_bounds.add(model.y_min <= model.cy[cid])
                model.bbox_bounds.add(model.z_max >= model.cz[cid])
                model.bbox_bounds.add(model.z_min <= model.cz[cid])

        # Objective
        int_stretch_expr = sum(
            10 * (model.L_int[i] - max(1, internal_edges[i][0].weight))
            for i in int_edge_indices
        )
        gw_stretch_expr = sum(
            10 * (model.L_gw[j] - max(1, doorway_edges[j][4]))
            for j in gw_indices
        )
        span_expr = (
            (model.x_max - model.x_min)
            + (model.y_max - model.y_min)
            + (model.z_max - model.z_min)
        )
        model.obj = pyo.Objective(
            expr=int_stretch_expr + gw_stretch_expr + span_expr,
            sense=pyo.minimize,
        )

        if options and options.timeout_seconds:
            if solver_name == "cbc":
                solver.options["seconds"] = options.timeout_seconds
            elif solver_name == "glpk":
                solver.options["tmlim"] = options.timeout_seconds

        results = solver.solve(model, tee=False)

        valid_status = (
            results.solver.status == PyomoSolverStatus.ok
            and results.solver.termination_condition
            in (
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit,
            )
        )
        if not valid_status:
            return _solve_parent_fallback(node, child_solutions, graph)

        raw_parent_coords: dict[int, tuple[int, int, int]] = {}
        for u in rooms:
            vx = pyo.value(model.x[u])
            vy = pyo.value(model.y[u])
            vz = pyo.value(model.z[u])
            if None in (vx, vy, vz):
                return _solve_parent_fallback(node, child_solutions, graph)
            assert vx is not None and vy is not None and vz is not None
            raw_parent_coords[u] = (int(round(vx)), int(round(vy)), int(round(vz)))

        raw_child_offsets: dict[str, tuple[int, int, int]] = {}
        for cid in child_ids:
            vcx = pyo.value(model.cx[cid])
            vcy = pyo.value(model.cy[cid])
            vcz = pyo.value(model.cz[cid])
            if None in (vcx, vcy, vcz):
                return _solve_parent_fallback(node, child_solutions, graph)
            assert vcx is not None and vcy is not None and vcz is not None
            raw_child_offsets[cid] = (int(round(vcx)), int(round(vcy)), int(round(vcz)))

        # Normalization
        all_pts: list[tuple[int, int, int]] = list(raw_parent_coords.values())
        for child in children:
            cid = child.node_id
            cx, cy, cz = raw_child_offsets[cid]
            c_sol = child_solutions.get(cid)
            box = _get_child_bbox_corners(c_sol.bounding_box) if c_sol else None
            if box:
                all_pts.append((cx + box[0], cy + box[2], cz + box[4]))
                all_pts.append((cx + box[1], cy + box[3], cz + box[5]))
            else:
                all_pts.append((cx, cy, cz))

        min_x = min(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        min_z = min(p[2] for p in all_pts)

        norm_parent_coords = {
            u: Coord3D(x - min_x, y - min_y, z - min_z)
            for u, (x, y, z) in raw_parent_coords.items()
        }
        norm_child_offsets = {
            cid: Coord3D(x - min_x, y - min_y, z - min_z)
            for cid, (x, y, z) in raw_child_offsets.items()
        }

        shifted_pts = [Coord3D(p[0] - min_x, p[1] - min_y, p[2] - min_z) for p in all_pts]
        enc = BoundingBox3D.from_coordinates(shifted_pts)
        parent_bbox = (
            Coord3D(enc.min_x, enc.min_y, enc.min_z),
            Coord3D(enc.max_x, enc.max_y, enc.max_z),
        )

        return NodeSolution(
            node_id=node.node_id,
            local_coordinates=norm_parent_coords,
            bounding_box=parent_bbox,
            metadata={"child_offsets": norm_child_offsets},
        )
    except Exception:
        return _solve_parent_fallback(node, child_solutions, graph)


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

        solutions: dict[str, NodeSolution] = {}

        def _solve_node(node: SolveTreeNode) -> NodeSolution:
            if node.node_id in solutions:
                return solutions[node.node_id]

            if not node.children:
                sol = solve_leaf_cluster(
                    node,
                    layout_graph,
                    options,
                    solver_name=self.solver_name,
                )
                solutions[node.node_id] = sol
                return sol

            # Post-order traversal: solve all children first
            child_solutions_map: dict[str, NodeSolution] = {}
            for child in node.children:
                child_sol = _solve_node(child)
                child_solutions_map[child.node_id] = child_sol

            parent_sol = solve_parent_cluster(
                node,
                child_solutions_map,
                layout_graph,
                options,
                solver_name=self.solver_name,
            )
            solutions[node.node_id] = parent_sol
            return parent_sol

        _solve_node(tree.root)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp)

        return TreeSolution(node_solutions=solutions)
