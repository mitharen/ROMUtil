"""MILP formulation and solver for parent container clusters and gateway alignment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.protocols import TreeSolver
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Room
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, SolverStatus


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

    if len(rooms) == 1 and not children:
        r = rooms[0]
        return NodeSolution(
            node_id=node.node_id,
            local_coordinates={r: Coord3D(0, 0, 0)},
            bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)),
            metadata={"child_offsets": {}},
        )

    try:
        solver = pyo.SolverFactory(solver_name)
        if not solver.available():
            raise LayoutSolverError(f"MILP solver '{solver_name}' is not available.")
    except Exception as exc:
        if isinstance(exc, LayoutSolverError):
            raise
        raise LayoutSolverError(f"Failed to initialize MILP solver '{solver_name}': {exc}") from exc

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

        def _room_box(u: int) -> tuple[Any, Any, Any, Any, Any, Any]:
            return (model.x[u], model.x[u], model.y[u], model.y[u], model.z[u], model.z[u])

        # Parent room non-overlap separation
        if len(rooms) > 1:
            parent_pairs = [
                (rooms[i], rooms[j])
                for i in range(len(rooms))
                for j in range(i + 1, len(rooms))
            ]
            add_disjunctive_box_separation(
                model=model,
                pairs=parent_pairs,
                get_box_a=_room_box,
                get_box_b=_room_box,
                prefix="pr",
                m_big=m_big,
            )


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
            p0, p1 = poles[0], poles[1]
            split_children = [
                c
                for c in children
                if c.node_id in child_solutions
                and p0 in child_solutions[c.node_id].local_coordinates
                and p1 in child_solutions[c.node_id].local_coordinates
            ]
            if split_children:
                def _span(c: SolveTreeNode) -> int:
                    c_coords = child_solutions[c.node_id].local_coordinates
                    c0, c1 = c_coords[p0], c_coords[p1]
                    return abs(c1.x - c0.x) + abs(c1.y - c0.y) + abs(c1.z - c0.z)

                primary_child = max(split_children, key=_span)
                pc0_prim = child_solutions[primary_child.node_id].local_coordinates[p0]
                pc1_prim = child_solutions[primary_child.node_id].local_coordinates[p1]
                primary_disp = (
                    pc1_prim.x - pc0_prim.x,
                    pc1_prim.y - pc0_prim.y,
                    pc1_prim.z - pc0_prim.z,
                )

                for child in split_children:
                    c_sol = child_solutions[child.node_id]
                    pc0 = c_sol.local_coordinates[p0]
                    model.align_x.add(model.x[p0] == model.cx[child.node_id] + pc0.x)
                    model.align_y.add(model.y[p0] == model.cy[child.node_id] + pc0.y)
                    model.align_z.add(model.z[p0] == model.cz[child.node_id] + pc0.z)
                    gateway_anchors[child.node_id].add(p0)

                    pc1 = c_sol.local_coordinates[p1]
                    child_disp = (pc1.x - pc0.x, pc1.y - pc0.y, pc1.z - pc0.z)
                    if child_disp == primary_disp:
                        model.align_x.add(model.x[p1] == model.cx[child.node_id] + pc1.x)
                        model.align_y.add(model.y[p1] == model.cy[child.node_id] + pc1.y)
                        model.align_z.add(model.z[p1] == model.cz[child.node_id] + pc1.z)
                        gateway_anchors[child.node_id].add(p1)

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

        # Macro child box accessor
        def _child_box(cid: str) -> tuple[Any, Any, Any, Any, Any, Any]:
            box = _get_child_bbox_corners(child_solutions[cid].bounding_box)
            assert box is not None
            return (
                model.cx[cid] + box[0],
                model.cx[cid] + box[1],
                model.cy[cid] + box[2],
                model.cy[cid] + box[3],
                model.cz[cid] + box[4],
                model.cz[cid] + box[5],
            )

        # Pairwise child-child macro-box separation
        child_pairs: list[tuple[str, str]] = []
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
                        child_pairs.append((c1.node_id, c2.node_id))

        add_disjunctive_box_separation(
            model=model,
            pairs=child_pairs,
            get_box_a=_child_box,
            get_box_b=_child_box,
            prefix="cb",
            m_big=m_big,
        )

        # Parent room vs child box clearance
        rc_pairs: list[tuple[int, str]] = []
        if rooms:
            for child in children:
                c_sol = child_solutions.get(child.node_id)
                if c_sol and c_sol.bounding_box is not None:
                    anchors = gateway_anchors[child.node_id]
                    for u in rooms:
                        if u not in anchors and u not in c_sol.local_coordinates:
                            rc_pairs.append((u, child.node_id))

        add_disjunctive_box_separation(
            model=model,
            pairs=rc_pairs,
            get_box_a=_room_box,
            get_box_b=_child_box,
            prefix="rc",
            m_big=m_big,
        )


        # Gateway corridor box helpers
        def _gw_corridor_interior_box(j: int) -> tuple[Any, Any, Any, Any, Any, Any]:
            c_id, u, v, (dx, dy, dz), _ = doorway_edges[j]
            cv = child_solutions[c_id].local_coordinates[v]
            target_x = model.cx[c_id] + cv.x
            target_y = model.cy[c_id] + cv.y
            target_z = model.cz[c_id] + cv.z
            if dx > 0:
                return (model.x[u] + 1, target_x - 1, model.y[u], model.y[u], model.z[u], model.z[u])
            elif dx < 0:
                return (target_x + 1, model.x[u] - 1, model.y[u], model.y[u], model.z[u], model.z[u])
            elif dy > 0:
                return (model.x[u], model.x[u], model.y[u] + 1, target_y - 1, model.z[u], model.z[u])
            elif dy < 0:
                return (model.x[u], model.x[u], target_y + 1, model.y[u] - 1, model.z[u], model.z[u])
            elif dz > 0:
                return (model.x[u], model.x[u], model.y[u], model.y[u], model.z[u] + 1, target_z - 1)
            else:  # dz < 0
                return (model.x[u], model.x[u], model.y[u], model.y[u], target_z + 1, model.z[u] - 1)

        def _gw_corridor_full_box(j: int) -> tuple[Any, Any, Any, Any, Any, Any]:
            c_id, u, v, (dx, dy, dz), _ = doorway_edges[j]
            cv = child_solutions[c_id].local_coordinates[v]
            target_x = model.cx[c_id] + cv.x
            target_y = model.cy[c_id] + cv.y
            target_z = model.cz[c_id] + cv.z
            if dx > 0:
                return (model.x[u], target_x, model.y[u], model.y[u], model.z[u], model.z[u])
            elif dx < 0:
                return (target_x, model.x[u], model.y[u], model.y[u], model.z[u], model.z[u])
            elif dy > 0:
                return (model.x[u], model.x[u], model.y[u], target_y, model.z[u], model.z[u])
            elif dy < 0:
                return (model.x[u], model.x[u], target_y, model.y[u], model.z[u], model.z[u])
            elif dz > 0:
                return (model.x[u], model.x[u], model.y[u], model.y[u], model.z[u], target_z)
            else:  # dz < 0
                return (model.x[u], model.x[u], model.y[u], model.y[u], target_z, model.z[u])

        def _int_corridor_box(i: int) -> tuple[Any, Any, Any, Any, Any, Any]:
            edge, (dx, dy, dz) = internal_edges[i]
            if dx > 0:
                return (model.x[edge.src], model.x[edge.dst], model.y[edge.src], model.y[edge.src], model.z[edge.src], model.z[edge.src])
            elif dx < 0:
                return (model.x[edge.dst], model.x[edge.src], model.y[edge.src], model.y[edge.src], model.z[edge.src], model.z[edge.src])
            elif dy > 0:
                return (model.x[edge.src], model.x[edge.src], model.y[edge.src], model.y[edge.dst], model.z[edge.src], model.z[edge.src])
            elif dy < 0:
                return (model.x[edge.src], model.x[edge.src], model.y[edge.dst], model.y[edge.src], model.z[edge.src], model.z[edge.src])
            elif dz > 0:
                return (model.x[edge.src], model.x[edge.src], model.y[edge.src], model.y[edge.src], model.z[edge.src], model.z[edge.dst])
            else:  # dz < 0
                return (model.x[edge.src], model.x[edge.src], model.y[edge.src], model.y[edge.src], model.z[edge.dst], model.z[edge.src])

        # Gateway corridor vs. third-party room non-penetration
        gw_room_pairs: list[tuple[int, int]] = []
        if rooms and doorway_edges:
            for j, (c_id, u, v, _, _) in enumerate(doorway_edges):
                c_sol = child_solutions.get(c_id)
                c_coords = c_sol.local_coordinates if c_sol else {}
                for w in rooms:
                    if w != u and w not in c_coords:
                        gw_room_pairs.append((j, w))

        add_disjunctive_box_separation(
            model=model,
            pairs=gw_room_pairs,
            get_box_a=_gw_corridor_interior_box,
            get_box_b=_room_box,
            prefix="gwr",
            m_big=m_big,
        )

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

        add_disjunctive_box_separation(
            model=model,
            pairs=gw_sibling_pairs,
            get_box_a=_gw_corridor_full_box,
            get_box_b=_child_box,
            prefix="gws",
            m_big=m_big,
        )

        # Corridor crossing separation (between perpendicular non-incident corridors)
        corridor_boxes: dict[str, tuple[Any, Any, Any, Any, Any, Any]] = {}
        gw_corridor_info: list[dict[str, Any]] = []
        for j, (c_id, u, v, (dx, dy, dz), _) in enumerate(doorway_edges):
            if sum(1 for d in (dx, dy, dz) if d != 0) != 1:
                continue
            cid = f"gw_{j}"
            act = "x" if dx != 0 else ("y" if dy != 0 else "z")
            corridor_boxes[cid] = _gw_corridor_full_box(j)
            gw_corridor_info.append({
                "cid": cid,
                "u": u,
                "c_id": c_id,
                "v": v,
                "act": act,
            })

        int_corridor_info: list[dict[str, Any]] = []
        for i, (edge, (dx, dy, dz)) in enumerate(internal_edges):
            if sum(1 for d in (dx, dy, dz) if d != 0) != 1:
                continue
            cid = f"int_{i}"
            act = "x" if dx != 0 else ("y" if dy != 0 else "z")
            corridor_boxes[cid] = _int_corridor_box(i)
            int_corridor_info.append({
                "cid": cid,
                "src": edge.src,
                "dst": edge.dst,
                "act": act,
            })

        crossing_pairs: list[tuple[str, str]] = []

        # 1. Doorway vs. Internal parent streets
        for gw in gw_corridor_info:
            for ic in int_corridor_info:
                if gw["act"] != ic["act"] and gw["u"] not in (ic["src"], ic["dst"]):
                    crossing_pairs.append((gw["cid"], ic["cid"]))

        # 2. Doorway vs. Doorway
        for j1 in range(len(gw_corridor_info)):
            for j2 in range(j1 + 1, len(gw_corridor_info)):
                gw1 = gw_corridor_info[j1]
                gw2 = gw_corridor_info[j2]
                if gw1["act"] != gw2["act"]:
                    incident = gw1["u"] == gw2["u"] or (gw1["c_id"] == gw2["c_id"] and gw1["v"] == gw2["v"])
                    if not incident:
                        crossing_pairs.append((gw1["cid"], gw2["cid"]))

        add_disjunctive_box_separation(
            model=model,
            pairs=crossing_pairs,
            get_box_a=lambda cid: corridor_boxes[cid],
            get_box_b=lambda cid: corridor_boxes[cid],
            prefix="cc",
            m_big=m_big,
        )


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
            (
                results.solver.status == PyomoSolverStatus.ok
                and results.solver.termination_condition
                in (
                    TerminationCondition.optimal,
                    TerminationCondition.feasible,
                )
            )
            or (
                results.solver.termination_condition
                in (
                    TerminationCondition.maxTimeLimit,
                    TerminationCondition.intermediateNonInteger,
                )
                and results.solver.status
                in (
                    PyomoSolverStatus.ok,
                    PyomoSolverStatus.aborted,
                    PyomoSolverStatus.warning,
                )
            )
        )
        if not valid_status:
            raise LayoutSolverError(
                f"MILP solver returned non-optimal status: {results.solver.status}, "
                f"termination condition: {results.solver.termination_condition} for cluster '{node.node_id}'."
            )

        raw_parent_coords: dict[int, tuple[int, int, int]] = {}
        for u in rooms:
            vx = pyo.value(model.x[u])
            vy = pyo.value(model.y[u])
            vz = pyo.value(model.z[u])
            if vx is None or vy is None or vz is None:
                raise LayoutSolverError(
                    f"MILP solver failed to retrieve coordinate values for room {u} in cluster '{node.node_id}'."
                )
            raw_parent_coords[u] = (int(round(vx)), int(round(vy)), int(round(vz)))

        raw_child_offsets: dict[str, tuple[int, int, int]] = {}
        for cid in child_ids:
            vcx = pyo.value(model.cx[cid])
            vcy = pyo.value(model.cy[cid])
            vcz = pyo.value(model.cz[cid])
            if vcx is None or vcy is None or vcz is None:
                raise LayoutSolverError(
                    f"MILP solver failed to retrieve offset values for child '{cid}' in cluster '{node.node_id}'."
                )
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

        solver_status = SolverStatus.FEASIBLE
        if results.solver.termination_condition == TerminationCondition.optimal:
            solver_status = SolverStatus.OPTIMAL
        elif results.solver.termination_condition in (
            TerminationCondition.maxTimeLimit,
            TerminationCondition.intermediateNonInteger,
        ):
            solver_status = SolverStatus.TIMEOUT

        return NodeSolution(
            node_id=node.node_id,
            local_coordinates=norm_parent_coords,
            bounding_box=parent_bbox,
            metadata={"child_offsets": norm_child_offsets, "status": solver_status},
        )
    except LayoutSolverError:
        raise
    except Exception as exc:
        raise LayoutSolverError(
            f"MILP solver failed for cluster '{node.node_id}': {exc}"
        ) from exc


class ParentClusterSolver:
    """Solver for parent container clusters and leaf clusters via MILP."""

    def __init__(self, solver_name: str = "cbc") -> None:
        self.solver_name = solver_name

    def solve(
        self,
        node: SolveTreeNode,
        child_solutions: Mapping[str, NodeSolution],
        graph: LayoutGraph,
        options: LayoutOptions | None = None,
    ) -> NodeSolution:
        """Solve cluster layout non-overlapping coordinates and child offsets."""
        return solve_parent_cluster(
            node=node,
            child_solutions=child_solutions,
            graph=graph,
            options=options,
            solver_name=self.solver_name,
        )


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

        def _solve_node(node: SolveTreeNode, g: LayoutGraph) -> NodeSolution:
            if node.node_id in solutions:
                return solutions[node.node_id]

            # Post-order traversal: solve all children first
            child_solutions_map: dict[str, NodeSolution] = {}
            for child in node.children:
                child_sol = _solve_node(child, g)
                child_solutions_map[child.node_id] = child_sol

            cut_type = node.metadata.get("cut_type")
            poles = tuple(node.metadata.get("poles", ()))
            if cut_type == "split_pair" and len(poles) == 2:
                p0, p1 = poles[0], poles[1]
                split_children = [
                    c
                    for c in node.children
                    if p0 in child_solutions_map[c.node_id].local_coordinates
                    and p1 in child_solutions_map[c.node_id].local_coordinates
                ]
                if len(split_children) > 1:
                    max_dx = max(
                        abs(child_solutions_map[c.node_id].local_coordinates[p1].x - child_solutions_map[c.node_id].local_coordinates[p0].x)
                        for c in split_children
                    )
                    max_dy = max(
                        abs(child_solutions_map[c.node_id].local_coordinates[p1].y - child_solutions_map[c.node_id].local_coordinates[p0].y)
                        for c in split_children
                    )
                    max_dz = max(
                        abs(child_solutions_map[c.node_id].local_coordinates[p1].z - child_solutions_map[c.node_id].local_coordinates[p0].z)
                        for c in split_children
                    )
                    max_span = max(1, max_dx, max_dy, max_dz)
                    for c in split_children:
                        c_sol = child_solutions_map[c.node_id]
                        cur_span = max(
                            abs(c_sol.local_coordinates[p1].x - c_sol.local_coordinates[p0].x),
                            abs(c_sol.local_coordinates[p1].y - c_sol.local_coordinates[p0].y),
                            abs(c_sol.local_coordinates[p1].z - c_sol.local_coordinates[p0].z),
                        )
                        if cur_span < max_span:
                            new_edges = [
                                replace(e, weight=max_span) if {e.src, e.dst} == {p0, p1} else e
                                for e in g.edges
                            ]
                            g_sub = replace(g, edges=tuple(new_edges))

                            def _clear_subtree(sub_n: SolveTreeNode) -> None:
                                solutions.pop(sub_n.node_id, None)
                                for sub_c in sub_n.children:
                                    _clear_subtree(sub_c)

                            _clear_subtree(c)
                            child_solutions_map[c.node_id] = _solve_node(c, g_sub)

            parent_sol = solve_parent_cluster(
                node,
                child_solutions_map,
                g,
                options,
                solver_name=self.solver_name,
            )
            solutions[node.node_id] = parent_sol
            return parent_sol

        _solve_node(tree.root, layout_graph)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp, layout_graph)

        return TreeSolution(node_solutions=solutions)
