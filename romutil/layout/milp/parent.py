"""MILP formulation and solver for parent container clusters and gateway alignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

EndpointRef = int | tuple[str, int]


@dataclass(frozen=True)
class ActiveEdge:
    """Active layout edge between room or boundary port endpoints."""

    edge_id: str
    src: EndpointRef
    dst: EndpointRef
    direction: tuple[int, int, int]
    min_length: int
    associated_rooms: frozenset[int]
    associated_children: frozenset[str]

import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
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

        def get_point(ref: EndpointRef) -> tuple[Any, Any, Any]:
            if isinstance(ref, int):
                return (model.x[ref], model.y[ref], model.z[ref])
            cid, v = ref
            loc = child_solutions[cid].local_coordinates[v]
            return (model.cx[cid] + loc.x, model.cy[cid] + loc.y, model.cz[cid] + loc.z)

        def _room_box(u: int) -> tuple[Any, Any, Any, Any, Any, Any]:
            px, py, pz = get_point(u)
            return (px, px, py, py, pz, pz)

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

        # Gateway alignments and active edge collection
        gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in children}
        aligned_children: set[str] = set()
        if not rooms and child_ids:
            aligned_children.add(child_ids[0])

        model.align_x = pyo.ConstraintList()
        model.align_y = pyo.ConstraintList()
        model.align_z = pyo.ConstraintList()

        active_edges: list[ActiveEdge] = []
        room_set = set(rooms)

        # Split-pair alignment
        split_children_by_poles: dict[tuple[int, int], list[SolveTreeNode]] = {}
        for child in children:
            if child.node_id not in child_solutions:
                continue
            poles: tuple[int, int] | None = None
            contract = child.cut_contract
            if isinstance(contract, SplitPairCut):
                poles = contract.poles
            elif (
                child.cut_type == CutType.SPLIT_PAIR
                or child.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair")
                or "poles" in child.metadata
            ):
                raw_poles = child.metadata.get("poles")
                if raw_poles and len(raw_poles) == 2:
                    poles = (int(raw_poles[0]), int(raw_poles[1]))

            if poles is not None:
                p0, p1 = poles[0], poles[1]
                c_coords = child_solutions[child.node_id].local_coordinates
                if p0 in room_set and p1 in room_set and p0 in c_coords and p1 in c_coords:
                    can_pair = (min(p0, p1), max(p0, p1))
                    split_children_by_poles.setdefault(can_pair, []).append(child)

        # Support container-level split metadata
        if node.cut_type in (CutType.SPLIT_PAIR, "split_pair") or "poles" in node.metadata:
            raw_node_poles = node.metadata.get("poles")
            if raw_node_poles and len(raw_node_poles) == 2:
                np0, np1 = int(raw_node_poles[0]), int(raw_node_poles[1])
                if np0 in room_set and np1 in room_set:
                    can_pair = (min(np0, np1), max(np0, np1))
                    for child in children:
                        if child.node_id in child_solutions:
                            c_coords = child_solutions[child.node_id].local_coordinates
                            if np0 in c_coords and np1 in c_coords:
                                if child not in split_children_by_poles.get(can_pair, []):
                                    split_children_by_poles.setdefault(can_pair, []).append(child)

        split_pole_pairs = set(split_children_by_poles.keys())

        is_split_pair_container = (
            node.node_type in (TreeNodeType.SPLIT_PAIR_CONTAINER, "split_pair_container")
            or (node.node_type == "container" and "poles" in node.metadata)
        )

        split_min_lengths: dict[tuple[int, int], int] = {}
        split_directions: dict[tuple[int, int], tuple[int, int, int]] = {}
        for (p0, p1), split_children in split_children_by_poles.items():
            if not split_children:
                continue

            for child in split_children:
                c_sol = child_solutions[child.node_id]
                pc0 = c_sol.local_coordinates[p0]
                model.align_x.add(model.x[p0] == model.cx[child.node_id] + pc0.x)
                model.align_y.add(model.y[p0] == model.cy[child.node_id] + pc0.y)
                model.align_z.add(model.z[p0] == model.cz[child.node_id] + pc0.z)
                gateway_anchors[child.node_id].add(p0)
                aligned_children.add(child.node_id)

            def _child_span(c: SolveTreeNode) -> int:
                c_coords = child_solutions[c.node_id].local_coordinates
                return (
                    abs(c_coords[p1].x - c_coords[p0].x)
                    + abs(c_coords[p1].y - c_coords[p0].y)
                    + abs(c_coords[p1].z - c_coords[p0].z)
                )

            prim_child = max(split_children, key=_child_span)
            prim_sol = child_solutions[prim_child.node_id]
            pc0_prim = prim_sol.local_coordinates[p0]
            pc1_prim = prim_sol.local_coordinates[p1]
            pdx = pc1_prim.x - pc0_prim.x
            pdy = pc1_prim.y - pc0_prim.y
            pdz = pc1_prim.z - pc0_prim.z

            is_axis_aligned = (sum(1 for d in (pdx, pdy, pdz) if d != 0) == 1)
            d_vec: tuple[int, int, int] | None = None

            if is_split_pair_container:
                if is_axis_aligned:
                    d_vec = (
                        1 if pdx > 0 else (-1 if pdx < 0 else 0),
                        1 if pdy > 0 else (-1 if pdy < 0 else 0),
                        1 if pdz > 0 else (-1 if pdz < 0 else 0),
                    )
                else:
                    abs_deltas = [
                        (abs(pdx), (1 if pdx > 0 else (-1 if pdx < 0 else 0), 0, 0)),
                        (abs(pdy), (0, 1 if pdy > 0 else (-1 if pdy < 0 else 0), 0)),
                        (abs(pdz), (0, 0, 1 if pdz > 0 else (-1 if pdz < 0 else 0))),
                    ]
                    max_delta, d_cand = max(abs_deltas, key=lambda t: t[0])
                    d_vec = (1, 0, 0) if max_delta == 0 else d_cand

            l_min = max(
                1,
                max(
                    max(
                        abs(child_solutions[c.node_id].local_coordinates[p1].x - child_solutions[c.node_id].local_coordinates[p0].x),
                        abs(child_solutions[c.node_id].local_coordinates[p1].y - child_solutions[c.node_id].local_coordinates[p0].y),
                        abs(child_solutions[c.node_id].local_coordinates[p1].z - child_solutions[c.node_id].local_coordinates[p0].z),
                    )
                    for c in split_children
                ),
            )

            for ge in graph.edges:
                if not ge.cut and {ge.src, ge.dst} == {p0, p1}:
                    l_min = max(l_min, max(1, ge.weight))

            can_pair = (min(p0, p1), max(p0, p1))
            split_min_lengths[can_pair] = l_min
            if d_vec is not None:
                split_directions[can_pair] = d_vec

        # 1-cut articulation point alignment
        for child in children:
            if child.node_id in aligned_children:
                continue
            c_sol = child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates
            a: int | None = None
            contract = child.cut_contract
            if isinstance(contract, ArticulationPointCut):
                a = contract.anchor_vnum
            elif (
                child.cut_type == CutType.ARTICULATION_POINT
                or child.metadata.get("cut_type") in (CutType.ARTICULATION_POINT, "articulation_point")
                or "anchor_vnum" in child.metadata
            ):
                raw_a = child.metadata.get("anchor_vnum")
                if raw_a is not None:
                    a = int(raw_a)

            if a is not None and a in room_set and a in c_coords:
                pc = c_coords[a]
                model.align_x.add(model.x[a] == model.cx[child.node_id] + pc.x)
                model.align_y.add(model.y[a] == model.cy[child.node_id] + pc.y)
                model.align_z.add(model.z[a] == model.cz[child.node_id] + pc.z)
                gateway_anchors[child.node_id].add(a)
                aligned_children.add(child.node_id)

        # 1-cut doorway / bridge edge alignment
        for child in children:
            if child.node_id in aligned_children:
                continue
            c_sol = child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates

            contract = child.cut_contract
            if isinstance(contract, BridgeCut):
                u, v = contract.bridge_edge
                dir_val = contract.direction
                weight = contract.weight
                if u in room_set and v in c_coords:
                    try:
                        disp = direction_to_displacement(dir_val)
                        active_edges.append(
                            ActiveEdge(
                                edge_id=f"gw_{child.node_id}_{u}_{v}",
                                src=u,
                                dst=(child.node_id, v),
                                direction=disp,
                                min_length=max(1, weight),
                                associated_rooms=frozenset({u}),
                                associated_children=frozenset({child.node_id}),
                            )
                        )
                        gateway_anchors[child.node_id].add(u)
                        aligned_children.add(child.node_id)
                        continue
                    except ValueError:
                        pass
                elif v in room_set and u in c_coords:
                    try:
                        disp = direction_to_displacement(dir_val)
                        active_edges.append(
                            ActiveEdge(
                                edge_id=f"gw_{child.node_id}_{u}_{v}",
                                src=(child.node_id, u),
                                dst=v,
                                direction=disp,
                                min_length=max(1, weight),
                                associated_rooms=frozenset({v}),
                                associated_children=frozenset({child.node_id}),
                            )
                        )
                        gateway_anchors[child.node_id].add(v)
                        aligned_children.add(child.node_id)
                        continue
                    except ValueError:
                        pass

            for edge in graph.edges:
                if edge.cut:
                    continue
                if edge.src in room_set and edge.dst in c_coords:
                    try:
                        disp = direction_to_displacement(edge.direction)
                        active_edges.append(
                            ActiveEdge(
                                edge_id=f"gw_{child.node_id}_{edge.src}_{edge.dst}",
                                src=edge.src,
                                dst=(child.node_id, edge.dst),
                                direction=disp,
                                min_length=max(1, edge.weight),
                                associated_rooms=frozenset({edge.src}),
                                associated_children=frozenset({child.node_id}),
                            )
                        )
                        gateway_anchors[child.node_id].add(edge.src)
                        aligned_children.add(child.node_id)
                        break
                    except ValueError:
                        pass
                elif edge.dst in room_set and edge.src in c_coords:
                    try:
                        disp = direction_to_displacement(edge.direction)
                        active_edges.append(
                            ActiveEdge(
                                edge_id=f"gw_{child.node_id}_{edge.src}_{edge.dst}",
                                src=(child.node_id, edge.src),
                                dst=edge.dst,
                                direction=disp,
                                min_length=max(1, edge.weight),
                                associated_rooms=frozenset({edge.dst}),
                                associated_children=frozenset({child.node_id}),
                            )
                        )
                        gateway_anchors[child.node_id].add(edge.dst)
                        aligned_children.add(child.node_id)
                        break
                    except ValueError:
                        pass

        # Child-child bridge alignment
        if len(children) > 1:
            for child in children:
                if child.node_id in aligned_children:
                    continue
                c_sol = child_solutions.get(child.node_id)
                if not c_sol:
                    continue
                c_coords = c_sol.local_coordinates

                contract = child.cut_contract
                if isinstance(contract, BridgeCut):
                    u, v = contract.bridge_edge
                    for other in children:
                        if other.node_id == child.node_id or other.node_id not in aligned_children:
                            continue
                        other_coords = (
                            child_solutions[other.node_id].local_coordinates
                            if other.node_id in child_solutions
                            else {}
                        )
                        if u in other_coords and v in c_coords:
                            try:
                                disp = direction_to_displacement(contract.direction)
                                active_edges.append(
                                    ActiveEdge(
                                        edge_id=f"ccb_{other.node_id}_{child.node_id}_{u}_{v}",
                                        src=(other.node_id, u),
                                        dst=(child.node_id, v),
                                        direction=disp,
                                        min_length=max(1, contract.weight),
                                        associated_rooms=frozenset(),
                                        associated_children=frozenset({other.node_id, child.node_id}),
                                    )
                                )
                                aligned_children.add(child.node_id)
                                break
                            except ValueError:
                                pass
                        elif v in other_coords and u in c_coords:
                            try:
                                disp = direction_to_displacement(contract.direction)
                                active_edges.append(
                                    ActiveEdge(
                                        edge_id=f"ccb_{child.node_id}_{other.node_id}_{u}_{v}",
                                        src=(child.node_id, u),
                                        dst=(other.node_id, v),
                                        direction=disp,
                                        min_length=max(1, contract.weight),
                                        associated_rooms=frozenset(),
                                        associated_children=frozenset({child.node_id, other.node_id}),
                                    )
                                )
                                aligned_children.add(child.node_id)
                                break
                            except ValueError:
                                pass
                if child.node_id in aligned_children:
                    continue
                for edge in graph.edges:
                    if edge.cut:
                        continue
                    for other in children:
                        if other.node_id == child.node_id or other.node_id not in aligned_children:
                            continue
                        other_coords = (
                            child_solutions[other.node_id].local_coordinates
                            if other.node_id in child_solutions
                            else {}
                        )
                        if edge.src in other_coords and edge.dst in c_coords:
                            try:
                                disp = direction_to_displacement(edge.direction)
                                active_edges.append(
                                    ActiveEdge(
                                        edge_id=f"ccb_{other.node_id}_{child.node_id}_{edge.src}_{edge.dst}",
                                        src=(other.node_id, edge.src),
                                        dst=(child.node_id, edge.dst),
                                        direction=disp,
                                        min_length=max(1, edge.weight),
                                        associated_rooms=frozenset(),
                                        associated_children=frozenset({other.node_id, child.node_id}),
                                    )
                                )
                                aligned_children.add(child.node_id)
                                break
                            except ValueError:
                                pass
                        elif edge.dst in other_coords and edge.src in c_coords:
                            try:
                                disp = direction_to_displacement(edge.direction)
                                active_edges.append(
                                    ActiveEdge(
                                        edge_id=f"ccb_{child.node_id}_{other.node_id}_{edge.src}_{edge.dst}",
                                        src=(child.node_id, edge.src),
                                        dst=(other.node_id, edge.dst),
                                        direction=disp,
                                        min_length=max(1, edge.weight),
                                        associated_rooms=frozenset(),
                                        associated_children=frozenset({child.node_id, other.node_id}),
                                    )
                                )
                                aligned_children.add(child.node_id)
                                break
                            except ValueError:
                                pass
                    if child.node_id in aligned_children:
                        break

        # Internal room-room edges
        pole_edges_handled: set[tuple[int, int]] = set()
        if rooms:
            for e in graph.edges:
                if not e.cut and e.src in room_set and e.dst in room_set:
                    can_p = (min(e.src, e.dst), max(e.src, e.dst))
                    if can_p in split_pole_pairs:
                        if can_p in pole_edges_handled:
                            continue
                        pole_edges_handled.add(can_p)
                        edge_min_len = max(e.weight, split_min_lengths.get(can_p, 1))
                        try:
                            disp = direction_to_displacement(e.direction)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"int_{e.src}_{e.dst}",
                                    src=e.src,
                                    dst=e.dst,
                                    direction=disp,
                                    min_length=max(1, edge_min_len),
                                    associated_rooms=frozenset({e.src, e.dst}),
                                    associated_children=frozenset(
                                        c.node_id for c in split_children_by_poles.get(can_p, [])
                                    ),
                                )
                            )
                        except ValueError:
                            pass
                        continue
                    try:
                        disp = direction_to_displacement(e.direction)
                        active_edges.append(
                            ActiveEdge(
                                edge_id=f"int_{e.src}_{e.dst}",
                                src=e.src,
                                dst=e.dst,
                                direction=disp,
                                min_length=max(1, e.weight),
                                associated_rooms=frozenset({e.src, e.dst}),
                                associated_children=frozenset(),
                            )
                        )
                    except ValueError:
                        continue

        for (p0, p1), split_children in split_children_by_poles.items():
            can_p = (min(p0, p1), max(p0, p1))
            if can_p in pole_edges_handled:
                continue
            if can_p not in split_directions:
                continue
            d_vec = split_directions[can_p]
            pole_edges_handled.add(can_p)
            active_edges.append(
                ActiveEdge(
                    edge_id=f"sp_{p0}_{p1}",
                    src=p0,
                    dst=p1,
                    direction=d_vec,
                    min_length=split_min_lengths[can_p],
                    associated_rooms=frozenset({p0, p1}),
                    associated_children=frozenset({c.node_id for c in split_children}),
                )
            )

        edge_indices = list(range(len(active_edges)))
        if edge_indices:
            model.L = pyo.Var(
                edge_indices,
                domain=pyo.Integers,
                bounds=lambda m, i: (active_edges[i].min_length, m_coord),
            )
            model.edge_x = pyo.ConstraintList()
            model.edge_y = pyo.ConstraintList()
            model.edge_z = pyo.ConstraintList()
            for i, ae in enumerate(active_edges):
                src_x, src_y, src_z = get_point(ae.src)
                dst_x, dst_y, dst_z = get_point(ae.dst)
                dx, dy, dz = ae.direction
                model.edge_x.add(dst_x - src_x == model.L[i] * dx)
                model.edge_y.add(dst_y - src_y == model.L[i] * dy)
                model.edge_z.add(dst_z - src_z == model.L[i] * dz)

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
        rc_room_pairs: list[tuple[int, tuple[str, int]]] = []
        if rooms:
            for child in children:
                c_sol = child_solutions.get(child.node_id)
                if c_sol and c_sol.bounding_box is not None:
                    anchors = gateway_anchors[child.node_id]
                    is_split_child = (
                        child.cut_type == CutType.SPLIT_PAIR
                        or isinstance(child.cut_contract, SplitPairCut)
                        or child.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair")
                        or len(child.metadata.get("poles", ())) == 2
                    )
                    if is_split_child:
                        for u in rooms:
                            if u not in anchors and u not in c_sol.local_coordinates:
                                for v in c_sol.local_coordinates:
                                    if v not in room_set:
                                        rc_room_pairs.append((u, (child.node_id, v)))
                    else:
                        for u in rooms:
                            if u not in anchors and u not in c_sol.local_coordinates:
                                rc_pairs.append((u, child.node_id))

        if rc_pairs:
            add_disjunctive_box_separation(
                model=model,
                pairs=rc_pairs,
                get_box_a=_room_box,
                get_box_b=_child_box,
                prefix="rc",
                m_big=m_big,
            )

        if rc_room_pairs:
            def _child_room_box(target: tuple[str, int]) -> tuple[Any, Any, Any, Any, Any, Any]:
                px, py, pz = get_point(target)
                return (px, px, py, py, pz, pz)

            add_disjunctive_box_separation(
                model=model,
                pairs=rc_room_pairs,
                get_box_a=_room_box,
                get_box_b=_child_room_box,
                prefix="rcr",
                m_big=m_big,
            )

        def _corridor_interior_box(ae: ActiveEdge) -> tuple[Any, Any, Any, Any, Any, Any]:
            x1, y1, z1 = get_point(ae.src)
            x2, y2, z2 = get_point(ae.dst)
            dx, dy, dz = ae.direction
            if dx > 0:
                return (x1 + 1, x2 - 1, y1, y1, z1, z1)
            elif dx < 0:
                return (x2 + 1, x1 - 1, y1, y1, z1, z1)
            elif dy > 0:
                return (x1, x1, y1 + 1, y2 - 1, z1, z1)
            elif dy < 0:
                return (x1, x1, y2 + 1, y1 - 1, z1, z1)
            elif dz > 0:
                return (x1, x1, y1, y1, z1 + 1, z2 - 1)
            else:  # dz < 0
                return (x1, x1, y1, y1, z2 + 1, z1 - 1)

        def _corridor_full_box(ae: ActiveEdge) -> tuple[Any, Any, Any, Any, Any, Any]:
            x1, y1, z1 = get_point(ae.src)
            x2, y2, z2 = get_point(ae.dst)
            dx, dy, dz = ae.direction
            if dx > 0:
                return (x1, x2, y1, y1, z1, z1)
            elif dx < 0:
                return (x2, x1, y1, y1, z1, z1)
            elif dy > 0:
                return (x1, x1, y1, y2, z1, z1)
            elif dy < 0:
                return (x1, x1, y2, y1, z1, z1)
            elif dz > 0:
                return (x1, x1, y1, y1, z1, z2)
            else:  # dz < 0
                return (x1, x1, y1, y1, z2, z1)

        # Corridor vs. third-party room non-penetration
        corridor_room_pairs: list[tuple[int, int]] = []
        if rooms and edge_indices:
            for i, ae in enumerate(active_edges):
                if ae.edge_id.startswith("sp_"):
                    continue
                if sum(1 for d in ae.direction if d != 0) != 1:
                    continue
                for w in rooms:
                    if w in ae.associated_rooms:
                        continue
                    if any(
                        w in child_solutions[cid].local_coordinates
                        for cid in ae.associated_children
                        if cid in child_solutions
                    ):
                        continue
                    corridor_room_pairs.append((i, w))

        add_disjunctive_box_separation(
            model=model,
            pairs=corridor_room_pairs,
            get_box_a=lambda i: _corridor_interior_box(active_edges[i]),
            get_box_b=_room_box,
            prefix="cwr",
            m_big=m_big,
        )

        # Corridor vs. sibling child macro-box non-penetration
        corridor_child_pairs: list[tuple[int, str]] = []
        if edge_indices and len(children) > 0:
            for i, ae in enumerate(active_edges):
                if ae.edge_id.startswith("sp_"):
                    continue
                if sum(1 for d in ae.direction if d != 0) != 1:
                    continue
                for s_child in children:
                    s_id = s_child.node_id
                    if s_id in ae.associated_children:
                        continue
                    s_sol = child_solutions.get(s_id)
                    if not s_sol or s_sol.bounding_box is None:
                        continue
                    if _get_child_bbox_corners(s_sol.bounding_box) is None:
                        continue
                    if any(r in s_sol.local_coordinates for r in ae.associated_rooms):
                        continue
                    if any(
                        set(s_sol.local_coordinates.keys())
                        & set(child_solutions[c_other].local_coordinates.keys())
                        for c_other in ae.associated_children
                        if c_other in child_solutions
                    ):
                        continue
                    corridor_child_pairs.append((i, s_id))

        add_disjunctive_box_separation(
            model=model,
            pairs=corridor_child_pairs,
            get_box_a=lambda i: _corridor_full_box(active_edges[i]),
            get_box_b=_child_box,
            prefix="cws",
            m_big=m_big,
        )

        # Corridor crossing separation
        crossing_pairs: list[tuple[int, int]] = []
        for i1 in range(len(active_edges)):
            ae1 = active_edges[i1]
            if ae1.edge_id.startswith("sp_"):
                continue
            if sum(1 for d in ae1.direction if d != 0) != 1:
                continue
            act1 = "x" if ae1.direction[0] != 0 else ("y" if ae1.direction[1] != 0 else "z")
            for i2 in range(i1 + 1, len(active_edges)):
                ae2 = active_edges[i2]
                if ae2.edge_id.startswith("sp_"):
                    continue
                if sum(1 for d in ae2.direction if d != 0) != 1:
                    continue
                act2 = "x" if ae2.direction[0] != 0 else ("y" if ae2.direction[1] != 0 else "z")
                if act1 == act2:
                    continue
                incident = (
                    bool(ae1.associated_rooms & ae2.associated_rooms)
                    or bool({ae1.src, ae1.dst} & {ae2.src, ae2.dst})
                )
                if not incident:
                    crossing_pairs.append((i1, i2))

        add_disjunctive_box_separation(
            model=model,
            pairs=crossing_pairs,
            get_box_a=lambda i: _corridor_full_box(active_edges[i]),
            get_box_b=lambda i: _corridor_full_box(active_edges[i]),
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

        req = node.metadata.get("required_pole_span")
        if req:
            rp0, rp1, r_axis, r_sign, r_span = req
            if rp0 in rooms and rp1 in rooms:
                if r_axis == "x":
                    expr = (model.x[rp1] - model.x[rp0]) * r_sign >= r_span
                elif r_axis == "y":
                    expr = (model.y[rp1] - model.y[rp0]) * r_sign >= r_span
                else:
                    expr = (model.z[rp1] - model.z[rp0]) * r_sign >= r_span
                model.req_pole_span = pyo.Constraint(expr=expr)

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
        stretch_expr = (
            sum(
                10 * (model.L[i] - active_edges[i].min_length)
                for i in edge_indices
            )
            if edge_indices
            else 0
        )
        span_expr = (
            (model.x_max - model.x_min)
            + (model.y_max - model.y_min)
            + (model.z_max - model.z_min)
        )
        model.obj = pyo.Objective(
            expr=stretch_expr + span_expr,
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

            pole_groups: dict[tuple[int, int], list[SolveTreeNode]] = {}
            for c in node.children:
                contract = c.cut_contract
                if isinstance(contract, SplitPairCut):
                    p_can = (min(contract.poles), max(contract.poles))
                    if (
                        p_can[0] in child_solutions_map[c.node_id].local_coordinates
                        and p_can[1] in child_solutions_map[c.node_id].local_coordinates
                    ):
                        pole_groups.setdefault(p_can, []).append(c)

            cut_type = node.cut_type or node.metadata.get("cut_type")
            poles = tuple(node.metadata.get("poles", ()))
            if (cut_type == CutType.SPLIT_PAIR or cut_type == "split_pair") and len(poles) == 2:
                p_can = (min(poles[0], poles[1]), max(poles[0], poles[1]))
                for c in node.children:
                    if (
                        p_can[0] in child_solutions_map[c.node_id].local_coordinates
                        and p_can[1] in child_solutions_map[c.node_id].local_coordinates
                    ):
                        if c not in pole_groups.get(p_can, []):
                            pole_groups.setdefault(p_can, []).append(c)

            for (p0, p1), split_children in pole_groups.items():
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

            for (p0, p1), split_children in pole_groups.items():
                if p0 in parent_sol.local_coordinates and p1 in parent_sol.local_coordinates:
                    parent_span = max(
                        abs(parent_sol.local_coordinates[p1].x - parent_sol.local_coordinates[p0].x),
                        abs(parent_sol.local_coordinates[p1].y - parent_sol.local_coordinates[p0].y),
                        abs(parent_sol.local_coordinates[p1].z - parent_sol.local_coordinates[p0].z),
                    )
                    for c in split_children:
                        c_sol = child_solutions_map[c.node_id]
                        if p0 in c_sol.local_coordinates and p1 in c_sol.local_coordinates:
                            cur_span = max(
                                abs(c_sol.local_coordinates[p1].x - c_sol.local_coordinates[p0].x),
                                abs(c_sol.local_coordinates[p1].y - c_sol.local_coordinates[p0].y),
                                abs(c_sol.local_coordinates[p1].z - c_sol.local_coordinates[p0].z),
                            )
                            if cur_span < parent_span:
                                pdx = c_sol.local_coordinates[p1].x - c_sol.local_coordinates[p0].x
                                pdy = c_sol.local_coordinates[p1].y - c_sol.local_coordinates[p0].y
                                pdz = c_sol.local_coordinates[p1].z - c_sol.local_coordinates[p0].z
                                if abs(pdx) == cur_span:
                                    axis, sign = "x", (1 if pdx > 0 else -1)
                                elif abs(pdy) == cur_span:
                                    axis, sign = "y", (1 if pdy > 0 else -1)
                                else:
                                    axis, sign = "z", (1 if pdz > 0 else -1)
                                c.metadata["required_pole_span"] = (p0, p1, axis, sign, parent_span)

                                new_edges = [
                                    replace(e, weight=parent_span) if {e.src, e.dst} == {p0, p1} else e
                                    for e in g.edges
                                ]
                                g_sub = replace(g, edges=tuple(new_edges))

                                def _clear_subtree(sub_n: SolveTreeNode) -> None:
                                    solutions.pop(sub_n.node_id, None)
                                    for sub_c in sub_n.children:
                                        _clear_subtree(sub_c)

                                _clear_subtree(c)
                                new_c_sol = _solve_node(c, g_sub)
                                child_solutions_map[c.node_id] = new_c_sol
                                if p0 in parent_sol.local_coordinates and p0 in new_c_sol.local_coordinates:
                                    pw = parent_sol.local_coordinates[p0]
                                    pl = new_c_sol.local_coordinates[p0]
                                    parent_sol.metadata.setdefault("child_offsets", {})[c.node_id] = Coord3D(
                                        pw.x - pl.x,
                                        pw.y - pl.y,
                                        pw.z - pl.z,
                                    )

            solutions[node.node_id] = parent_sol
            return parent_sol

        _solve_node(tree.root, layout_graph)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp, layout_graph)

        return TreeSolution(node_solutions=solutions)
