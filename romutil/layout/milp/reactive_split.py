"""Reactive directional cut-set splitting and 3D deformation diagnostics for hierarchical MILP."""

from __future__ import annotations

import collections
import re
from typing import Any, Collection, Mapping, Sequence

import networkx as nx
import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutGraph,
    NodeSolution,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.layout.milp.clearance import ClearanceViolation
from romutil.layout.milp.cluster import UnresolvableCollisionError
from romutil.layout.milp.formulation import calculate_dynamic_bounds
from romutil.layout.milp.inventory import (
    ActiveEdge,
    ClusterInventory,
    EndpointRef,
    get_child_boundary_ports,
    get_child_coordinate,
    get_child_strictly_interior_rooms,
)
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Direction
from romutil.types import Coord3D

_DIAGNOSTIC_WEIGHT = 10_000.0
_CYCLE_WEIGHT_RE = re.compile(r"\(cycle weight\s+(\d+)\s*>\s*0\)")
_CARDINAL_DIRECTIONS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
_DIR_CODE_TO_AXIS: dict[str, int] = {
    "r": 0,
    "l": 0,
    "a": 1,
    "b": 1,
    "u": 2,
    "d": 2,
}


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


def _is_one_cut_node(node: SolveTreeNode) -> bool:
    """Return True if node uses a 1-cut contract (articulation point or bridge)."""
    if isinstance(node.cut_contract, (ArticulationPointCut, BridgeCut)):
        return True
    if "anchor_vnum" in node.metadata or "bridge_edge" in node.metadata:
        return True
    return node.metadata.get("cut_type") in (
        CutType.ARTICULATION_POINT,
        "articulation_point",
        CutType.BRIDGE,
        "bridge",
    )


def _is_two_cut_shell(node: SolveTreeNode) -> bool:
    """Return True if node is a nested 2-cut shell eligible for selective unpacking."""
    if _is_one_cut_node(node):
        return False
    if isinstance(node.cut_contract, SplitPairCut):
        return True
    if node.cut_type == CutType.SPLIT_PAIR:
        return True
    if node.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair") or "poles" in node.metadata:
        return True
    return node.node_type == TreeNodeType.SPLIT_PAIR_CONTAINER


def is_split_pair_origin(child: SolveTreeNode) -> bool:
    """Return True if child is an original SplitPairCut child or a piece split from one."""
    if child.metadata.get("reactive_split_origin") is not None:
        return True
    if isinstance(child.cut_contract, SplitPairCut):
        return True
    if child.cut_type == CutType.SPLIT_PAIR:
        return True
    if child.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair") or "poles" in child.metadata:
        return True
    return False


def _collect_core_candidate_rooms(node: SolveTreeNode) -> set[int]:
    """Collect rooms owned directly by node or inside its nested 2-cut shells."""
    rooms = set(node.rooms)
    for child in node.children:
        if _is_two_cut_shell(child):
            rooms |= _collect_core_candidate_rooms(child)
    return rooms


def get_child_interface_rooms(
    child: SolveTreeNode,
    inventory: ClusterInventory,
    child_solutions: Mapping[str, NodeSolution],
) -> tuple[list[int], dict[int, Coord3D]]:
    """Return distinct interface rooms I(C) touched by active edges or alignments and their local coords."""
    cid = child.node_id
    c_sol = child_solutions.get(cid)
    if c_sol is None:
        return [], {}

    ae_rooms: set[int] = set()
    for ae in inventory.active_edges:
        if ae.edge_id.startswith("sp_port_"):
            continue
        if ae.edge_id.startswith("sp_span_"):
            if cid in ae.associated_children:
                if isinstance(ae.src, int):
                    ae_rooms.add(ae.src)
                if isinstance(ae.dst, int):
                    ae_rooms.add(ae.dst)
            continue
        if isinstance(ae.src, tuple) and ae.src[0] == cid:
            ae_rooms.add(ae.src[1])
        if isinstance(ae.dst, tuple) and ae.dst[0] == cid:
            ae_rooms.add(ae.dst[1])

    align_poles: set[int] = set()
    b_ports = get_child_boundary_ports(child)
    for c_align in inventory.child_alignments:
        target_coord: Coord3D | None = None
        if c_align.base_child_id == cid:
            target_coord = c_align.base_coord
        elif c_align.target_child_id == cid:
            target_coord = c_align.target_coord
        if target_coord is None:
            continue
        for p in sorted(b_ports):
            p_loc = get_child_coordinate(c_sol, p)
            if p_loc == target_coord:
                align_poles.add(p)

    # Prefer active-edge internal rooms when >= 2; otherwise include alignment poles.
    if len(ae_rooms) >= 2:
        iface = sorted(ae_rooms)
    else:
        iface = sorted(ae_rooms | align_poles)

    coords: dict[int, Coord3D] = {}
    for r in iface:
        loc = get_child_coordinate(c_sol, r)
        if loc is not None:
            coords[r] = loc
    return [r for r in iface if r in coords], coords


def is_splittable_child(
    child: SolveTreeNode,
    inventory: ClusterInventory,
    child_solutions: Mapping[str, NodeSolution],
) -> bool:
    """Return True if child is eligible for reactive directional cut-set splitting."""
    if not is_split_pair_origin(child):
        return False
    core_rooms = _collect_core_candidate_rooms(child)
    if len(core_rooms) <= 1:
        return False
    iface_rooms, _ = get_child_interface_rooms(child, inventory, child_solutions)
    return len(iface_rooms) >= 2


def _match_alignment_pole(
    child: SolveTreeNode,
    c_sol: NodeSolution,
    coord: Coord3D,
) -> int | None:
    """Find boundary port of child matching coord."""
    for p in sorted(get_child_boundary_ports(child)):
        if get_child_coordinate(c_sol, p) == coord:
            return p
    return None


def _child_capable_axes(
    child: SolveTreeNode,
    iface: Collection[int],
    parent_attach_rooms: set[int],
    graph: LayoutGraph,
) -> set[int]:
    """Return cardinal axis indices present on potentially cuttable edges of child."""
    core_cand = _collect_core_candidate_rooms(child)
    if len(core_cand) <= 1:
        return set()
    orig_poles = set(get_child_boundary_ports(child))
    cand_ext = (orig_poles | set(iface) | parent_attach_rooms) - core_cand
    ext_neighbors: dict[int, set[int]] = collections.defaultdict(set)
    seen_undir: set[tuple[int, int]] = set()
    core_axes: set[int] = set()
    ext_edges: list[tuple[int, int]] = []

    for edge in graph.edges:
        if edge.cut or edge.src == edge.dst:
            continue
        ed = _safe_displacement(edge.direction)
        if ed is None:
            continue
        undir = (min(edge.src, edge.dst), max(edge.src, edge.dst))
        if undir in seen_undir:
            continue
        seen_undir.add(undir)
        ax = 0 if ed[0] != 0 else (1 if ed[1] != 0 else (2 if ed[2] != 0 else None))
        if ax is None:
            continue
        if edge.src in core_cand and edge.dst in core_cand:
            core_axes.add(ax)
        elif edge.src in core_cand and edge.dst in cand_ext:
            ext_neighbors[edge.dst].add(edge.src)
            ext_edges.append((edge.dst, ax))
        elif edge.dst in core_cand and edge.src in cand_ext:
            ext_neighbors[edge.src].add(edge.dst)
            ext_edges.append((edge.src, ax))

    for ext_p, ax in ext_edges:
        if len(ext_neighbors[ext_p]) >= 2:
            core_axes.add(ax)
    return core_axes


def compute_split_lp_signature(
    iface: Sequence[int],
    s_rooms: Collection[int],
    t_rooms: Collection[int],
    direction: tuple[int, int, int],
) -> tuple[int, int] | None:
    """Return (axis_idx, lp_sign) matching solve_3d_deformation_diagnostic's Delta reference."""
    if not iface:
        return None
    nz = [idx for idx, val in enumerate(direction) if val != 0]
    if len(nz) != 1:
        return None
    ax_idx = nz[0]
    dir_sign = 1 if direction[ax_idx] > 0 else -1
    ref_r = iface[0]
    if ref_r in s_rooms:
        return (ax_idx, dir_sign)
    if ref_r in t_rooms:
        return (ax_idx, -dir_sign)
    return None


def solve_3d_deformation_diagnostic(
    inventory: ClusterInventory,
    splittable_children: Sequence[SolveTreeNode],
    solver_name: str = "cbc",
    graph: LayoutGraph | None = None,
    failed_splits: set[tuple[str, int, int]] | None = None,
) -> list[tuple[str, set[int], set[int], tuple[int, int, int]]]:
    """Solve lightweight 3D interface-room deformation diagnostic LP to detect split axes and signs."""
    if not splittable_children:
        return []

    child_by_id = {c.node_id: c for c in inventory.children}
    child_interfaces: dict[str, list[int]] = {}
    child_iface_coords: dict[str, dict[int, Coord3D]] = {}
    for c in splittable_children:
        iface, coords = get_child_interface_rooms(c, inventory, inventory.child_solutions)
        if len(iface) >= 2:
            child_interfaces[c.node_id] = iface
            child_iface_coords[c.node_id] = coords

    if not child_interfaces:
        return []

    m_coord = max(50, calculate_dynamic_bounds(inventory))
    bound = 10 * m_coord
    model = pyo.ConcreteModel()

    child_strictly_interior: set[int] = set()
    for child in inventory.children:
        child_strictly_interior.update(
            get_child_strictly_interior_rooms(child, inventory.child_solutions.get(child.node_id))
        )
    external_ports = {p for p in inventory.boundary_ports if p not in child_strictly_interior}
    all_local_rooms = set(inventory.rooms) | external_ports
    local_rooms = tuple(sorted(all_local_rooms))

    if local_rooms:
        model.x = pyo.Var(local_rooms, domain=pyo.Reals, bounds=(-bound, bound), initialize=0.0)
        model.y = pyo.Var(local_rooms, domain=pyo.Reals, bounds=(-bound, bound), initialize=0.0)
        model.z = pyo.Var(local_rooms, domain=pyo.Reals, bounds=(-bound, bound), initialize=0.0)
        anchor = min(inventory.rooms) if inventory.rooms else min(external_ports)
        model.x[anchor].fix(0.0)
        model.y[anchor].fix(0.0)
        model.z[anchor].fix(0.0)

    child_ids = [c.node_id for c in inventory.children]
    model.cx = pyo.Var(child_ids, domain=pyo.Reals, bounds=(-bound, bound), initialize=0.0)
    model.cy = pyo.Var(child_ids, domain=pyo.Reals, bounds=(-bound, bound), initialize=0.0)
    model.cz = pyo.Var(child_ids, domain=pyo.Reals, bounds=(-bound, bound), initialize=0.0)
    if not local_rooms and child_ids:
        first_c = child_ids[0]
        model.cx[first_c].fix(0.0)
        model.cy[first_c].fix(0.0)
        model.cz[first_c].fix(0.0)

    # Continuous deformation variables Delta[C, r, ax] = dp - dm for r in I(C)[1:]
    delta_keys: list[tuple[str, int, str]] = []
    for cid, iface in child_interfaces.items():
        for r in iface[1:]:
            for ax in ("x", "y", "z"):
                delta_keys.append((cid, r, ax))

    model.dp = pyo.Var(delta_keys, domain=pyo.NonNegativeReals, initialize=0.0)
    model.dm = pyo.Var(delta_keys, domain=pyo.NonNegativeReals, initialize=0.0)
    delta_key_set = set(delta_keys)

    unit_dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    axis_names = ("x", "y", "z")
    failed_set = set(failed_splits) if failed_splits else set()
    capable_axes_by_cid: dict[str, set[int]] = {}
    if graph is not None:
        for cid, iface in child_interfaces.items():
            capable_axes_by_cid[cid] = _child_capable_axes(
                child_by_id[cid], iface, all_local_rooms, graph
            )

    for cid, iface in child_interfaces.items():
        r0 = iface[0]
        c_node = child_by_id[cid]
        cap_axes = capable_axes_by_cid.get(cid, {0, 1, 2})
        for r in iface[1:]:
            for ax_idx, ax_name in enumerate(axis_names):
                if ax_idx not in cap_axes:
                    model.dp[cid, r, ax_name].fix(0.0)
                    model.dm[cid, r, ax_name].fix(0.0)
                    continue
                u_dir = unit_dirs[ax_idx]
                allow_dp = (cid, ax_idx, 1) not in failed_set
                allow_dm = (cid, ax_idx, -1) not in failed_set
                if graph is not None:
                    if allow_dp and _find_directional_cut(
                        c_node, {r0}, {r}, u_dir, all_local_rooms, graph
                    ) is None:
                        allow_dp = False
                    if allow_dm and _find_directional_cut(
                        c_node, {r}, {r0}, u_dir, all_local_rooms, graph
                    ) is None:
                        allow_dm = False
                if not allow_dp:
                    model.dp[cid, r, ax_name].fix(0.0)
                if not allow_dm:
                    model.dm[cid, r, ax_name].fix(0.0)

    def _delta_expr(cid: str, r: int, ax: str) -> Any:
        if (cid, r, ax) in delta_key_set:
            return model.dp[cid, r, ax] - model.dm[cid, r, ax]
        return 0.0

    def _diag_point(ref: EndpointRef) -> tuple[Any, Any, Any]:
        if isinstance(ref, int):
            return (model.x[ref], model.y[ref], model.z[ref])
        cid, v = ref
        c_sol = inventory.child_solutions[cid]
        loc = get_child_coordinate(c_sol, v)
        if loc is None:
            loc = c_sol.local_coordinates[v]
        return (
            model.cx[cid] + loc.x + _delta_expr(cid, v, "x"),
            model.cy[cid] + loc.y + _delta_expr(cid, v, "y"),
            model.cz[cid] + loc.z + _delta_expr(cid, v, "z"),
        )

    # Child-child pole alignment constraints; relax secondary pole when p0, p1 not in all_local_rooms.
    model.align = pyo.ConstraintList()
    seen_child_pairs: set[tuple[str, str]] = set()
    for c_align in inventory.child_alignments:
        b_id = c_align.base_child_id
        t_id = c_align.target_child_id
        pair_key = (b_id, t_id)
        is_secondary = pair_key in seen_child_pairs
        seen_child_pairs.add(pair_key)

        b_child = child_by_id.get(b_id)
        t_child = child_by_id.get(t_id)
        b_sol = inventory.child_solutions.get(b_id)
        t_sol = inventory.child_solutions.get(t_id)
        b_pole = _match_alignment_pole(b_child, b_sol, c_align.base_coord) if b_child and b_sol else None
        t_pole = _match_alignment_pole(t_child, t_sol, c_align.target_coord) if t_child and t_sol else None

        def _resolve_pole_delta_room(cid_k: str, pole_v: int | None, p_coord: Coord3D) -> int | None:
            if pole_v is None:
                return None
            c_iface = child_interfaces.get(cid_k)
            if not c_iface:
                return pole_v
            if pole_v in c_iface:
                return pole_v
            c_coords_k = child_iface_coords[cid_k]
            return min(
                c_iface,
                key=lambda r: (
                    abs(c_coords_k[r].x - p_coord.x)
                    + abs(c_coords_k[r].y - p_coord.y)
                    + abs(c_coords_k[r].z - p_coord.z),
                    r,
                ),
            )

        relax_pole = (
            is_secondary
            and b_pole is not None
            and t_pole is not None
            and b_pole not in all_local_rooms
            and t_pole not in all_local_rooms
        )
        b_delta_r = _resolve_pole_delta_room(b_id, b_pole, c_align.base_coord)
        t_delta_r = _resolve_pole_delta_room(t_id, t_pole, c_align.target_coord)
        for ax, val_attr in (("x", "x"), ("y", "y"), ("z", "z")):
            base_expr = getattr(model, f"c{ax}")[b_id] + getattr(c_align.base_coord, val_attr)
            target_expr = getattr(model, f"c{ax}")[t_id] + getattr(c_align.target_coord, val_attr)
            if relax_pole and b_delta_r is not None and t_delta_r is not None:
                base_expr = base_expr + _delta_expr(b_id, b_delta_r, ax)
                target_expr = target_expr + _delta_expr(t_id, t_delta_r, ax)
            model.align.add(base_expr == target_expr)

    # Strictly cardinal ActiveEdge constraints (zero transverse slack, zero Z offset on planar edges)
    edge_indices = list(range(len(inventory.active_edges)))
    model.sp_span_slack_p = pyo.VarList(domain=pyo.NonNegativeReals)
    model.sp_span_slack_m = pyo.VarList(domain=pyo.NonNegativeReals)
    if edge_indices:
        model.L = pyo.Var(
            edge_indices,
            domain=pyo.Reals,
            bounds=lambda _m, i: (inventory.active_edges[i].min_length, bound),
        )
        model.edge_eq = pyo.ConstraintList()
        for i, ae in enumerate(inventory.active_edges):
            src_x, src_y, src_z = _diag_point(ae.src)
            dst_x, dst_y, dst_z = _diag_point(ae.dst)
            dx, dy, dz = ae.direction
            is_sp_span = ae.edge_id.startswith("sp_span_")

            if dx != 0:
                model.edge_eq.add(dst_x - src_x == model.L[i] * dx)
            elif (is_sp_span or dz != 0) and ae.allow_transverse_slack and (dy != 0 or dz != 0):
                sp = model.sp_span_slack_p.add()
                sm = model.sp_span_slack_m.add()
                model.edge_eq.add(dst_x - src_x == sp - sm)
            else:
                model.edge_eq.add(dst_x - src_x == 0)

            if dy != 0:
                model.edge_eq.add(dst_y - src_y == model.L[i] * dy)
            elif (is_sp_span or dz != 0) and ae.allow_transverse_slack and (dx != 0 or dz != 0):
                sp = model.sp_span_slack_p.add()
                sm = model.sp_span_slack_m.add()
                model.edge_eq.add(dst_y - src_y == sp - sm)
            else:
                model.edge_eq.add(dst_y - src_y == 0)

            if dz != 0:
                model.edge_eq.add(dst_z - src_z == model.L[i] * dz)
            else:
                model.edge_eq.add(dst_z - src_z == 0)

    # Objective: W_diag * sum(dp + dm) + normal piecewise stretch
    stretch_expr: Any = 0.0
    if edge_indices:
        coord_unit = max(1, int(getattr(inventory, "coord_unit", 1)))
        model.s_stretch_1 = pyo.Var(edge_indices, domain=pyo.NonNegativeReals, bounds=(0, 2 * coord_unit))
        model.s_stretch_2 = pyo.Var(edge_indices, domain=pyo.NonNegativeReals, bounds=(0, 4 * coord_unit))
        model.s_stretch_3 = pyo.Var(edge_indices, domain=pyo.NonNegativeReals, bounds=(0, None))
        model.stretch_decompose = pyo.ConstraintList()
        for i in edge_indices:
            min_len = inventory.active_edges[i].min_length
            model.stretch_decompose.add(
                model.L[i] - min_len
                == model.s_stretch_1[i] + model.s_stretch_2[i] + model.s_stretch_3[i]
            )
        stretch_expr = sum(
            10.0 * model.s_stretch_1[i]
            + 30.0 * model.s_stretch_2[i]
            + 100.0 * model.s_stretch_3[i]
            for i in edge_indices
        )

    # Deterministic tie-breaker across children prefers larger core children so LP does not split 0.5/0.5.
    ordered_cids = sorted(
        child_interfaces.keys(),
        key=lambda cid: (-len(_collect_core_candidate_rooms(child_by_id[cid])), -len(_collect_subtree_rooms(child_by_id[cid])), cid),
    )
    weight_mult = {cid: 1.0 + 1e-4 * idx for idx, cid in enumerate(ordered_cids)}
    diag_penalty = sum(
        _DIAGNOSTIC_WEIGHT * weight_mult[cid] * (model.dp[cid, r, ax] + model.dm[cid, r, ax])
        for (cid, r, ax) in delta_keys
    )
    sp_slack_penalty = (
        sum(1e5 * (model.sp_span_slack_p[k] + model.sp_span_slack_m[k]) for k in model.sp_span_slack_p)
        if len(model.sp_span_slack_p) > 0
        else 0.0
    )
    model.obj = pyo.Objective(expr=diag_penalty + stretch_expr + sp_slack_penalty, sense=pyo.minimize)

    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        return []
    if solver_name == "cbc":
        solver.options["seconds"] = 5
        solver.options["timeMode"] = "elapsed"
    elif solver_name == "glpk":
        solver.options["tmlim"] = 5

    try:
        results = solver.solve(model, tee=False)
    except Exception:
        return []

    term = getattr(getattr(results, "solver", None), "termination_condition", None)
    status = getattr(getattr(results, "solver", None), "status", None)
    if term not in (TerminationCondition.optimal, TerminationCondition.feasible) and status != PyomoSolverStatus.ok:
        return []

    requests: list[tuple[str, set[int], set[int], tuple[int, int, int]]] = []
    axis_names = ("x", "y", "z")
    unit_dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    for cid in ordered_cids:
        iface = child_interfaces[cid]
        deltas: dict[int, list[float]] = {iface[0]: [0.0, 0.0, 0.0]}
        for r in iface[1:]:
            vals = []
            for ax in axis_names:
                vp = pyo.value(model.dp[cid, r, ax], exception=False) or 0.0
                vm = pyo.value(model.dm[cid, r, ax], exception=False) or 0.0
                vals.append(float(vp) - float(vm))
            deltas[r] = vals

        best_ax: int | None = None
        best_spread = 0.5
        best_min = 0.0
        best_max = 0.0
        for ax_idx in (0, 1, 2):
            ax_vals = [deltas[r][ax_idx] for r in iface]
            lo = min(ax_vals)
            hi = max(ax_vals)
            spread = hi - lo
            if spread > best_spread + 1e-6:
                best_spread = spread
                best_ax = ax_idx
                best_min = lo
                best_max = hi

        if best_ax is None:
            continue

        mid = 0.5 * (best_min + best_max)
        s_rooms = {r for r in iface if deltas[r][best_ax] < mid}
        t_rooms = {r for r in iface if deltas[r][best_ax] >= mid}
        if s_rooms and t_rooms:
            requests.append((cid, s_rooms, t_rooms, unit_dirs[best_ax]))

    return requests


def generate_coordinate_threshold_partitions(
    child_sol: NodeSolution | None,
    s_rooms: set[int],
    t_rooms: set[int],
    direction: tuple[int, int, int],
) -> list[tuple[set[int], set[int], tuple[int, int, int]]]:
    """Generate ordered coordinate threshold partitions when an LP partition has no cut.

    When a splittable piece exposes 3 or more interface rooms, continuous diagnostic
    variables may move a subset of co-planar/aligned interface rooms relative to others
    across a degenerate edge alignment. Ordering interface rooms by their solved local
    coordinates along the same axis yields valid geometric bisections that honor the
    underlying deformation axis while exposing cuttable internal bridge edges.
    """
    if child_sol is None:
        return []
    iface = s_rooms | t_rooms
    if len(iface) < 3:
        return []
    nz = [idx for idx, val in enumerate(direction) if val != 0]
    if len(nz) != 1:
        return []
    ax = nz[0]

    coord_vals: dict[int, int] = {}
    for r in iface:
        pt = get_child_coordinate(child_sol, r)
        if pt is None:
            return []
        coord_vals[r] = (pt.x, pt.y, pt.z)[ax]

    distinct_vals = sorted(set(coord_vals.values()))
    if len(distinct_vals) < 2:
        return []

    unit_pos = (
        (1, 0, 0) if ax == 0 else ((0, 1, 0) if ax == 1 else (0, 0, 1))
    )
    unit_neg = (-unit_pos[0], -unit_pos[1], -unit_pos[2])

    scored_thresholds: list[tuple[int, int, int, set[int], set[int]]] = []
    for idx in range(len(distinct_vals) - 1):
        thr = 0.5 * (distinct_vals[idx] + distinct_vals[idx + 1])
        cand_s = {r for r in iface if coord_vals[r] < thr}
        cand_t = {r for r in iface if coord_vals[r] >= thr}
        if not cand_s or not cand_t:
            continue
        overlap_same = len(cand_s & s_rooms) + len(cand_t & t_rooms)
        overlap_flip = len(cand_s & t_rooms) + len(cand_t & s_rooms)
        best_match = max(overlap_same, overlap_flip)
        imbalance = abs(len(cand_s) - len(cand_t))
        scored_thresholds.append((-best_match, imbalance, idx, cand_s, cand_t))

    scored_thresholds.sort(key=lambda item: (item[0], item[1], item[2]))

    primary_sig = (
        frozenset(s_rooms),
        frozenset(t_rooms),
        direction,
    )
    seen: set[tuple[frozenset[int], frozenset[int], tuple[int, int, int]]] = {primary_sig}
    candidates: list[tuple[set[int], set[int], tuple[int, int, int]]] = []
    for cand_dir in (unit_pos, unit_neg):
        for _neg_m, _imb, _idx, cand_s, cand_t in scored_thresholds:
            sig = (frozenset(cand_s), frozenset(cand_t), cand_dir)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append((cand_s, cand_t, cand_dir))
    return candidates


def _child_covered_exit_axes(
    child: SolveTreeNode,
    inventory: ClusterInventory | None,
    graph: LayoutGraph,
) -> tuple[set[int], set[int]]:
    """Return (axes_with_any_exit, axes_covering_both_poles) for child's boundary exits."""
    cid = child.node_id
    axes_per_room: dict[int, set[int]] = collections.defaultdict(set)
    any_axes: set[int] = set()

    if inventory is not None:
        for ae in inventory.active_edges:
            if ae.edge_id.startswith("sp_"):
                continue
            dx, dy, dz = ae.direction
            ax = 0 if dx != 0 else (1 if dy != 0 else (2 if dz != 0 else None))
            if ax is None:
                continue
            if isinstance(ae.src, tuple) and ae.src[0] == cid:
                any_axes.add(ax)
                axes_per_room[ae.src[1]].add(ax)
            if isinstance(ae.dst, tuple) and ae.dst[0] == cid:
                any_axes.add(ax)
                axes_per_room[ae.dst[1]].add(ax)
    else:
        sub = _collect_subtree_rooms(child)
        for edge in graph.edges:
            if edge.cut:
                continue
            disp = _safe_displacement(edge.direction)
            if disp is None:
                continue
            ax = 0 if disp[0] != 0 else (1 if disp[1] != 0 else (2 if disp[2] != 0 else None))
            if ax is None:
                continue
            if edge.src in sub and edge.dst not in sub:
                any_axes.add(ax)
                axes_per_room[edge.src].add(ax)
            elif edge.dst in sub and edge.src not in sub:
                any_axes.add(ax)
                axes_per_room[edge.dst].add(ax)

    both_axes: set[int] = set()
    if len(axes_per_room) >= 2:
        for ax in (0, 1, 2):
            if all(ax in ax_set for ax_set in axes_per_room.values()):
                both_axes.add(ax)
    return any_axes, both_axes


def diagnose_unresolvable_collision(
    exc: UnresolvableCollisionError,
    effective_children: Sequence[SolveTreeNode],
    inventory: ClusterInventory | None,
    child_solutions: Mapping[str, NodeSolution],
    graph: LayoutGraph,
) -> list[tuple[str, set[int], set[int], tuple[int, int, int]]]:
    """Rank candidate directional splits for colliding splittable children after UnresolvableCollisionError."""
    if inventory is None:
        return []

    colliding_cids = {
        ent_id
        for kind, ent_id in exc.colliding_entities
        if kind == "child" and isinstance(ent_id, str)
    }
    if not colliding_cids:
        for viol in exc.collisions:
            for item in (viol.item_a, viol.item_b):
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and item[0] != "edge":
                    colliding_cids.add(item[0])

    splittable_colliding = [
        c
        for c in effective_children
        if c.node_id in colliding_cids and is_splittable_child(c, inventory, child_solutions)
    ]
    if not splittable_colliding:
        return []

    # Parse minimum DAG contradiction cycle weights per axis across pruned_reasons.
    axis_min_weight: dict[int, int] = {}
    for reason_map in exc.pruned_reasons:
        for d_code, reason_str in reason_map.items():
            ax = _DIR_CODE_TO_AXIS.get(d_code)
            if ax is None:
                continue
            m = _CYCLE_WEIGHT_RE.search(reason_str)
            weight = int(m.group(1)) if m else 1
            if ax not in axis_min_weight or weight < axis_min_weight[ax]:
                axis_min_weight[ax] = weight

    # Pick largest colliding splittable child first (by core rooms, then subtree rooms).
    splittable_colliding.sort(
        key=lambda c: (
            -len(_collect_core_candidate_rooms(c)),
            -len(_collect_subtree_rooms(c)),
            c.node_id,
        )
    )

    candidates: list[tuple[str, set[int], set[int], tuple[int, int, int]]] = []
    unit_dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for child in splittable_colliding:
        cid = child.node_id
        iface, coords = get_child_interface_rooms(child, inventory, child_solutions)
        if len(iface) < 2:
            continue
        sol = child_solutions.get(cid)
        ports = get_child_boundary_ports(child)
        port_coords: dict[int, Coord3D] = {}
        if sol is not None:
            port_coords.update(sol.port_coordinates)
            if "port_coordinates" in sol.metadata:
                port_coords.update(sol.metadata["port_coordinates"])
            arc = sol.metadata.get("all_room_coordinates")
            if isinstance(arc, Mapping):
                for p in ports:
                    if p not in port_coords and p in arc:
                        port_coords[p] = arc[p]
        any_covered, both_covered = _child_covered_exit_axes(child, inventory, graph)
        ordered_axes = sorted(
            (0, 1, 2),
            key=lambda ax: (
                1 if ax in any_covered else 0,
                1 if ax in both_covered else 0,
                axis_min_weight.get(ax, 1_000_000),
                1 if ax == 2 else 0,
                ax,
            ),
        )
        for ax in ordered_axes:
            def _ax_val(c3: Coord3D, _ax: int = ax) -> int:
                return c3.x if _ax == 0 else (c3.y if _ax == 1 else c3.z)

            vals = sorted({_ax_val(coords[r]) for r in iface})
            u_dir = unit_dirs[ax]
            if len(vals) >= 2:
                lo, hi = vals[0], vals[-1]
                mid = 0.5 * (lo + hi)
                thresholds = [0.5 * (vals[i] + vals[i + 1]) for i in range(len(vals) - 1)]

                def _pinch_score(thr: float) -> tuple[int, float]:
                    s_set = {r for r in iface if _ax_val(coords[r]) < thr}
                    t_set = set(iface) - s_set
                    pinches = 0
                    for p in ports:
                        pc = port_coords.get(p)
                        if pc is None:
                            continue
                        pv = _ax_val(pc)
                        for grp in (s_set, t_set):
                            has_lo = any(_ax_val(coords[r]) < pv for r in grp)
                            has_hi = any(_ax_val(coords[r]) > pv for r in grp)
                            if has_lo and has_hi:
                                pinches += 1
                    return (pinches, abs(thr - mid))

                thresholds.sort(key=_pinch_score)
                for thr in thresholds:
                    s_rooms = {r for r in iface if _ax_val(coords[r]) < thr}
                    t_rooms = set(iface) - s_rooms
                    if not s_rooms or not t_rooms:
                        continue
                    candidates.append((cid, s_rooms, t_rooms, u_dir))
                    candidates.append((cid, s_rooms, t_rooms, (-u_dir[0], -u_dir[1], -u_dir[2])))
            else:
                s_rooms = {iface[0]}
                t_rooms = set(iface[1:])
                if s_rooms and t_rooms:
                    candidates.append((cid, s_rooms, t_rooms, u_dir))
                    candidates.append((cid, s_rooms, t_rooms, (-u_dir[0], -u_dir[1], -u_dir[2])))

    return candidates


def detect_planar_transverse_shear(
    cluster_sol: NodeSolution,
    inventory: ClusterInventory,
    effective_children: Sequence[SolveTreeNode],
    child_solutions: Mapping[str, NodeSolution],
) -> list[tuple[ActiveEdge, int, int]]:
    """Return list of (ActiveEdge, shear_axis, shear_delta) for planar edges touching splittable children."""
    child_by_id = {c.node_id: c for c in effective_children}
    splittable_ids = {
        c.node_id
        for c in effective_children
        if is_splittable_child(c, inventory, child_solutions)
    }
    split_origin_by_cid = {
        c.node_id: c.metadata.get("reactive_split_origin")
        for c in effective_children
        if c.metadata.get("reactive_split_origin") is not None
    }

    child_offsets = cluster_sol.metadata.get("child_offsets", {})

    def _resolve_coord(ref: EndpointRef) -> Coord3D | None:
        if isinstance(ref, int):
            return cluster_sol.local_coordinates.get(ref) or cluster_sol.port_coordinates.get(ref)
        cid, r = ref
        c_sol = child_solutions.get(cid)
        if c_sol is None:
            return None
        loc = get_child_coordinate(c_sol, r)
        if loc is None:
            return None
        off = _as_coord3d(child_offsets.get(cid))
        return Coord3D(off.x + loc.x, off.y + loc.y, off.z + loc.z)

    sheared: list[tuple[ActiveEdge, int, int]] = []
    for ae in inventory.active_edges:
        if ae.edge_id.startswith("sp_"):
            continue
        dx, dy, dz = ae.direction
        if dz != 0 or abs(dx) + abs(dy) != 1:
            continue

        touches_splittable = any(cid in splittable_ids for cid in ae.associated_children)
        connects_split_siblings = False
        if len(ae.associated_children) == 2:
            c1, c2 = tuple(ae.associated_children)
            o1 = split_origin_by_cid.get(c1)
            o2 = split_origin_by_cid.get(c2)
            if o1 is not None and o1 == o2 and (c1 in splittable_ids or c2 in splittable_ids):
                connects_split_siblings = True
        if not (touches_splittable or connects_split_siblings):
            continue

        p_src = _resolve_coord(ae.src)
        p_dst = _resolve_coord(ae.dst)
        if p_src is None or p_dst is None:
            continue

        dvx = p_dst.x - p_src.x
        dvy = p_dst.y - p_src.y
        dvz = p_dst.z - p_src.z
        if dx != 0:
            if dvy != 0:
                sheared.append((ae, 1, dvy))
            if dvz != 0:
                sheared.append((ae, 2, dvz))
        elif dy != 0:
            if dvx != 0:
                sheared.append((ae, 0, dvx))
            if dvz != 0:
                sheared.append((ae, 2, dvz))

    # Deterministic order: largest absolute shear first, then edge_id.
    sheared.sort(key=lambda item: (-abs(item[2]), item[0].edge_id, item[1]))
    # Filter so at least one endpoint belongs to an existing child in child_by_id
    return [
        item
        for item in sheared
        if any(cid in child_by_id for cid in item[0].associated_children)
    ]


def compute_total_edge_length(
    cluster_sol: NodeSolution,
    inventory: ClusterInventory,
    child_solutions: Mapping[str, NodeSolution],
) -> int:
    """Compute total Manhattan edge length across all non-synthetic ActiveEdges in cluster_sol."""
    child_offsets = cluster_sol.metadata.get("child_offsets", {})

    def _resolve_coord(ref: EndpointRef) -> Coord3D | None:
        if isinstance(ref, int):
            return cluster_sol.local_coordinates.get(ref) or cluster_sol.port_coordinates.get(ref)
        cid, r = ref
        c_sol = child_solutions.get(cid)
        if c_sol is None:
            return None
        loc = get_child_coordinate(c_sol, r)
        if loc is None:
            return None
        off = _as_coord3d(child_offsets.get(cid))
        return Coord3D(off.x + loc.x, off.y + loc.y, off.z + loc.z)

    total = 0
    for ae in inventory.active_edges:
        if ae.edge_id.startswith("sp_"):
            continue
        p1 = _resolve_coord(ae.src)
        p2 = _resolve_coord(ae.dst)
        if p1 is not None and p2 is not None:
            total += abs(p2.x - p1.x) + abs(p2.y - p1.y) + abs(p2.z - p1.z)
    return total


def diagnose_transverse_shear(
    cluster_sol: NodeSolution,
    inventory: ClusterInventory,
    effective_children: Sequence[SolveTreeNode],
    child_solutions: Mapping[str, NodeSolution],
    solver_name: str = "cbc",
    graph: LayoutGraph | None = None,
    failed_splits: set[tuple[str, int, int]] | None = None,
) -> list[tuple[str, set[int], set[int], tuple[int, int, int]]]:
    """Generate split requests from sheared planar ActiveEdges (trying 3D diagnostic LP first)."""
    splittable_children = [
        c
        for c in effective_children
        if is_splittable_child(c, inventory, child_solutions)
    ]
    if not splittable_children:
        return []

    eff_graph = graph if graph is not None else getattr(inventory, "graph", None)
    diag_requests = solve_3d_deformation_diagnostic(
        inventory=inventory,
        splittable_children=splittable_children,
        solver_name=solver_name,
        graph=eff_graph,
        failed_splits=failed_splits,
    )
    if diag_requests:
        return diag_requests

    sheared = detect_planar_transverse_shear(
        cluster_sol=cluster_sol,
        inventory=inventory,
        effective_children=effective_children,
        child_solutions=child_solutions,
    )
    if not sheared:
        return []

    splittable_by_id = {c.node_id: c for c in splittable_children}
    unit_dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    failed_set = set(failed_splits) if failed_splits else set()
    candidates: list[tuple[str, set[int], set[int], tuple[int, int, int]]] = []

    for ae, shear_ax, shear_val in sheared:
        u_dir = unit_dirs[shear_ax]
        neg_dir = (-u_dir[0], -u_dir[1], -u_dir[2])
        # Endpoint movement needed to eliminate shear (dst - src == 0):
        # src endpoint must shift by +shear_val; dst endpoint must shift by -shear_val.
        endpoint_specs: list[tuple[str, int, int]] = []
        if isinstance(ae.src, tuple) and ae.src[0] in splittable_by_id:
            endpoint_specs.append((ae.src[0], ae.src[1], shear_val))
        if isinstance(ae.dst, tuple) and ae.dst[0] in splittable_by_id:
            endpoint_specs.append((ae.dst[0], ae.dst[1], -shear_val))

        for cid, r_edge, req_shift in endpoint_specs:
            iface, _ = get_child_interface_rooms(
                splittable_by_id[cid], inventory, child_solutions
            )
            other_iface = set(iface) - {r_edge}
            if not other_iface or r_edge not in iface:
                continue
            if req_shift > 0:
                s_rooms, t_rooms = other_iface, {r_edge}
            else:
                s_rooms, t_rooms = {r_edge}, other_iface
            for cand_dir in (u_dir, neg_dir):
                lp_sig = compute_split_lp_signature(iface, s_rooms, t_rooms, cand_dir)
                if lp_sig is not None and (cid, lp_sig[0], lp_sig[1]) in failed_set:
                    continue
                candidates.append((cid, s_rooms, t_rooms, cand_dir))

    return candidates


def _unpack_child_shells(
    child: SolveTreeNode,
    s_rooms: set[int],
    t_rooms: set[int],
    graph: LayoutGraph,
) -> tuple[set[int], list[SolveTreeNode], list[SolveTreeNode]]:
    """Selectively unpack nested 2-cut shells containing S/T rooms or lying on S-T paths."""
    core_rooms: set[int] = set()
    kept_sub_children: list[SolveTreeNode] = []
    unpacked_nodes: list[SolveTreeNode] = []
    targets = s_rooms | t_rooms

    def _unpack_node(cur: SolveTreeNode, cur_s: set[int], cur_t: set[int]) -> None:
        core_rooms.update(cur.rooms)
        two_cut_children: list[SolveTreeNode] = []
        for sc in cur.children:
            if _is_two_cut_shell(sc):
                two_cut_children.append(sc)
            else:
                kept_sub_children.append(sc)

        if not two_cut_children:
            return

        sc_subtrees = {sc.node_id: _collect_subtree_rooms(sc) for sc in two_cut_children}
        sc_poles: dict[str, tuple[int, int]] = {}
        for sc in two_cut_children:
            contract = sc.cut_contract
            if isinstance(contract, SplitPairCut):
                sc_poles[sc.node_id] = contract.poles
            elif "poles" in sc.metadata and len(sc.metadata["poles"]) == 2:
                raw_p = sc.metadata["poles"]
                sc_poles[sc.node_id] = (int(raw_p[0]), int(raw_p[1]))

        cur_poles = set(get_child_boundary_ports(cur)) - set(cur.rooms)
        skel_rooms = set(cur.rooms) | cur_poles
        for poles in sc_poles.values():
            skel_rooms.update(poles)

        g_skel = nx.Graph()
        for r in skel_rooms:
            g_skel.add_node(r)
        cur_pole_pair = frozenset(cur_poles) if len(cur_poles) == 2 else None
        for edge in graph.edges:
            if edge.cut:
                continue
            if edge.src in skel_rooms and edge.dst in skel_rooms and edge.src != edge.dst:
                if cur_pole_pair is not None and frozenset({edge.src, edge.dst}) == cur_pole_pair:
                    continue
                g_skel.add_edge(edge.src, edge.dst)

        for sc in two_cut_children:
            sc_p = sc_poles.get(sc.node_id)
            if sc_p is not None:
                g_skel.add_edge(sc_p[0], sc_p[1])

        s_anchor = "__S_SRC__"
        t_anchor = "__T_SNK__"
        g_skel.add_node(s_anchor)
        g_skel.add_node(t_anchor)

        def _attach_targets(anchor_name: str, tgt_rooms: set[int]) -> None:
            for r in tgt_rooms:
                if r in skel_rooms:
                    g_skel.add_edge(anchor_name, r)
                    continue
                for sc in two_cut_children:
                    if r in sc_subtrees[sc.node_id]:
                        for p in sc_poles.get(sc.node_id, ()):
                            g_skel.add_edge(anchor_name, p)

        _attach_targets(s_anchor, cur_s)
        _attach_targets(t_anchor, cur_t)
        g_skel.add_edge(s_anchor, t_anchor)

        bicomp_st: set[Any] = set()
        for comp in nx.biconnected_components(g_skel):
            if s_anchor in comp and t_anchor in comp:
                bicomp_st = set(comp)
                break

        for sc in two_cut_children:
            sub = sc_subtrees[sc.node_id]
            sc_p = sc_poles.get(sc.node_id)
            contains_target = bool(sub & targets)
            on_st_path = sc_p is not None and sc_p[0] in bicomp_st and sc_p[1] in bicomp_st
            if contains_target or on_st_path:
                unpacked_nodes.append(sc)
                child_s = cur_s & sub
                child_t = cur_t & sub
                if sc_p is not None:
                    if sc_p[0] in bicomp_st:
                        child_s = child_s | {sc_p[0]}
                    if sc_p[1] in bicomp_st:
                        child_t = child_t | {sc_p[1]}
                _unpack_node(sc, child_s, child_t)
            else:
                kept_sub_children.append(sc)

    _unpack_node(child, s_rooms, t_rooms)
    return core_rooms, kept_sub_children, unpacked_nodes


def _child_attachment_rooms(
    sub_child: SolveTreeNode,
    candidate_rooms: Collection[int],
    graph: LayoutGraph,
) -> set[int]:
    """Return rooms in candidate_rooms to which sub_child attaches."""
    cand_set = set(candidate_rooms)
    ports = get_child_boundary_ports(sub_child) & cand_set
    if ports:
        return ports
    sub_rooms = _collect_subtree_rooms(sub_child)
    out: set[int] = set()
    for edge in graph.edges:
        if edge.cut:
            continue
        if edge.src in cand_set and edge.dst in sub_rooms:
            out.add(edge.src)
        elif edge.dst in cand_set and edge.src in sub_rooms:
            out.add(edge.dst)
    return out


def _edmonds_karp_min_cut(
    nodes: Collection[Any],
    capacities: Mapping[tuple[Any, Any], int],
    source: Any,
    sink: Any,
) -> tuple[int, set[Any]]:
    """Run Edmonds-Karp max-flow / min-cut and return (max_flow, source_side_reachable)."""
    adj: dict[Any, list[Any]] = {u: [] for u in nodes}
    cap: dict[tuple[Any, Any], int] = collections.defaultdict(int)
    for (u, v), c in capacities.items():
        if c <= 0:
            continue
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
        cap[(u, v)] += c

    flow_total = 0
    while True:
        parent: dict[Any, Any] = {source: None}
        q: collections.deque[Any] = collections.deque([source])
        while q and sink not in parent:
            curr = q.popleft()
            for nxt in adj[curr]:
                if nxt not in parent and cap[(curr, nxt)] > 0:
                    parent[nxt] = curr
                    q.append(nxt)
                    if nxt == sink:
                        break
        if sink not in parent:
            break
        aug = 10**12
        v = sink
        while v != source:
            u = parent[v]
            aug = min(aug, cap[(u, v)])
            v = u
        flow_total += aug
        v = sink
        while v != source:
            u = parent[v]
            cap[(u, v)] -= aug
            cap[(v, u)] += aug
            v = u

    reachable: set[Any] = {source}
    q_reach: collections.deque[Any] = collections.deque([source])
    while q_reach:
        curr = q_reach.popleft()
        for nxt in adj[curr]:
            if nxt not in reachable and cap[(curr, nxt)] > 0:
                reachable.add(nxt)
                q_reach.append(nxt)
    return flow_total, reachable


def _build_piece_node(
    node_id: str,
    orig_child: SolveTreeNode,
    rooms: frozenset[int],
    sub_children: Sequence[SolveTreeNode],
    parent_attach_rooms: set[int],
    sibling_rooms: set[int],
    severed_edges: set[tuple[int, int]],
    graph: LayoutGraph,
    orig_root_id: str,
) -> SolveTreeNode:
    """Construct a piece SolveTreeNode with exact cut contract, boundary ports, and severed metadata."""
    subtree_rooms = set(rooms)
    for sc in sub_children:
        subtree_rooms |= _collect_subtree_rooms(sc)

    severed_pair_set = {frozenset(e) for e in severed_edges}

    # Un-severed edges between this piece and parent_attach_rooms (node rooms / external ports)
    parent_edges: list[tuple[int, int, int, int]] = []  # (parent_room, my_room, dir_p_to_my, weight)
    seen_p_edges: set[frozenset[int]] = set()
    severed_parent_ports: set[int] = set()
    for edge in graph.edges:
        if edge.cut:
            continue
        pair = frozenset({edge.src, edge.dst})
        disp = _safe_displacement(edge.direction)
        if disp is None:
            continue
        if edge.src in parent_attach_rooms and edge.dst in subtree_rooms:
            if pair in severed_pair_set:
                severed_parent_ports.add(edge.src)
                continue
            if pair not in seen_p_edges:
                seen_p_edges.add(pair)
                parent_edges.append((edge.src, edge.dst, int(edge.direction), max(1, edge.weight)))
        elif edge.dst in parent_attach_rooms and edge.src in subtree_rooms:
            if pair in severed_pair_set:
                severed_parent_ports.add(edge.dst)
                continue
            if pair not in seen_p_edges:
                seen_p_edges.add(pair)
                rev_dir = int(Direction(edge.direction).invert())
                parent_edges.append((edge.dst, edge.src, rev_dir, max(1, edge.weight)))

    unique_parent_ports = {pe[0] for pe in parent_edges}
    meta: dict[str, Any] = {
        "reactive_split_origin": orig_root_id,
    }
    if "forced_axis_alignments" in orig_child.metadata:
        meta["forced_axis_alignments"] = orig_child.metadata["forced_axis_alignments"]

    node_type = TreeNodeType.RIGID_CLUSTER

    if len(unique_parent_ports) == 2:
        p_sorted = tuple(sorted(unique_parent_ports))
        poles = (p_sorted[0], p_sorted[1])
        meta["cut_type"] = CutType.SPLIT_PAIR
        meta["poles"] = poles
        meta["cut_contract"] = SplitPairCut(poles=poles)
    elif len(unique_parent_ports) == 1:
        p_only = next(iter(unique_parent_ports))
        if len(parent_edges) == 1:
            p_room, my_room, dir_p_to_my, weight = parent_edges[0]
            meta["cut_type"] = CutType.BRIDGE
            meta["anchor_vnum"] = p_room
            meta["bridge_edge"] = (p_room, my_room)
            meta["direction"] = dir_p_to_my
            meta["weight"] = weight
            meta["cut_contract"] = BridgeCut(
                bridge_edge=(p_room, my_room),
                direction=dir_p_to_my,
                weight=weight,
                anchor_vnum=p_room,
            )
        else:
            meta["cut_type"] = CutType.ARTICULATION_POINT
            meta["anchor_vnum"] = p_only
            meta["cut_contract"] = ArticulationPointCut(anchor_vnum=p_only)
    else:
        # Interior middle piece or multi-port piece: anchor on a severed inter-piece edge to sibling_rooms
        sibling_bridge: tuple[int, int, int, int] | None = None
        for edge in graph.edges:
            if edge.cut:
                continue
            pair = frozenset({edge.src, edge.dst})
            if pair not in severed_pair_set:
                continue
            if _safe_displacement(edge.direction) is None:
                continue
            if edge.src in sibling_rooms and edge.dst in subtree_rooms:
                sibling_bridge = (edge.src, edge.dst, int(edge.direction), max(1, edge.weight))
                break
            if edge.dst in sibling_rooms and edge.src in subtree_rooms:
                rev_dir = int(Direction(edge.direction).invert())
                sibling_bridge = (edge.dst, edge.src, rev_dir, max(1, edge.weight))
                break
        if sibling_bridge is not None:
            out_r, my_r, dir_o_to_m, weight = sibling_bridge
            meta["cut_type"] = CutType.BRIDGE
            meta["anchor_vnum"] = out_r
            meta["bridge_edge"] = (out_r, my_r)
            meta["direction"] = dir_o_to_m
            meta["weight"] = weight
            meta["cut_contract"] = BridgeCut(
                bridge_edge=(out_r, my_r),
                direction=dir_o_to_m,
                weight=weight,
                anchor_vnum=out_r,
            )
        elif unique_parent_ports:
            p_first = min(unique_parent_ports)
            meta["cut_type"] = CutType.ARTICULATION_POINT
            meta["anchor_vnum"] = p_first
            meta["cut_contract"] = ArticulationPointCut(anchor_vnum=p_first)

    # Collect boundary ports:
    # 1. Severed parent ports (when severed from a pole of an enclosing container)
    # 2. Sub-child rooms inside this piece participating in severed_edges
    b_ports: set[int] = set(severed_parent_ports)
    for sc in sub_children:
        if "boundary_ports" in sc.metadata:
            b_ports.update(sc.metadata["boundary_ports"])
    for u, v in severed_edges:
        if u in subtree_rooms and u not in rooms:
            b_ports.add(u)
        if v in subtree_rooms and v not in rooms:
            b_ports.add(v)
    if "anchor_vnum" in meta:
        b_ports.discard(int(meta["anchor_vnum"]))
    if "poles" in meta:
        b_ports -= set(meta["poles"])
    b_ports -= set(rooms)
    if b_ports:
        meta["boundary_ports"] = tuple(sorted(b_ports))

    all_severed = set(severed_edges)
    for e_raw in orig_child.metadata.get("severed_bridge_edges", ()):
        all_severed.add((min(int(e_raw[0]), int(e_raw[1])), max(int(e_raw[0]), int(e_raw[1]))))
    for sc in sub_children:
        for e_raw in sc.metadata.get("severed_bridge_edges", ()):
            all_severed.add((min(int(e_raw[0]), int(e_raw[1])), max(int(e_raw[0]), int(e_raw[1]))))
    if all_severed:
        meta["severed_bridge_edges"] = tuple(sorted(all_severed))

    return SolveTreeNode(
        node_id=node_id,
        node_type=node_type,
        rooms=rooms,
        children=tuple(sub_children),
        metadata=meta,
    )


def _find_directional_cut(
    child: SolveTreeNode,
    s_rooms: set[int],
    t_rooms: set[int],
    direction: tuple[int, int, int],
    parent_attach_rooms: set[int],
    graph: LayoutGraph,
) -> tuple[
    set[int],
    set[int],
    list[SolveTreeNode],
    list[SolveTreeNode],
    set[int],
    set[int],
    list[SolveTreeNode],
    set[tuple[int, int]],
] | None:
    """Compute a valid directional cut-set separating s_rooms from t_rooms along direction."""
    if direction not in _CARDINAL_DIRECTIONS:
        return None
    if not s_rooms or not t_rooms or (s_rooms & t_rooms):
        return None

    core_rooms, kept_sub_children, unpacked_nodes = _unpack_child_shells(
        child, s_rooms, t_rooms, graph
    )
    if len(core_rooms) <= 1:
        return None

    # External ports adjacent to core_rooms (excluding parent rooms? No, parent_attach_rooms touching core_rooms)
    orig_poles = set(get_child_boundary_ports(child))
    ext_ports: set[int] = set()
    for p in (orig_poles | s_rooms | t_rooms | parent_attach_rooms) - core_rooms:
        ext_ports.add(p)

    # Filter ext_ports to only rooms that actually have an uncut edge to core_rooms
    connected_ext_ports: set[int] = set()
    for edge in graph.edges:
        if edge.cut:
            continue
        if edge.src in core_rooms and edge.dst in ext_ports:
            connected_ext_ports.add(edge.dst)
        elif edge.dst in core_rooms and edge.src in ext_ports:
            connected_ext_ports.add(edge.src)
    ext_ports = connected_ext_ports
    all_graph_nodes = core_rooms | ext_ports

    # Group kept sub-children into atomic sc_groups keyed by attachment rooms in all_graph_nodes
    sc_attach = {
        sc.node_id: _child_attachment_rooms(sc, all_graph_nodes, graph)
        for sc in kept_sub_children
    }
    sc_rooms_map = {sc.node_id: _collect_subtree_rooms(sc) for sc in kept_sub_children}
    sc_adj: dict[str, set[str]] = {sc.node_id: set() for sc in kept_sub_children}
    room_to_sc: dict[int, str] = {}
    for sc in kept_sub_children:
        for r in sc_rooms_map[sc.node_id]:
            room_to_sc[r] = sc.node_id
    for edge in graph.edges:
        if edge.cut:
            continue
        c1 = room_to_sc.get(edge.src)
        c2 = room_to_sc.get(edge.dst)
        if c1 is not None and c2 is not None and c1 != c2:
            sc_adj[c1].add(c2)
            sc_adj[c2].add(c1)

    sc_groups: list[tuple[list[SolveTreeNode], set[int]]] = []
    sc_by_id = {sc.node_id: sc for sc in kept_sub_children}
    seen_sc: set[str] = set()
    for sc in kept_sub_children:
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

    # Build directed edge metadata on all_graph_nodes (excluding parent-to-parent edges)
    adj: dict[int, list[int]] = collections.defaultdict(list)
    edge_disp: dict[tuple[int, int], tuple[int, int, int]] = {}
    seen_undir: set[tuple[int, int]] = set()
    parent_port_pairs = {frozenset({p1, p2}) for p1 in ext_ports for p2 in ext_ports if p1 != p2}

    for edge in graph.edges:
        if edge.cut or edge.src == edge.dst:
            continue
        if edge.src not in all_graph_nodes or edge.dst not in all_graph_nodes:
            continue
        if frozenset({edge.src, edge.dst}) in parent_port_pairs:
            continue
        ed = _safe_displacement(edge.direction)
        if ed is None:
            continue
        undir = (min(edge.src, edge.dst), max(edge.src, edge.dst))
        if undir in seen_undir:
            continue
        seen_undir.add(undir)
        adj[edge.src].append(edge.dst)
        adj[edge.dst].append(edge.src)
        edge_disp[(edge.src, edge.dst)] = ed
        edge_disp[(edge.dst, edge.src)] = (-ed[0], -ed[1], -ed[2])

    s_seeds = {r for r in s_rooms if r in all_graph_nodes}
    t_seeds = {r for r in t_rooms if r in all_graph_nodes}
    if not s_seeds or not t_seeds:
        return None

    # Pass 1: Balanced 1-edge cut strictly inside core_rooms
    best_cut: tuple[tuple[int, int, int, int], set[int], set[int]] | None = None
    for u, v in sorted(seen_undir):
        if u not in core_rooms or v not in core_rooms:
            continue
        d_uv = edge_disp[(u, v)]
        if d_uv == direction:
            cut_u, cut_v = u, v
        elif edge_disp[(v, u)] == direction:
            cut_u, cut_v = v, u
        else:
            continue

        # Flood fill from s_seeds without crossing (cut_u, cut_v)
        side_s_all = set(s_seeds)
        q_bfs: collections.deque[int] = collections.deque(sorted(s_seeds))
        while q_bfs:
            curr = q_bfs.popleft()
            for nxt in adj[curr]:
                if (curr == cut_u and nxt == cut_v) or (curr == cut_v and nxt == cut_u):
                    continue
                if nxt not in side_s_all:
                    side_s_all.add(nxt)
                    q_bfs.append(nxt)

        if side_s_all & t_seeds:
            continue
        if cut_u not in side_s_all or cut_v in side_s_all:
            continue

        side_t_all = all_graph_nodes - side_s_all
        side_s_core = side_s_all & core_rooms
        side_t_core = side_t_all & core_rooms
        if not side_s_core or not side_t_core:
            continue

        valid_groups = True
        s_grp_rooms = 0
        t_grp_rooms = 0
        for grp_nodes, pts in sc_groups:
            if (pts & side_s_all) and (pts & side_t_all):
                valid_groups = False
                break
            g_size = sum(len(sc_rooms_map[g.node_id]) for g in grp_nodes)
            if pts & side_t_all:
                t_grp_rooms += g_size
            else:
                s_grp_rooms += g_size
        if not valid_groups:
            continue

        sz_s = len(side_s_core) + s_grp_rooms
        sz_t = len(side_t_core) + t_grp_rooms
        score = (max(sz_s, sz_t), abs(sz_s - sz_t), cut_u, cut_v)
        if best_cut is None or score < best_cut[0]:
            best_cut = (score, side_s_all, side_t_all)

    chosen_side_s_all: set[int] | None = None
    chosen_side_t_all: set[int] | None = None
    if best_cut is not None:
        chosen_side_s_all = best_cut[1]
        chosen_side_t_all = best_cut[2]
    else:
        # Pass 2: Edmonds-Karp directed min-cut
        inf_cap = 10**9
        ext_cut_cap = len(core_rooms) + 1
        src_node = "__EK_SRC__"
        snk_node = "__EK_SNK__"
        ek_nodes: list[Any] = [src_node, snk_node, *sorted(all_graph_nodes)]

        # Precompute degree of each ext_port into core_rooms
        ext_core_deg = {
            p: sum(1 for nb in adj[p] if nb in core_rooms)
            for p in ext_ports
        }

        def _run_pass2(locked_ext_ports: set[int]) -> tuple[set[int], set[int]] | None:
            caps: dict[tuple[Any, Any], int] = collections.defaultdict(int)
            for s in s_seeds:
                caps[(src_node, s)] = inf_cap
            for t in t_seeds:
                caps[(t, snk_node)] = inf_cap

            # Strongly couple sc_groups via bidirectional infinite arcs
            for _grp_nodes, pts in sc_groups:
                pts_list = sorted(pts & all_graph_nodes)
                if len(pts_list) >= 2:
                    p0 = pts_list[0]
                    for pk in pts_list[1:]:
                        caps[(p0, pk)] = inf_cap
                        caps[(pk, p0)] = inf_cap

            for u, v in seen_undir:
                touches_locked_ext = (u in locked_ext_ports) or (v in locked_ext_ports)
                touches_ext = (u in ext_ports) or (v in ext_ports)
                if touches_locked_ext:
                    caps[(u, v)] = inf_cap
                    caps[(v, u)] = inf_cap
                    continue

                cut_cap = ext_cut_cap if touches_ext else 1
                d_uv = edge_disp[(u, v)]
                if d_uv == direction:
                    caps[(u, v)] = cut_cap
                    caps[(v, u)] = inf_cap
                elif edge_disp[(v, u)] == direction:
                    caps[(v, u)] = cut_cap
                    caps[(u, v)] = inf_cap
                else:
                    caps[(u, v)] = inf_cap
                    caps[(v, u)] = inf_cap

            flow_val, reachable = _edmonds_karp_min_cut(ek_nodes, caps, src_node, snk_node)
            if flow_val >= inf_cap or flow_val <= 0:
                return None
            side_s_res = {x for x in reachable if isinstance(x, int)}
            side_t_res = all_graph_nodes - side_s_res
            return side_s_res, side_t_res

        locked_ports = {p for p, deg in ext_core_deg.items() if deg <= 1}
        res_p2 = _run_pass2(locked_ports)
        if res_p2 is not None:
            s_res, t_res = res_p2
            if not (s_res & core_rooms) or not (t_res & core_rooms):
                # Lock any ext_port that isolated an empty core side and retry once
                extra_locked = set(locked_ports)
                if not (s_res & core_rooms):
                    extra_locked.update(s_res & ext_ports)
                if not (t_res & core_rooms):
                    extra_locked.update(t_res & ext_ports)
                res_p2 = _run_pass2(extra_locked)
            if res_p2 is not None:
                s_res, t_res = res_p2
                if (s_res & core_rooms) and (t_res & core_rooms):
                    chosen_side_s_all = s_res
                    chosen_side_t_all = t_res

    if chosen_side_s_all is None or chosen_side_t_all is None:
        return None

    side_s_core = chosen_side_s_all & core_rooms
    side_t_core = chosen_side_t_all & core_rooms
    if not side_s_core or not side_t_core:
        return None

    # Partition kept_sub_children across side_s and side_t without bisecting any sc_group
    children_s: list[SolveTreeNode] = []
    children_t: list[SolveTreeNode] = []
    for grp_nodes, pts in sc_groups:
        in_s = bool(pts & chosen_side_s_all)
        in_t = bool(pts & chosen_side_t_all)
        if in_s and in_t:
            return None
        if in_t:
            children_t.extend(grp_nodes)
        else:
            children_s.extend(grp_nodes)

    subtree_s = set(side_s_core)
    for sc in children_s:
        subtree_s |= _collect_subtree_rooms(sc)
    subtree_t = set(side_t_core)
    for sc in children_t:
        subtree_t |= _collect_subtree_rooms(sc)

    # Identify all edges crossing side_s_all -> side_t_all
    severed_edges: set[tuple[int, int]] = set()
    for u, v in seen_undir:
        if u in chosen_side_s_all and v in chosen_side_t_all:
            if edge_disp[(u, v)] != direction:
                return None
            severed_edges.add((min(u, v), max(u, v)))
        elif v in chosen_side_s_all and u in chosen_side_t_all:
            if edge_disp[(v, u)] != direction:
                return None
            severed_edges.add((min(u, v), max(u, v)))

    if not severed_edges:
        return None

    return (
        side_s_core,
        side_t_core,
        children_s,
        children_t,
        subtree_s,
        subtree_t,
        unpacked_nodes,
        severed_edges,
    )


def split_child_along_direction(
    child: SolveTreeNode,
    s_rooms: set[int],
    t_rooms: set[int],
    direction: tuple[int, int, int],
    parent_attach_rooms: set[int],
    graph: LayoutGraph,
    split_tag: str,
) -> tuple[SolveTreeNode, SolveTreeNode, list[SolveTreeNode]] | None:
    """Split child into (piece_S, piece_T) along a directional edge cut-set separating S from T."""
    cut_res = _find_directional_cut(
        child=child,
        s_rooms=s_rooms,
        t_rooms=t_rooms,
        direction=direction,
        parent_attach_rooms=parent_attach_rooms,
        graph=graph,
    )
    if cut_res is None:
        return None
    (
        side_s_core,
        side_t_core,
        children_s,
        children_t,
        subtree_s,
        subtree_t,
        unpacked_nodes,
        severed_edges,
    ) = cut_res

    orig_root_id = str(child.metadata.get("reactive_split_origin") or child.node_id)
    piece_s = _build_piece_node(
        node_id=f"{child.node_id}__{split_tag}_S",
        orig_child=child,
        rooms=frozenset(side_s_core),
        sub_children=children_s,
        parent_attach_rooms=parent_attach_rooms,
        sibling_rooms=subtree_t,
        severed_edges=severed_edges,
        graph=graph,
        orig_root_id=orig_root_id,
    )
    piece_t = _build_piece_node(
        node_id=f"{child.node_id}__{split_tag}_T",
        orig_child=child,
        rooms=frozenset(side_t_core),
        sub_children=children_t,
        parent_attach_rooms=parent_attach_rooms,
        sibling_rooms=subtree_s,
        severed_edges=severed_edges,
        graph=graph,
        orig_root_id=orig_root_id,
    )
    return piece_s, piece_t, unpacked_nodes


__all__ = [
    "compute_split_lp_signature",
    "compute_total_edge_length",
    "detect_planar_transverse_shear",
    "diagnose_transverse_shear",
    "diagnose_unresolvable_collision",
    "generate_coordinate_threshold_partitions",
    "get_child_interface_rooms",
    "is_split_pair_origin",
    "is_splittable_child",
    "solve_3d_deformation_diagnostic",
    "split_child_along_direction",
]
