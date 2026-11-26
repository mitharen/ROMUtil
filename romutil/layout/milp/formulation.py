"""Stage 2: Basic MILP formulation for layout cluster optimization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pyomo.environ as pyo

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.milp.clearance import ClearanceConstraintBuilder
from romutil.layout.milp.clearance_boxes import extract_entity_coord
from romutil.layout.milp.inventory import (
    ActiveEdge,
    ClusterInventory,
    EndpointRef,
    get_child_coordinate,
    get_child_strictly_interior_rooms,
)
from romutil.types import LayoutOptions

if TYPE_CHECKING:
    from romutil.layout.milp.solver import SolvedClusterExtraction


def calculate_dynamic_bounds(inventory: ClusterInventory) -> int:
    """Calculate tight dynamic coordinate and Big-M separation bounds for cluster optimization.

    Bounds the coordinate domain and Big-M relaxation values proportional to the cluster's
    topological and geometric scale:
    - Child component bounding spans derived from child solutions and keep-out sub-boxes.
    - Active edge minimum length requirements with stretch headroom.
    - Parent room count scale to ensure collision-free room placement.
    - A tight baseline floor (50) ensuring mathematical soundness for small leaf clusters
      without numerical instability from excessive relaxation constants.

    Args:
        inventory: ClusterInventory containing classified rooms, children, child solutions,
            keep-out sub-boxes, and active edges.

    Returns:
        Integer coordinate bounding value (m_coord).
    """
    baseline_floor = 50

    # 1. Child component bounding spans and local coordinate offsets
    child_spans = 0
    all_child_ids: set[str] = set()
    if inventory.children:
        all_child_ids.update(c.node_id for c in inventory.children)
    if inventory.child_solutions:
        all_child_ids.update(inventory.child_solutions.keys())
    if inventory.child_sub_boxes:
        all_child_ids.update(inventory.child_sub_boxes.keys())

    for cid in all_child_ids:
        boxes: list[BoundingBox3D] = []

        # Collect boxes from child solutions
        if inventory.child_solutions and cid in inventory.child_solutions:
            c_sol = inventory.child_solutions[cid]
            if c_sol.bounding_box is not None:
                boxes.append(c_sol.bounding_box)
            if c_sol.sub_boxes:
                boxes.extend(c_sol.sub_boxes)

        # Collect boxes from derived keep-out sub-boxes
        if inventory.child_sub_boxes and cid in inventory.child_sub_boxes:
            for raw_sb in inventory.child_sub_boxes[cid]:
                if isinstance(raw_sb, BoundingBox3D):
                    boxes.append(raw_sb)
                else:
                    boxes.append(BoundingBox3D.from_coordinates([raw_sb[0], raw_sb[1]]))

        if boxes:
            c_min_x = min(b.min_x for b in boxes)
            c_max_x = max(b.max_x for b in boxes)
            c_min_y = min(b.min_y for b in boxes)
            c_max_y = max(b.max_y for b in boxes)
            c_min_z = min(b.min_z for b in boxes)
            c_max_z = max(b.max_z for b in boxes)

            # Sum of child dimensions (width + height + depth)
            width = c_max_x - c_min_x + 1
            height = c_max_y - c_min_y + 1
            depth = c_max_z - c_min_z + 1
            span = width + height + depth

            # Local origin coordinate offset
            max_offset = max(
                abs(c_min_x),
                abs(c_max_x),
                abs(c_min_y),
                abs(c_max_y),
                abs(c_min_z),
                abs(c_max_z),
            )
            child_spans += max(span, 2 * max_offset)
        else:
            # Default span estimate for unresolved child components
            child_spans += 20

    # 2. Active edge requirements with stretch headroom
    edge_min_sum = 0
    max_edge_min = 0
    if inventory.active_edges:
        edge_min_sum = sum(ae.min_length for ae in inventory.active_edges)
        max_edge_min = max(ae.min_length for ae in inventory.active_edges)

    # Multiplier accounts for corridor stretching and detour clearances around obstacles
    edge_scale = max(2 * edge_min_sum, 2 * max_edge_min)

    # 3. Parent room scale proportional to room count and grid resolution
    coord_unit = max(1, int(getattr(inventory, "coord_unit", 1)))
    room_scale = max(len(inventory.rooms), len(inventory.boundary_ports)) * 10 * coord_unit

    total_scale = child_spans + edge_scale + room_scale
    return max(baseline_floor, total_scale)


class ClusterFormulation:
    """MILP formulation model and geometric accessors for layout cluster optimization."""

    def __init__(
        self,
        inventory: ClusterInventory,
        options: LayoutOptions | None = None,
        m_coord: int | None = None,
        w_transverse: float | None = None,
    ) -> None:
        """Initialize the MILP formulation with decision variables and base constraints.

        Args:
            inventory: ClusterInventory containing classified rooms, children, alignments,
                and active edges.
            options: Optional layout configuration parameters.
            m_coord: Optional explicit coordinate bound overriding dynamic calculation.
            w_transverse: Optional explicit transverse slack penalty weight overriding quadratic scaling.
        """
        self.inventory = inventory
        self.options = options
        self.m_coord = m_coord if m_coord is not None else calculate_dynamic_bounds(inventory)
        self.m_big = 2 * self.m_coord
        self.m_scale = max(
            50,
            sum(e.min_length for e in inventory.active_edges),
        )
        self.w_transverse = (
            w_transverse
            if w_transverse is not None
            else 10.0 * (self.m_scale**2)
        )
        self.model = pyo.ConcreteModel()
        self._edge_transverse_slack: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._build_model()
        self.clearance_builder = ClearanceConstraintBuilder(self)

    def _build_model(self) -> None:
        """Declare variables, fix anchors, and attach baseline constraints and objective."""
        inventory = self.inventory
        model = self.model
        m_coord = self.m_coord

        # Local room and boundary port coordinates
        child_strictly_interior: set[int] = set()
        for child in inventory.children:
            child_strictly_interior.update(
                get_child_strictly_interior_rooms(child, inventory.child_solutions.get(child.node_id))
            )

        external_ports = {p for p in inventory.boundary_ports if p not in child_strictly_interior}
        local_rooms = tuple(sorted(set(inventory.rooms) | external_ports))
        self.local_rooms = local_rooms
        if local_rooms:
            model.x = pyo.Var(local_rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord), initialize=0)
            model.y = pyo.Var(local_rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord), initialize=0)
            model.z = pyo.Var(local_rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord), initialize=0)

            anchor: int | None = None
            if inventory.rooms:
                anchor = min(inventory.rooms)
            elif external_ports:
                anchor = min(external_ports)

            if anchor is not None:
                model.x[anchor].fix(0)
                model.y[anchor].fix(0)
                model.z[anchor].fix(0)

        # Child macro-box origins
        child_ids = [c.node_id for c in inventory.children]
        model.cx = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord), initialize=0)
        model.cy = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord), initialize=0)
        model.cz = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord), initialize=0)

        if not local_rooms and child_ids:
            first_c = child_ids[0]
            model.cx[first_c].fix(0)
            model.cy[first_c].fix(0)
            model.cz[first_c].fix(0)

        # Gateway child-child alignment equalities
        model.align_x = pyo.ConstraintList()
        model.align_y = pyo.ConstraintList()
        model.align_z = pyo.ConstraintList()

        for c_align in inventory.child_alignments:
            model.align_x.add(
                model.cx[c_align.base_child_id] + c_align.base_coord.x
                == model.cx[c_align.target_child_id] + c_align.target_coord.x
            )
            model.align_y.add(
                model.cy[c_align.base_child_id] + c_align.base_coord.y
                == model.cy[c_align.target_child_id] + c_align.target_coord.y
            )
            model.align_z.add(
                model.cz[c_align.base_child_id] + c_align.base_coord.z
                == model.cz[c_align.target_child_id] + c_align.target_coord.z
            )

        # Active edge displacement and transverse slack constraints
        edge_indices = list(range(len(inventory.active_edges)))
        model.s_transverse_p = pyo.VarList(domain=pyo.NonNegativeReals)
        model.s_transverse_m = pyo.VarList(domain=pyo.NonNegativeReals)

        if edge_indices:
            model.L = pyo.Var(
                edge_indices,
                domain=pyo.Integers,
                bounds=lambda m, i: (inventory.active_edges[i].min_length, m_coord),
            )
            model.edge_x = pyo.ConstraintList()
            model.edge_y = pyo.ConstraintList()
            model.edge_z = pyo.ConstraintList()

            for i, ae in enumerate(inventory.active_edges):
                src_x, src_y, src_z = self.get_point(ae.src)
                dst_x, dst_y, dst_z = self.get_point(ae.dst)
                dx, dy, dz = ae.direction

                if dx != 0:
                    model.edge_x.add(dst_x - src_x == model.L[i] * dx)
                elif ae.allow_transverse_slack and (dy != 0 or dz != 0):
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_x.add(dst_x - src_x == sp - sm)
                    self._edge_transverse_slack[(ae.edge_id, "x")] = (sp, sm)
                else:
                    model.edge_x.add(dst_x - src_x == 0)

                if dy != 0:
                    model.edge_y.add(dst_y - src_y == model.L[i] * dy)
                elif ae.allow_transverse_slack and (dx != 0 or dz != 0):
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_y.add(dst_y - src_y == sp - sm)
                    self._edge_transverse_slack[(ae.edge_id, "y")] = (sp, sm)
                else:
                    model.edge_y.add(dst_y - src_y == 0)

                if dz != 0:
                    model.edge_z.add(dst_z - src_z == model.L[i] * dz)
                else:
                    model.edge_z.add(dst_z - src_z == 0)

        self._build_objective()

    def _build_objective(self) -> None:
        """Formulate multi-objective function with convex piecewise stretch and transverse penalties."""
        inventory = self.inventory
        model = self.model
        edge_indices = list(range(len(inventory.active_edges)))

        # 1. Piecewise-linear convex stretch penalties
        stretch_expr = 0
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
        else:
            model.s_stretch_1 = pyo.Var([], domain=pyo.NonNegativeReals)
            model.s_stretch_2 = pyo.Var([], domain=pyo.NonNegativeReals)
            model.s_stretch_3 = pyo.Var([], domain=pyo.NonNegativeReals)

        # 2. Transverse shear penalty
        transverse_penalty = (
            sum(
                self.w_transverse * (model.s_transverse_p[k] + model.s_transverse_m[k])
                for k in model.s_transverse_p
            )
            if hasattr(model, "s_transverse_p") and len(model.s_transverse_p) > 0
            else 0
        )

        self._stretch_expr: Any = stretch_expr
        model.obj = pyo.Objective(
            expr=stretch_expr + transverse_penalty,
            sense=pyo.minimize,
        )

    def get_point(self, ref: EndpointRef) -> tuple[Any, Any, Any]:
        """Return 3D coordinate expression (x, y, z) for an endpoint reference."""
        if isinstance(ref, int):
            return (self.model.x[ref], self.model.y[ref], self.model.z[ref])
        cid, v = ref
        c_sol = self.inventory.child_solutions[cid]
        loc = get_child_coordinate(c_sol, v)
        if loc is None:
            loc = c_sol.local_coordinates[v]
        return (
            self.model.cx[cid] + loc.x,
            self.model.cy[cid] + loc.y,
            self.model.cz[cid] + loc.z,
        )

    def get_room_box(self, u: int | Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box (min_x, max_x, min_y, max_y, min_z, max_z) for a parent room."""
        vnum = u.vnum if hasattr(u, "vnum") else u
        px, py, pz = self.get_point(vnum)
        lo, hi = (
            getattr(self.inventory, "room_extents", {}).get(vnum, (0, 0))
            if isinstance(vnum, int)
            else (0, 0)
        )
        return (px - lo, px + hi, py - lo, py + hi, pz, pz)

    def get_child_sub_box(self, key: tuple[str, int] | Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a child sub-box indexed by (child_id, index)."""
        if isinstance(key, tuple):
            cid, k = key
        else:
            cid, k = key.child_id, key.box_index
        raw_sb = self.inventory.child_sub_boxes[cid][k]
        sb = (
            raw_sb
            if isinstance(raw_sb, BoundingBox3D)
            else BoundingBox3D.from_coordinates([raw_sb[0], raw_sb[1]])
        )
        return (
            self.model.cx[cid] + sb.min_x,
            self.model.cx[cid] + sb.max_x,
            self.model.cy[cid] + sb.min_y,
            self.model.cy[cid] + sb.max_y,
            self.model.cz[cid] + sb.min_z,
            self.model.cz[cid] + sb.max_z,
        )

    def _get_corridor_box(
        self,
        ae: ActiveEdge,
        *,
        interior: bool,
        include_transverse_slack: bool = False,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a corridor segment, optionally including transverse slack."""
        x1, y1, z1 = self.get_point(ae.src)
        x2, y2, z2 = self.get_point(ae.dst)
        dx, dy, dz = ae.direction
        src_lo, src_hi = getattr(ae, "src_extent", (0, 0))
        dst_lo, dst_hi = getattr(ae, "dst_extent", (0, 0))
        t_lo, t_hi = getattr(ae, "transverse_extent", (0, 0))
        use_slack = include_transverse_slack and ae.allow_transverse_slack
        slack_x = self._edge_transverse_slack.get((ae.edge_id, "x")) if use_slack else None
        slack_y = self._edge_transverse_slack.get((ae.edge_id, "y")) if use_slack else None

        if dx != 0:
            if interior:
                x_min, x_max = (
                    (x1 + src_hi + 1, x2 - dst_lo - 1)
                    if dx > 0
                    else (x2 + dst_hi + 1, x1 - src_lo - 1)
                )
            else:
                x_min, x_max = (x1, x2) if dx > 0 else (x2, x1)
            y_min = y1 - t_lo - slack_y[1] if slack_y is not None else y1 - t_lo
            y_max = y1 + t_hi + slack_y[0] if slack_y is not None else y1 + t_hi
            return (x_min, x_max, y_min, y_max, z1, z1)
        if dy != 0:
            if interior:
                y_min, y_max = (
                    (y1 + src_hi + 1, y2 - dst_lo - 1)
                    if dy > 0
                    else (y2 + dst_hi + 1, y1 - src_lo - 1)
                )
            else:
                y_min, y_max = (y1, y2) if dy > 0 else (y2, y1)
            x_min = x1 - t_lo - slack_x[1] if slack_x is not None else x1 - t_lo
            x_max = x1 + t_hi + slack_x[0] if slack_x is not None else x1 + t_hi
            return (x_min, x_max, y_min, y_max, z1, z1)
        margin = 1 if interior else 0
        z_min, z_max = (z1 + margin, z2 - margin) if dz > 0 else (z2 + margin, z1 - margin)
        x_min = x1 - t_lo - slack_x[1] if slack_x is not None else x1 - t_lo
        x_max = x1 + t_hi + slack_x[0] if slack_x is not None else x1 + t_hi
        y_min = y1 - t_lo - slack_y[1] if slack_y is not None else y1 - t_lo
        y_max = y1 + t_hi + slack_y[0] if slack_y is not None else y1 + t_hi
        return (x_min, x_max, y_min, y_max, z_min, z_max)

    def get_corridor_interior_box(self, ae: ActiveEdge) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a corridor's interior (excluding endpoints)."""
        return self._get_corridor_box(ae, interior=True, include_transverse_slack=False)

    def get_corridor_full_box(self, ae: ActiveEdge) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for an entire corridor segment (including endpoints)."""
        return self._get_corridor_box(ae, interior=False, include_transverse_slack=False)

    def get_corridor_envelope_box(
        self,
        ae: ActiveEdge,
        *,
        interior: bool = True,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D corridor bounding box enclosing any permitted transverse slack."""
        return self._get_corridor_box(ae, interior=interior, include_transverse_slack=True)

    def has_transverse_slack(self) -> bool:
        """Return True if the formulation declares any transverse shear slack variables."""
        return (
            hasattr(self.model, "s_transverse_p") and len(self.model.s_transverse_p) > 0
        ) or (
            hasattr(self.model, "s_transverse_m") and len(self.model.s_transverse_m) > 0
        )

    def lock_transverse_slack(self) -> None:
        """Fix all transverse shear slack variables to zero."""
        for attr in ("s_transverse_p", "s_transverse_m"):
            if hasattr(self.model, attr):
                for var_data in getattr(self.model, attr).values():
                    var_data.fix(0)

    def unlock_transverse_slack(self) -> None:
        """Unfix all transverse shear slack variables so shear can be used if required."""
        for attr in ("s_transverse_p", "s_transverse_m"):
            if hasattr(self.model, attr):
                for var_data in getattr(self.model, attr).values():
                    var_data.unfix()

    def unlock_vertical_transverse_slack(self) -> bool:
        """Unfix transverse shear slack variables only for vertical (dz != 0) active edges."""
        vertical_ids = {
            ae.edge_id
            for ae in self.inventory.active_edges
            if ae.direction[2] != 0 and ae.allow_transverse_slack
        }
        if not vertical_ids:
            return False
        unfixed_any = False
        for (edge_id, _axis), (sp, sm) in self._edge_transverse_slack.items():
            if edge_id in vertical_ids and (sp.fixed or sm.fixed):
                sp.unfix()
                sm.unfix()
                unfixed_any = True
        return unfixed_any

    def enforce_min_transverse_slack(
        self,
        min_total: int = 1,
        *,
        horizontal_only: bool = False,
    ) -> bool:
        """Enforce a minimum total transverse shear slack to prune infeasible zero-shear regions."""
        if horizontal_only:
            horiz_ids = {
                ae.edge_id
                for ae in self.inventory.active_edges
                if ae.direction[2] == 0 and ae.allow_transverse_slack
            }
            svars = [
                v
                for (eid, _axis), pair in self._edge_transverse_slack.items()
                if eid in horiz_ids
                for v in pair
            ]
            cname = "horizontal_transverse_slack_lb"
        else:
            svars = [v for pair in self._edge_transverse_slack.values() for v in pair]
            cname = "transverse_slack_lb"

        if not svars or hasattr(self.model, cname):
            return False
        setattr(self.model, cname, pyo.Constraint(expr=sum(svars) >= min_total))
        return True

    def clear_transverse_slack_lower_bounds(self) -> None:
        """Remove any temporary transverse slack lower-bound constraints from the model."""
        for cname in ("transverse_slack_lb", "horizontal_transverse_slack_lb"):
            if hasattr(self.model, cname):
                self.model.del_component(cname)

    def enforce_max_transverse_slack(self, max_total: int) -> bool:
        """Enforce a maximum total transverse shear slack bound across all slack variables."""
        svars = [v for pair in self._edge_transverse_slack.values() for v in pair]
        if not svars or max_total < 0:
            return False
        if hasattr(self.model, "transverse_slack_ub"):
            self.model.del_component("transverse_slack_ub")
        self.model.transverse_slack_ub = pyo.Constraint(expr=sum(svars) <= max_total)
        return True

    def clear_transverse_slack_upper_bound(self) -> None:
        """Remove any temporary transverse slack upper-bound constraint from the model."""
        if hasattr(self.model, "transverse_slack_ub"):
            self.model.del_component("transverse_slack_ub")

    def _compute_canonical_transverse_shear(
        self,
        extraction: SolvedClusterExtraction | None = None,
        *,
        unfixed_only: bool = False,
    ) -> dict[tuple[str, str], tuple[int, int]]:
        """Compute canonical integer (s_p, s_m) shear pairs for transverse slack edges."""
        edge_by_id = {ae.edge_id: ae for ae in self.inventory.active_edges}
        canonical: dict[tuple[str, str], tuple[int, int]] = {}
        for (edge_id, axis), (sp, sm) in self._edge_transverse_slack.items():
            if unfixed_only and sp.fixed and sm.fixed:
                continue
            delta: int | None = None
            if extraction is not None and edge_id in edge_by_id:
                ae = edge_by_id[edge_id]
                try:
                    src_coord = extract_entity_coord(ae.src, extraction, self.inventory)
                    dst_coord = extract_entity_coord(ae.dst, extraction, self.inventory)
                except KeyError:
                    delta = None
                else:
                    axis_idx = 0 if axis == "x" else 1
                    delta = int(dst_coord[axis_idx] - src_coord[axis_idx])
            if delta is None:
                p_cur = pyo.value(sp, exception=False) or 0.0
                m_cur = pyo.value(sm, exception=False) or 0.0
                delta = int(round(float(p_cur) - float(m_cur)))
            canonical[(edge_id, axis)] = (max(0, delta), max(0, -delta))
        return canonical

    def has_nonzero_transverse_shear(
        self,
        extraction: SolvedClusterExtraction | None = None,
    ) -> bool:
        """Return True if any unfixed transverse slack edge carries nonzero canonical shear."""
        if not self.has_transverse_slack():
            return False
        return any(
            p + m > 0
            for p, m in self._compute_canonical_transverse_shear(
                extraction, unfixed_only=True
            ).values()
        )

    @contextmanager
    def fixed_transverse_slack_stretch_polish(
        self,
        extraction: SolvedClusterExtraction | None = None,
    ) -> Iterator[bool]:
        """Temporarily fix unfixed transverse slack to canonical shear and minimize pure stretch."""
        if not self.has_transverse_slack():
            yield False
            return
        canonical = self._compute_canonical_transverse_shear(extraction, unfixed_only=True)
        if not canonical or sum(p + m for p, m in canonical.values()) == 0:
            yield False
            return

        saved_fixed: list[tuple[Any, bool, Any, bool]] = []
        deactivated_bounds: list[str] = []
        try:
            for key, (p_val, m_val) in canonical.items():
                sp, sm = self._edge_transverse_slack[key]
                saved_fixed.append((sp, bool(sp.fixed), sm, bool(sm.fixed)))
                sp.fix(p_val)
                sm.fix(m_val)

            for cname in (
                "transverse_slack_lb",
                "horizontal_transverse_slack_lb",
                "transverse_slack_ub",
            ):
                if hasattr(self.model, cname):
                    constr = getattr(self.model, cname)
                    if constr.active:
                        constr.deactivate()
                        deactivated_bounds.append(cname)

            self.model.obj.deactivate()
            self.model._stretch_polish_obj = pyo.Objective(
                expr=self._stretch_expr,
                sense=pyo.minimize,
            )
            yield True
        finally:
            if hasattr(self.model, "_stretch_polish_obj"):
                self.model.del_component("_stretch_polish_obj")
            if hasattr(self.model, "obj"):
                self.model.obj.activate()
            for cname in deactivated_bounds:
                if hasattr(self.model, cname):
                    getattr(self.model, cname).activate()
            for sp, sp_was_fixed, sm, sm_was_fixed in saved_fixed:
                if not sp_was_fixed:
                    sp.unfix()
                if not sm_was_fixed:
                    sm.unfix()

    def apply_clearances(self) -> None:
        """Enforce disjunctive Big-M geometric clearance and non-overlap constraints."""
        self.clearance_builder.build()


def formulate_cluster_model(
    inventory: ClusterInventory,
    options: LayoutOptions | None = None,
    m_coord: int | None = None,
    w_transverse: float | None = None,
) -> ClusterFormulation:
    """Build the baseline Pyomo model for layout cluster optimization.

    Factory function instantiating ClusterFormulation.

    Args:
        inventory: ClusterInventory containing classified rooms, children, alignments,
            and active edges.
        options: Optional layout configuration parameters.
        m_coord: Optional explicit coordinate bound overriding dynamic calculation.
        w_transverse: Optional explicit transverse slack penalty weight overriding quadratic scaling.

    Returns:
        ClusterFormulation wrapping the Pyomo model and coordinate accessors.
    """
    return ClusterFormulation(
        inventory=inventory,
        options=options,
        m_coord=m_coord,
        w_transverse=w_transverse,
    )


__all__ = [
    "ClusterFormulation",
    "calculate_dynamic_bounds",
    "formulate_cluster_model",
]
