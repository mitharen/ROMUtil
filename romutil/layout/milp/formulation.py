"""Stage 2: Basic MILP formulation for layout cluster optimization."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.milp.clearance import ClearanceConstraintBuilder
from romutil.layout.milp.inventory import (
    ActiveEdge,
    BoundaryExit,
    ClusterInventory,
    EndpointRef,
    get_child_coordinate,
    get_child_strictly_interior_rooms,
)
from romutil.types import LayoutOptions


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

    # 3. Parent room scale proportional to room count
    room_scale = max(len(inventory.rooms), len(inventory.boundary_ports)) * 10

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
                elif ae.allow_transverse_slack and dy != 0:
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_x.add(dst_x - src_x == sp - sm)
                else:
                    model.edge_x.add(dst_x - src_x == 0)

                if dy != 0:
                    model.edge_y.add(dst_y - src_y == model.L[i] * dy)
                elif ae.allow_transverse_slack and dx != 0:
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_y.add(dst_y - src_y == sp - sm)
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
            model.s_stretch_1 = pyo.Var(edge_indices, domain=pyo.NonNegativeReals, bounds=(0, 2))
            model.s_stretch_2 = pyo.Var(edge_indices, domain=pyo.NonNegativeReals, bounds=(0, 4))
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
        return (px, px, py, py, pz, pz)

    def get_child_sub_box(self, key: tuple[str, int] | Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a child sub-box indexed by (child_id, index)."""
        if hasattr(key, "child_id") and hasattr(key, "box_index"):
            cid, k = key.child_id, key.box_index
        else:
            cid, k = key
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

    def get_corridor_interior_box(self, ae: ActiveEdge) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a corridor's interior (excluding endpoints)."""
        x1, y1, z1 = self.get_point(ae.src)
        x2, y2, z2 = self.get_point(ae.dst)
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

    def get_corridor_full_box(self, ae: ActiveEdge) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for an entire corridor segment (including endpoints)."""
        x1, y1, z1 = self.get_point(ae.src)
        x2, y2, z2 = self.get_point(ae.dst)
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

    def get_boundary_exit_interior_box(self, be: BoundaryExit) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a boundary exit's interior segment (excluding the boundary port)."""
        xu, yu, zu = self.get_point(be.port)
        dx, dy, dz = be.direction
        length = be.length
        if dx > 0:
            return (xu + 1, xu + length, yu, yu, zu, zu)
        elif dx < 0:
            return (xu - length, xu - 1, yu, yu, zu, zu)
        elif dy > 0:
            return (xu, xu, yu + 1, yu + length, zu, zu)
        elif dy < 0:
            return (xu, xu, yu - length, yu - 1, zu, zu)
        elif dz > 0:
            return (xu, xu, yu, yu, zu + 1, zu + length)
        else:  # dz < 0
            return (xu, xu, yu, yu, zu - length, zu - 1)

    def get_boundary_exit_full_box(self, be: BoundaryExit) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for an entire boundary exit corridor segment (including the boundary port)."""
        xu, yu, zu = self.get_point(be.port)
        dx, dy, dz = be.direction
        length = be.length
        if dx > 0:
            return (xu, xu + length, yu, yu, zu, zu)
        elif dx < 0:
            return (xu - length, xu, yu, yu, zu, zu)
        elif dy > 0:
            return (xu, xu, yu, yu + length, zu, zu)
        elif dy < 0:
            return (xu, xu, yu - length, yu, zu, zu)
        elif dz > 0:
            return (xu, xu, yu, yu, zu, zu + length)
        else:  # dz < 0
            return (xu, xu, yu, yu, zu - length, zu)

    def apply_clearances(self) -> None:
        """Enforce disjunctive Big-M geometric clearance and non-overlap constraints."""
        self.clearance_builder.build()

    def apply_clearance_constraints(self) -> None:
        """Enforce disjunctive Big-M geometric clearance constraints (alias for apply_clearances)."""
        self.apply_clearances()

    def apply_parent_room_separation(self) -> None:
        """Enforce room-room non-overlap separation among parent rooms."""
        self.clearance_builder.add_parent_room_separation()

    def apply_child_child_separation(self) -> None:
        """Enforce pairwise child-child sub-box separation among non-shared child clusters."""
        self.clearance_builder.add_child_child_separation()

    def apply_room_child_separation(self) -> None:
        """Enforce parent room vs. child sub-box clearance."""
        self.clearance_builder.add_room_child_separation()

    def apply_corridor_room_separation(self) -> None:
        """Enforce corridor segment non-penetration against third-party parent rooms."""
        self.clearance_builder.add_corridor_room_separation()

    def apply_corridor_child_separation(self) -> None:
        """Enforce corridor segment non-penetration against sibling child sub-boxes."""
        self.clearance_builder.add_corridor_child_separation()

    def apply_corridor_crossing_separation(self) -> None:
        """Enforce corridor crossing separation between non-incident orthogonal corridor segments."""
        self.clearance_builder.add_corridor_crossing_separation()


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
