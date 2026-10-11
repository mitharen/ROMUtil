"""Stage 2: Basic MILP formulation for layout cluster optimization."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.milp.clearance import ClearanceConstraintBuilder
from romutil.layout.milp.inventory import ActiveEdge, ClusterInventory, EndpointRef
from romutil.types import LayoutOptions


class ClusterFormulation:
    """MILP formulation model and geometric accessors for layout cluster optimization."""

    def __init__(
        self,
        inventory: ClusterInventory,
        options: LayoutOptions | None = None,
    ) -> None:
        """Initialize the MILP formulation with decision variables and base constraints.

        Args:
            inventory: ClusterInventory containing classified rooms, children, alignments,
                and active edges.
            options: Optional layout configuration parameters.
        """
        self.inventory = inventory
        self.options = options
        self.m_coord = max(1000, 100 * (len(inventory.rooms) + len(inventory.children) * 10))
        self.m_big = 2 * self.m_coord
        self.model = pyo.ConcreteModel()
        self._build_model()

    def _build_model(self) -> None:
        """Declare variables, fix anchors, and attach baseline constraints and objective."""
        inventory = self.inventory
        model = self.model
        m_coord = self.m_coord

        # Parent room coordinates
        if inventory.rooms:
            model.x = pyo.Var(inventory.rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))
            model.y = pyo.Var(inventory.rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))
            model.z = pyo.Var(inventory.rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))

            anchor = inventory.rooms[0]
            model.x[anchor].fix(0)
            model.y[anchor].fix(0)
            model.z[anchor].fix(0)

        # Child macro-box origins
        child_ids = [c.node_id for c in inventory.children]
        model.cx = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.cy = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.cz = pyo.Var(child_ids, domain=pyo.Integers, bounds=(-m_coord, m_coord))

        if not inventory.rooms and child_ids:
            first_c = child_ids[0]
            model.cx[first_c].fix(0)
            model.cy[first_c].fix(0)
            model.cz[first_c].fix(0)

        # Gateway alignment equalities
        model.align_x = pyo.ConstraintList()
        model.align_y = pyo.ConstraintList()
        model.align_z = pyo.ConstraintList()

        for align in inventory.room_alignments:
            model.align_x.add(
                model.x[align.room_vnum] == model.cx[align.child_id] + align.child_coord.x
            )
            model.align_y.add(
                model.y[align.room_vnum] == model.cy[align.child_id] + align.child_coord.y
            )
            model.align_z.add(
                model.z[align.room_vnum] == model.cz[align.child_id] + align.child_coord.z
            )

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
                elif ae.allow_transverse_slack:
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_x.add(dst_x - src_x == sp - sm)
                else:
                    model.edge_x.add(dst_x - src_x == 0)

                if dy != 0:
                    model.edge_y.add(dst_y - src_y == model.L[i] * dy)
                elif ae.allow_transverse_slack:
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_y.add(dst_y - src_y == sp - sm)
                else:
                    model.edge_y.add(dst_y - src_y == 0)

                if dz != 0:
                    model.edge_z.add(dst_z - src_z == model.L[i] * dz)
                elif ae.allow_transverse_slack:
                    sp = model.s_transverse_p.add()
                    sm = model.s_transverse_m.add()
                    model.edge_z.add(dst_z - src_z == sp - sm)
                else:
                    model.edge_z.add(dst_z - src_z == 0)

        # Parent bounding box variables and constraints
        model.x_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.x_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))

        model.bbox_bounds = pyo.ConstraintList()
        for u in inventory.rooms:
            model.bbox_bounds.add(model.x_max >= model.x[u])
            model.bbox_bounds.add(model.x_min <= model.x[u])
            model.bbox_bounds.add(model.y_max >= model.y[u])
            model.bbox_bounds.add(model.y_min <= model.y[u])
            model.bbox_bounds.add(model.z_max >= model.z[u])
            model.bbox_bounds.add(model.z_min <= model.z[u])

        for child in inventory.children:
            cid = child.node_id
            c_sol = inventory.child_solutions.get(cid)
            box = c_sol.bounding_box if c_sol else None
            if box is not None:
                model.bbox_bounds.add(model.x_max >= model.cx[cid] + box.max_x)
                model.bbox_bounds.add(model.x_min <= model.cx[cid] + box.min_x)
                model.bbox_bounds.add(model.y_max >= model.cy[cid] + box.max_y)
                model.bbox_bounds.add(model.y_min <= model.cy[cid] + box.min_y)
                model.bbox_bounds.add(model.z_max >= model.cz[cid] + box.max_z)
                model.bbox_bounds.add(model.z_min <= model.cz[cid] + box.min_z)
            else:
                model.bbox_bounds.add(model.x_max >= model.cx[cid])
                model.bbox_bounds.add(model.x_min <= model.cx[cid])
                model.bbox_bounds.add(model.y_max >= model.cy[cid])
                model.bbox_bounds.add(model.y_min <= model.cy[cid])
                model.bbox_bounds.add(model.z_max >= model.cz[cid])
                model.bbox_bounds.add(model.z_min <= model.cz[cid])

        # Objective function
        stretch_expr = (
            sum(
                10 * (model.L[i] - inventory.active_edges[i].min_length)
                for i in edge_indices
            )
            if edge_indices
            else 0
        )
        transverse_penalty = (
            sum(
                500.0 * (model.s_transverse_p[k] + model.s_transverse_m[k])
                for k in model.s_transverse_p
            )
            if hasattr(model, "s_transverse_p") and len(model.s_transverse_p) > 0
            else 0
        )
        span_expr = (
            (model.x_max - model.x_min)
            + (model.y_max - model.y_min)
            + (model.z_max - model.z_min)
        )
        model.obj = pyo.Objective(
            expr=stretch_expr + span_expr + transverse_penalty,
            sense=pyo.minimize,
        )

    def get_point(self, ref: EndpointRef) -> tuple[Any, Any, Any]:
        """Return 3D coordinate expression (x, y, z) for an endpoint reference."""
        if isinstance(ref, int):
            return (self.model.x[ref], self.model.y[ref], self.model.z[ref])
        cid, v = ref
        loc = self.inventory.child_solutions[cid].local_coordinates[v]
        return (
            self.model.cx[cid] + loc.x,
            self.model.cy[cid] + loc.y,
            self.model.cz[cid] + loc.z,
        )

    def get_room_box(self, u: int) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box (min_x, max_x, min_y, max_y, min_z, max_z) for a parent room."""
        px, py, pz = self.get_point(u)
        return (px, px, py, py, pz, pz)

    def get_child_sub_box(self, key: tuple[str, int]) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for a child sub-box indexed by (child_id, index)."""
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

    def apply_clearances(self) -> None:
        """Enforce disjunctive Big-M geometric clearance and non-overlap constraints."""
        ClearanceConstraintBuilder(self).build()

    def apply_clearance_constraints(self) -> None:
        """Enforce disjunctive Big-M geometric clearance constraints (alias for apply_clearances)."""
        self.apply_clearances()

    def apply_parent_room_separation(self) -> None:
        """Enforce room-room non-overlap separation among parent rooms."""
        ClearanceConstraintBuilder(self).add_parent_room_separation()

    def apply_child_child_separation(self) -> None:
        """Enforce pairwise child-child sub-box separation among non-shared child clusters."""
        ClearanceConstraintBuilder(self).add_child_child_separation()

    def apply_room_child_separation(self) -> None:
        """Enforce parent room vs. child sub-box clearance."""
        ClearanceConstraintBuilder(self).add_room_child_separation()

    def apply_corridor_room_separation(self) -> None:
        """Enforce corridor segment non-penetration against third-party parent rooms."""
        ClearanceConstraintBuilder(self).add_corridor_room_separation()

    def apply_corridor_child_separation(self) -> None:
        """Enforce corridor segment non-penetration against sibling child sub-boxes."""
        ClearanceConstraintBuilder(self).add_corridor_child_separation()

    def apply_corridor_crossing_separation(self) -> None:
        """Enforce corridor crossing separation between non-incident orthogonal corridor segments."""
        ClearanceConstraintBuilder(self).add_corridor_crossing_separation()


def formulate_cluster_model(
    inventory: ClusterInventory,
    options: LayoutOptions | None = None,
) -> ClusterFormulation:
    """Build the baseline Pyomo model for layout cluster optimization.

    Factory function instantiating ClusterFormulation.

    Args:
        inventory: ClusterInventory containing classified rooms, children, alignments,
            and active edges.
        options: Optional layout configuration parameters.

    Returns:
        ClusterFormulation wrapping the Pyomo model and coordinate accessors.
    """
    return ClusterFormulation(inventory=inventory, options=options)


__all__ = [
    "ClusterFormulation",
    "formulate_cluster_model",
]
