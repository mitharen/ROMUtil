"""Stage 3: 1D positional constraint DAG for pre-injection clearance cut filtering."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.milp.inventory import (
    ActiveEdge,
    EndpointRef,
    get_child_coordinate,
)

if TYPE_CHECKING:
    from romutil.layout.milp.formulation import ClusterFormulation
    from romutil.layout.milp.inventory import ClusterInventory

logger = logging.getLogger(__name__)

AXIS_X: int = 0
AXIS_Y: int = 1
AXIS_Z: int = 2
AXIS_NAMES: tuple[str, str, str] = ("x", "y", "z")

DIR_RIGHT: str = "r"
DIR_LEFT: str = "l"
DIR_AHEAD: str = "a"
DIR_BEHIND: str = "b"
DIR_UP: str = "u"
DIR_DOWN: str = "d"

ALL_DIRECTIONS: tuple[str, ...] = (
    DIR_RIGHT,
    DIR_LEFT,
    DIR_AHEAD,
    DIR_BEHIND,
    DIR_UP,
    DIR_DOWN,
)

# Maps direction code to (axis_index, sense_a_gt_b)
# sense_a_gt_b is True if item A is strictly greater than item B along axis (A in +axis of B)
DIRECTION_AXIS_MAP: Mapping[str, tuple[int, bool]] = {
    DIR_RIGHT: (AXIS_X, True),
    DIR_LEFT: (AXIS_X, False),
    DIR_AHEAD: (AXIS_Y, True),
    DIR_BEHIND: (AXIS_Y, False),
    DIR_UP: (AXIS_Z, True),
    DIR_DOWN: (AXIS_Z, False),
}

NodeRef = int | str


@dataclass(frozen=True)
class SpatialBoundExpr:
    """1D affine coordinate bound expression [min_node + min_const, max_node + max_const]."""

    min_node: NodeRef
    min_const: int
    max_node: NodeRef
    max_const: int


EntitySpatialBounds = tuple[SpatialBoundExpr, SpatialBoundExpr, SpatialBoundExpr]


def resolve_endpoint_node_offset(
    ref: EndpointRef,
    inventory: ClusterInventory | None,
) -> tuple[NodeRef, tuple[int, int, int]]:
    """Map an endpoint reference to its base decision variable node and 3D integer coordinate offset.

    For parent room or boundary port integers, the base variable node is the integer itself
    with zero offset (0, 0, 0).
    For child port references (cid, room_id), the base variable node is the child cluster origin
    identifier (str) with offset equal to the child-local port coordinate.
    """
    if isinstance(ref, int):
        return ref, (0, 0, 0)

    cid, v = ref
    child_solutions = getattr(inventory, "child_solutions", {}) if inventory is not None else {}
    if cid in child_solutions:
        c_sol = child_solutions[cid]
        loc = get_child_coordinate(c_sol, v)
        if loc is None and hasattr(c_sol, "local_coordinates") and v in c_sol.local_coordinates:
            loc = c_sol.local_coordinates[v]
        if loc is not None:
            return cid, (int(loc.x), int(loc.y), int(loc.z))

    return cid, (0, 0, 0)


def _get_child_sub_box_bounds(
    cid: str,
    k_box: int,
    inventory: ClusterInventory | None,
) -> EntitySpatialBounds | None:
    """Retrieve affine bounding expressions for a child sub-box (cid, k_box)."""
    if inventory is None:
        return None
    child_sub_boxes = getattr(inventory, "child_sub_boxes", {})
    if cid not in child_sub_boxes or k_box >= len(child_sub_boxes[cid]):
        return None
    raw_sb = child_sub_boxes[cid][k_box]
    sb = (
        raw_sb
        if isinstance(raw_sb, BoundingBox3D)
        else BoundingBox3D.from_coordinates([raw_sb[0], raw_sb[1]])
    )
    return (
        SpatialBoundExpr(min_node=cid, min_const=int(sb.min_x), max_node=cid, max_const=int(sb.max_x)),
        SpatialBoundExpr(min_node=cid, min_const=int(sb.min_y), max_node=cid, max_const=int(sb.max_y)),
        SpatialBoundExpr(min_node=cid, min_const=int(sb.min_z), max_node=cid, max_const=int(sb.max_z)),
    )


def _get_corridor_bounds(
    ae: ActiveEdge,
    inventory: ClusterInventory | None,
    full: bool = False,
) -> EntitySpatialBounds:
    """Compute 1D affine coordinate bound expressions for corridor segments.

    Args:
        ae: ActiveEdge representing the corridor connection.
        inventory: ClusterInventory containing child solutions and endpoint references.
        full: If True, include endpoints (full box); if False, exclude endpoints by 1 unit (interior box).
    """
    src_node, src_off = resolve_endpoint_node_offset(ae.src, inventory)
    dst_node, dst_off = resolve_endpoint_node_offset(ae.dst, inventory)
    dx, dy, dz = ae.direction
    interior_margin = 0 if full else 1

    # X axis
    if dx > 0:
        min_nx, min_cx = src_node, src_off[0] + interior_margin
        max_nx, max_cx = dst_node, dst_off[0] - interior_margin
    elif dx < 0:
        min_nx, min_cx = dst_node, dst_off[0] + interior_margin
        max_nx, max_cx = src_node, src_off[0] - interior_margin
    else:
        min_nx, min_cx = src_node, src_off[0]
        max_nx, max_cx = src_node, src_off[0]
    bounds_x = SpatialBoundExpr(min_nx, min_cx, max_nx, max_cx)

    # Y axis
    if dy > 0:
        min_ny, min_cy = src_node, src_off[1] + interior_margin
        max_ny, max_cy = dst_node, dst_off[1] - interior_margin
    elif dy < 0:
        min_ny, min_cy = dst_node, dst_off[1] + interior_margin
        max_ny, max_cy = src_node, src_off[1] - interior_margin
    else:
        min_ny, min_cy = src_node, src_off[1]
        max_ny, max_cy = src_node, src_off[1]
    bounds_y = SpatialBoundExpr(min_ny, min_cy, max_ny, max_cy)

    # Z axis
    if dz > 0:
        min_nz, min_cz = src_node, src_off[2] + interior_margin
        max_nz, max_cz = dst_node, dst_off[2] - interior_margin
    elif dz < 0:
        min_nz, min_cz = dst_node, dst_off[2] + interior_margin
        max_nz, max_cz = src_node, src_off[2] - interior_margin
    else:
        min_nz, min_cz = src_node, src_off[2]
        max_nz, max_cz = src_node, src_off[2]
    bounds_z = SpatialBoundExpr(min_nz, min_cz, max_nz, max_cz)

    return bounds_x, bounds_y, bounds_z


def get_entity_spatial_bounds(
    item: Any,
    category: str,
    inventory: ClusterInventory | None,
    *,
    is_item_a: bool = True,
) -> EntitySpatialBounds | None:
    """Retrieve 3D affine bounding box expressions for an item in a collision violation category.

    Args:
        item: Layout entity reference (room ID, sub-box key, corridor edge index, doorway tuple).
        category: Clearance violation category name.
        inventory: ClusterInventory for resolving child sub-boxes, active edges, and solutions.
        is_item_a: True if item corresponds to item_a in the violation pair; False for item_b.

    Returns:
        3-tuple of SpatialBoundExpr for X, Y, Z axes, or None if the item cannot be resolved.
    """
    if category == "parent_room":
        if not isinstance(item, int):
            return None
        return (
            SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
            SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
            SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
        )

    if category == "child_child":
        if not (isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int)):
            return None
        return _get_child_sub_box_bounds(item[0], item[1], inventory)

    if category == "room_child":
        if is_item_a:
            if not isinstance(item, int):
                return None
            return (
                SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
            )
        else:
            if not (isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int)):
                return None
            return _get_child_sub_box_bounds(item[0], item[1], inventory)

    if category in ("doorway_room", "doorway_child"):
        if is_item_a:
            if not (isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], (tuple, list)) and len(item[1]) == 3):
                return None
            u, disp = item
            dx, dy, dz = int(disp[0]), int(disp[1]), int(disp[2])
            return (
                SpatialBoundExpr(min_node=u, min_const=dx, max_node=u, max_const=dx),
                SpatialBoundExpr(min_node=u, min_const=dy, max_node=u, max_const=dy),
                SpatialBoundExpr(min_node=u, min_const=dz, max_node=u, max_const=dz),
            )
        else:
            if category == "doorway_room":
                if not isinstance(item, int):
                    return None
                return (
                    SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                    SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                    SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                )
            else:
                if not (isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int)):
                    return None
                return _get_child_sub_box_bounds(item[0], item[1], inventory)

    if category in ("corridor_room", "corridor_child"):
        if is_item_a:
            if not isinstance(item, int) or inventory is None:
                return None
            active_edges = getattr(inventory, "active_edges", ())
            if item < 0 or item >= len(active_edges):
                return None
            return _get_corridor_bounds(active_edges[item], inventory, full=False)
        else:
            if category == "corridor_room":
                if not isinstance(item, int):
                    return None
                return (
                    SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                    SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                    SpatialBoundExpr(min_node=item, min_const=0, max_node=item, max_const=0),
                )
            else:
                if not (isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int)):
                    return None
                return _get_child_sub_box_bounds(item[0], item[1], inventory)

    if category == "corridor_crossing":
        if not isinstance(item, int) or inventory is None:
            return None
        active_edges = getattr(inventory, "active_edges", ())
        if item < 0 or item >= len(active_edges):
            return None
        return _get_corridor_bounds(active_edges[item], inventory, full=True)

    return None


class PositionalConstraintDAG:
    """Incremental 1D positional constraint DAG tracking relative coordinate bounds across 3 axes.

    For each axis k in {X, Y, Z}, maintains a system of difference constraints
    where a directed edge u -(w)-> v represents the relative lower bound:
        coord_k(v) - coord_k(u) >= w

    Provides microsecond cycle and contradiction detection to determine if a candidate
    separation cut can be satisfied without introducing a positive cycle (integer infeasibility).
    """

    def __init__(self) -> None:
        self._nodes: dict[int, set[NodeRef]] = {AXIS_X: set(), AXIS_Y: set(), AXIS_Z: set()}
        self._dist: dict[int, dict[NodeRef, dict[NodeRef, int]]] = {
            AXIS_X: collections.defaultdict(dict),
            AXIS_Y: collections.defaultdict(dict),
            AXIS_Z: collections.defaultdict(dict),
        }
        self._edges: dict[int, dict[NodeRef, dict[NodeRef, int]]] = {
            AXIS_X: collections.defaultdict(dict),
            AXIS_Y: collections.defaultdict(dict),
            AXIS_Z: collections.defaultdict(dict),
        }

    def _ensure_node(self, axis: int, u: NodeRef) -> None:
        """Register a node on the specified axis if not already present."""
        if u not in self._nodes[axis]:
            self._nodes[axis].add(u)
            self._dist[axis][u][u] = 0

    def get_lower_bound(self, axis: int, u: NodeRef, v: NodeRef) -> int | None:
        """Retrieve the maximum lower bound on coord_axis(v) - coord_axis(u).

        Returns 0 if u == v, the path weight if a directed path exists, or None if no path exists.
        """
        if u == v:
            return 0
        if u not in self._nodes[axis] or v not in self._nodes[axis]:
            return None
        return self._dist[axis][u].get(v)

    def can_add_edge(self, axis: int, u: NodeRef, v: NodeRef, weight: int) -> bool:
        """Determine if adding the edge u -(weight)-> v would introduce a positive cycle.

        Enforcing coord_axis(v) - coord_axis(u) >= weight creates a contradiction if the existing
        graph already establishes that coord_axis(u) - coord_axis(v) >= w_vu with w_vu + weight > 0.
        """
        if u == v:
            return weight <= 0

        if u not in self._nodes[axis] or v not in self._nodes[axis]:
            return True

        w_vu = self.get_lower_bound(axis, v, u)
        if w_vu is not None and w_vu + weight > 0:
            return False

        return True

    def add_edge(self, axis: int, u: NodeRef, v: NodeRef, weight: int) -> bool:
        """Add a directed constraint edge u -(weight)-> v on the specified axis.

        Returns True if the edge was successfully integrated without contradiction,
        or False if rejected due to positive cycle formation.
        """
        if not self.can_add_edge(axis, u, v, weight):
            return False

        self._ensure_node(axis, u)
        self._ensure_node(axis, v)

        if u == v:
            return True

        self._edges[axis][u][v] = max(self._edges[axis][u].get(v, weight), weight)

        current_d = self._dist[axis][u].get(v)
        if current_d is not None and current_d >= weight:
            return True

        self._dist[axis][u][v] = weight

        nodes_to_u = [
            (i, d_iu)
            for i, d_map in self._dist[axis].items()
            if (d_iu := d_map.get(u)) is not None
        ]
        nodes_from_v = [
            (j, d_vj)
            for j, d_vj in self._dist[axis][v].items()
        ]

        for i, d_iu in nodes_to_u:
            dist_i = self._dist[axis][i]
            for j, d_vj in nodes_from_v:
                candidate = d_iu + weight + d_vj
                old_val = dist_i.get(j)
                if old_val is None or candidate > old_val:
                    dist_i[j] = candidate

        return True

    def add_equality(self, axis: int, u: NodeRef, v: NodeRef, offset: int) -> bool:
        """Add an exact coordinate displacement equality coord_axis(v) - coord_axis(u) == offset."""
        if not self.can_add_edge(axis, u, v, offset):
            return False
        if not self.can_add_edge(axis, v, u, -offset):
            return False
        added_uv = self.add_edge(axis, u, v, offset)
        added_vu = self.add_edge(axis, v, u, -offset)
        return added_uv and added_vu

    def check_candidate_direction(
        self,
        axis: int,
        sense_a_gt_b: bool,
        bounds_a: SpatialBoundExpr,
        bounds_b: SpatialBoundExpr,
    ) -> tuple[bool, str]:
        """Check whether a candidate separation direction along an axis is topologically valid.

        Args:
            axis: Coordinate axis index (0 for X, 1 for Y, 2 for Z).
            sense_a_gt_b: True if item A is strictly greater than item B (A in +axis of B);
                False if item B is strictly greater than item A (B in +axis of A).
            bounds_a: SpatialBoundExpr for item A along the axis.
            bounds_b: SpatialBoundExpr for item B along the axis.

        Returns:
            Tuple of (is_valid, explanation_message).
        """
        axis_name = AXIS_NAMES[axis]
        if sense_a_gt_b:
            # Enforce min_A - max_B >= 1
            # (min_node_a + min_const_a) - (max_node_b + max_const_b) >= 1
            # min_node_a - max_node_b >= 1 + max_const_b - min_const_a
            # Directed edge: max_node_b -(d)-> min_node_a
            src_node = bounds_b.max_node
            dst_node = bounds_a.min_node
            d = 1 + bounds_b.max_const - bounds_a.min_const
            desc = f"{dst_node} - {src_node} >= {d}"
        else:
            # Enforce min_B - max_A >= 1
            # (min_node_b + min_const_b) - (max_node_a + max_const_a) >= 1
            # min_node_b - max_node_a >= 1 + max_const_a - min_const_b
            # Directed edge: max_node_a -(d)-> min_node_b
            src_node = bounds_a.max_node
            dst_node = bounds_b.min_node
            d = 1 + bounds_a.max_const - bounds_b.min_const
            desc = f"{dst_node} - {src_node} >= {d}"

        if not self.can_add_edge(axis, src_node, dst_node, d):
            w_existing = self.get_lower_bound(axis, dst_node, src_node)
            upper_bound = -w_existing if w_existing is not None else "0"
            return False, (
                f"Contradicted along axis '{axis_name}': requires {desc}, but DAG enforces "
                f"upper bound <= {upper_bound} (cycle weight {w_existing + d if w_existing is not None else d} > 0)."
            )

        return True, "Feasible"

    def filter_candidate_directions(
        self,
        item_a: Any,
        item_b: Any,
        category: str,
        inventory: ClusterInventory | None,
    ) -> tuple[list[str], dict[str, str]]:
        """Filter candidate Big-M separation directions against the positional constraint DAG.

        Args:
            item_a: First entity reference in the clearance collision.
            item_b: Second entity reference in the clearance collision.
            category: Collision category (e.g., 'parent_room', 'child_child', 'corridor_room').
            inventory: ClusterInventory containing classified nodes, edges, and sub-boxes.

        Returns:
            Tuple of (valid_directions, pruned_reasons) where valid_directions contains the allowed
            direction characters ('r', 'l', 'a', 'b', 'u', 'd') and pruned_reasons maps eliminated
            directions to diagnostic explanations.
        """
        bounds_a = get_entity_spatial_bounds(item_a, category, inventory, is_item_a=True)
        bounds_b = get_entity_spatial_bounds(item_b, category, inventory, is_item_a=False)

        if bounds_a is None or bounds_b is None:
            return list(ALL_DIRECTIONS), {}

        valid_dirs: list[str] = []
        pruned_reasons: dict[str, str] = {}

        for d_name in ALL_DIRECTIONS:
            axis, sense = DIRECTION_AXIS_MAP[d_name]
            is_valid, reason = self.check_candidate_direction(
                axis, sense, bounds_a[axis], bounds_b[axis]
            )
            if is_valid:
                valid_dirs.append(d_name)
            else:
                pruned_reasons[d_name] = reason

        return valid_dirs, pruned_reasons

    def commit_direction(
        self,
        item_a: Any,
        item_b: Any,
        category: str,
        inventory: ClusterInventory | None,
        direction: str,
    ) -> bool:
        """Commit a chosen or fixed separation direction into the DAG.

        Args:
            item_a: First entity reference.
            item_b: Second entity reference.
            category: Collision category.
            inventory: ClusterInventory.
            direction: Direction code ('r', 'l', 'a', 'b', 'u', 'd').

        Returns:
            True if the directional cut was successfully integrated into the DAG; False if contradicted.
        """
        if direction not in DIRECTION_AXIS_MAP:
            return False

        bounds_a = get_entity_spatial_bounds(item_a, category, inventory, is_item_a=True)
        bounds_b = get_entity_spatial_bounds(item_b, category, inventory, is_item_a=False)
        if bounds_a is None or bounds_b is None:
            return True

        axis, sense_a_gt_b = DIRECTION_AXIS_MAP[direction]
        if sense_a_gt_b:
            src_node = bounds_b[axis].max_node
            dst_node = bounds_a[axis].min_node
            d = 1 + bounds_b[axis].max_const - bounds_a[axis].min_const
        else:
            src_node = bounds_a[axis].max_node
            dst_node = bounds_b[axis].min_node
            d = 1 + bounds_a[axis].max_const - bounds_b[axis].min_const

        return self.add_edge(axis, src_node, dst_node, d)

    @classmethod
    def from_formulation(cls, formulation: ClusterFormulation) -> PositionalConstraintDAG:
        """Instantiate and seed a PositionalConstraintDAG from a ClusterFormulation instance.

        Extracts directional minimum distance bounds from active edges and exact coordinate
        equalities from child-child alignments.
        """
        dag = cls()
        inventory = getattr(formulation, "inventory", None)
        if inventory is None:
            return dag

        # 1. Seed from child alignments (rigid coordinate equalities)
        child_alignments = getattr(inventory, "child_alignments", ())
        if isinstance(child_alignments, (list, tuple)):
            for c_align in child_alignments:
                base_cid = getattr(c_align, "base_child_id", None)
                target_cid = getattr(c_align, "target_child_id", None)
                base_c = getattr(c_align, "base_coord", None)
                target_c = getattr(c_align, "target_coord", None)
                if base_cid is not None and target_cid is not None and base_c is not None and target_c is not None:
                    for axis, attr in enumerate(AXIS_NAMES):
                        w = int(getattr(base_c, attr)) - int(getattr(target_c, attr))
                        dag.add_equality(axis, base_cid, target_cid, w)

        # 2. Seed from active edges
        active_edges = getattr(inventory, "active_edges", ())
        if isinstance(active_edges, (list, tuple)):
            for ae in active_edges:
                if not hasattr(ae, "direction") or not hasattr(ae, "src") or not hasattr(ae, "dst"):
                    continue
                dx, dy, dz = ae.direction
                min_len = int(getattr(ae, "min_length", 1))
                allow_transverse = bool(getattr(ae, "allow_transverse_slack", False))

                src_node, src_off = resolve_endpoint_node_offset(ae.src, inventory)
                dst_node, dst_off = resolve_endpoint_node_offset(ae.dst, inventory)

                # Process X axis
                if dx > 0:
                    w = min_len * dx + src_off[0] - dst_off[0]
                    dag.add_edge(AXIS_X, src_node, dst_node, w)
                elif dx < 0:
                    w = min_len * (-dx) + dst_off[0] - src_off[0]
                    dag.add_edge(AXIS_X, dst_node, src_node, w)

                # Process Y axis
                if dy > 0:
                    w = min_len * dy + src_off[1] - dst_off[1]
                    dag.add_edge(AXIS_Y, src_node, dst_node, w)
                elif dy < 0:
                    w = min_len * (-dy) + dst_off[1] - src_off[1]
                    dag.add_edge(AXIS_Y, dst_node, src_node, w)

                # Process Z axis
                if dz > 0:
                    w = min_len * dz + src_off[2] - dst_off[2]
                    dag.add_edge(AXIS_Z, src_node, dst_node, w)
                elif dz < 0:
                    w = min_len * (-dz) + dst_off[2] - src_off[2]
                    dag.add_edge(AXIS_Z, dst_node, src_node, w)

        return dag
