"""Exact collision-time geometry growth for hierarchical MILP clearance cuts."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.milp.inventory import (
    ActiveEdge,
    EndpointRef,
    get_child_coordinate,
)
from romutil.layout.milp.spatial_dag import (
    AXIS_Z,
    DIRECTION_AXIS_MAP,
    ClusterPositionalDAG,
    EntitySpatialBounds,
    SpatialBoundExpr,
    _get_corridor_bounds,
    _room_spatial_bounds,
    resolve_endpoint_node_offset,
)
from romutil.layout.positional_dag import NodeRef
from romutil.types import Coord3D

if TYPE_CHECKING:
    from romutil.layout.milp.inventory import ClusterInventory
    from romutil.layout.milp.solver import SolvedClusterExtraction

_PLANAR_DIRS: tuple[str, ...] = ("r", "l", "a", "b")
_ALL_DIRS: tuple[str, ...] = ("r", "l", "a", "b", "u", "d")


def _as_bbox3d(raw_sb: BoundingBox3D | tuple[Coord3D, Coord3D]) -> BoundingBox3D:
    """Normalize a BoundingBox3D or legacy 2-point tuple into a BoundingBox3D."""
    if isinstance(raw_sb, BoundingBox3D):
        return raw_sb
    return BoundingBox3D.from_coordinates([raw_sb[0], raw_sb[1]])


def extract_entity_coord(
    ref: EndpointRef,
    extraction: SolvedClusterExtraction,
    inventory: ClusterInventory,
) -> tuple[int, int, int]:
    """Retrieve absolute integer coordinate (x, y, z) for an endpoint reference."""
    if isinstance(ref, int):
        if ref in extraction.raw_parent_coords:
            return extraction.raw_parent_coords[ref]
        if ref in extraction.raw_port_coords:
            return extraction.raw_port_coords[ref]
        raise KeyError(
            f"Room or boundary port {ref} not found in solved extraction coordinates."
        )

    cid, v = ref
    cx, cy, cz = extraction.raw_child_offsets[cid]
    c_sol = inventory.child_solutions[cid]
    loc = get_child_coordinate(c_sol, v)
    if loc is None:
        loc = c_sol.local_coordinates[v]
    return (cx + loc.x, cy + loc.y, cz + loc.z)


def sub_boxes_to_cells(
    sub_boxes: Sequence[BoundingBox3D | tuple[Coord3D, Coord3D]],
) -> set[tuple[int, int, int]]:
    """Expand keep-out sub-boxes into discrete 3D integer cell coordinates."""
    cells: set[tuple[int, int, int]] = set()
    for raw_sb in sub_boxes:
        box = _as_bbox3d(raw_sb)
        for x in range(box.min_x, box.max_x + 1):
            for y in range(box.min_y, box.max_y + 1):
                for z in range(box.min_z, box.max_z + 1):
                    cells.add((x, y, z))
    return cells


def fill_enclosed_2d_holes(
    pts_2d: set[tuple[int, int]],
    max_hole_size: int,
) -> set[tuple[int, int]]:
    """Fill 2D-enclosed interior holes with cell count at most max_hole_size."""
    if not pts_2d or max_hole_size <= 0:
        return set(pts_2d)

    min_x = min(x for x, _y in pts_2d) - 1
    max_x = max(x for x, _y in pts_2d) + 1
    min_y = min(y for _x, y in pts_2d) - 1
    max_y = max(y for _x, y in pts_2d) + 1

    exterior: set[tuple[int, int]] = {(min_x, min_y)}
    queue: collections.deque[tuple[int, int]] = collections.deque([(min_x, min_y)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if (
                min_x <= nx <= max_x
                and min_y <= ny <= max_y
                and (nx, ny) not in pts_2d
                and (nx, ny) not in exterior
            ):
                exterior.add((nx, ny))
                queue.append((nx, ny))

    filled = set(pts_2d)
    unvisited = {
        (x, y)
        for x in range(min_x + 1, max_x)
        for y in range(min_y + 1, max_y)
        if (x, y) not in pts_2d and (x, y) not in exterior
    }
    while unvisited:
        seed = next(iter(unvisited))
        comp = {seed}
        unvisited.remove(seed)
        comp_q: collections.deque[tuple[int, int]] = collections.deque([seed])
        while comp_q:
            cx, cy = comp_q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cx + dx, cy + dy)
                if nb in unvisited:
                    unvisited.remove(nb)
                    comp.add(nb)
                    comp_q.append(nb)
        if len(comp) <= max_hole_size:
            filled |= comp

    return filled


def dilate_2d_along_axis(
    pts_2d: set[tuple[int, int]],
    axis: str,
    neg: int = 1,
    pos: int = 1,
) -> set[tuple[int, int]]:
    """Expand each 2D cell along axis ('x' or 'y') by [-neg, +pos]."""
    out = set(pts_2d)
    if axis == "x":
        for x, y in pts_2d:
            for delta in range(-neg, pos + 1):
                out.add((x + delta, y))
    else:
        for x, y in pts_2d:
            for delta in range(-neg, pos + 1):
                out.add((x, y + delta))
    return out


def fill_narrow_gaps_along_axis(
    pts_2d: set[tuple[int, int]],
    axis: str,
    min_clear_span: int,
) -> set[tuple[int, int]]:
    """Fill collinear empty gaps strictly shorter than min_clear_span along axis."""
    if min_clear_span <= 1 or not pts_2d:
        return set(pts_2d)

    filled = set(pts_2d)
    if axis == "x":
        by_row: dict[int, list[int]] = collections.defaultdict(list)
        for x, y in pts_2d:
            by_row[y].append(x)
        for y, xs in by_row.items():
            xs.sort()
            for i in range(len(xs) - 1):
                gap = xs[i + 1] - xs[i] - 1
                if 0 < gap < min_clear_span:
                    for gx in range(xs[i] + 1, xs[i + 1]):
                        filled.add((gx, y))
    else:
        by_col: dict[int, list[int]] = collections.defaultdict(list)
        for x, y in pts_2d:
            by_col[x].append(y)
        for x, ys in by_col.items():
            ys.sort()
            for i in range(len(ys) - 1):
                gap = ys[i + 1] - ys[i] - 1
                if 0 < gap < min_clear_span:
                    for gy in range(ys[i] + 1, ys[i + 1]):
                        filled.add((x, gy))
    return filled


def max_rect_2d_containing(
    pts_2d: set[tuple[int, int]],
    sx: int,
    sy: int,
) -> tuple[int, int, int, int]:
    """Return the maximum-area solid rectangle in pts_2d containing (sx, sy)."""
    if (sx, sy) not in pts_2d:
        return (sx, sx, sy, sy)

    min_x = min(x for x, _y in pts_2d)
    max_x = max(x for x, _y in pts_2d)
    min_y = min(y for _x, y in pts_2d)
    max_y = max(y for _x, y in pts_2d)

    left: dict[int, int] = {}
    right: dict[int, int] = {}
    for y in range(min_y, max_y + 1):
        if (sx, y) not in pts_2d:
            continue
        lx = sx
        while (lx - 1, y) in pts_2d:
            lx -= 1
        rx = sx
        while (rx + 1, y) in pts_2d:
            rx += 1
        left[y] = lx
        right[y] = rx

    y_lo = sy
    while (y_lo - 1) in left:
        y_lo -= 1
    y_hi = sy
    while (y_hi + 1) in left:
        y_hi += 1

    best_score: tuple[int, int, int, int, int] = (-1, 0, 0, 0, 0)
    best_rect = (sx, sx, sy, sy)
    for y0 in range(y_lo, sy + 1):
        cur_l = min_x
        cur_r = max_x
        for y in range(y0, sy + 1):
            cur_l = max(cur_l, left[y])
            cur_r = min(cur_r, right[y])
        if cur_l > cur_r:
            continue
        for y1 in range(sy, y_hi + 1):
            cur_l = max(cur_l, left[y1])
            cur_r = min(cur_r, right[y1])
            if cur_l > cur_r:
                break
            area = (cur_r - cur_l + 1) * (y1 - y0 + 1)
            score = (area, -cur_l, -y0, cur_r, y1)
            if score > best_score:
                best_score = score
                best_rect = (cur_l, cur_r, y0, y1)

    return best_rect


def max_box_3d_containing(
    pts_3d: set[tuple[int, int, int]],
    sx: int,
    sy: int,
    sz: int,
) -> BoundingBox3D:
    """Grow a solid 3D box in pts_3d containing (sx, sy, sz) by extending Z from the max 2D rect."""
    if (sx, sy, sz) not in pts_3d:
        return BoundingBox3D(sx, sx, sy, sy, sz, sz)

    slice_2d = {(x, y) for x, y, z in pts_3d if z == sz}
    x0, x1, y0, y1 = max_rect_2d_containing(slice_2d, sx, sy)
    z0 = z1 = sz
    while all((x, y, z0 - 1) in pts_3d for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)):
        z0 -= 1
    while all((x, y, z1 + 1) in pts_3d for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)):
        z1 += 1
    return BoundingBox3D(x0, x1, y0, y1, z0, z1)


def compute_planar_component_sizes(inventory: ClusterInventory) -> dict[int | str, int]:
    """Compute conservative cell-count lower bounds for strictly planar (dz == 0) components."""
    room_extents = getattr(inventory, "room_extents", {})

    def _room_weight(vnum: int) -> int:
        lo, hi = room_extents.get(vnum, (0, 0)) if isinstance(room_extents, Mapping) else (0, 0)
        side = lo + hi + 1
        return side * side

    adj: dict[int | str, set[int | str]] = collections.defaultdict(set)
    node_weight: dict[int | str, int] = {r: _room_weight(r) for r in inventory.rooms}
    for bp in inventory.boundary_ports:
        node_weight.setdefault(bp, _room_weight(bp))

    for child in inventory.children:
        cid = child.node_id
        cells = sub_boxes_to_cells(inventory.child_sub_boxes.get(cid, ()))
        by_z: dict[int, int] = collections.defaultdict(int)
        for _x, _y, z in cells:
            by_z[z] += 1
        node_weight[cid] = max(1, min(by_z.values())) if by_z else 1

    for ae in inventory.active_edges:
        if ae.direction[2] != 0:
            continue
        u: int | str = ae.src[0] if isinstance(ae.src, tuple) else ae.src
        v: int | str = ae.dst[0] if isinstance(ae.dst, tuple) else ae.dst
        node_weight.setdefault(u, 1)
        node_weight.setdefault(v, 1)
        adj[u].add(v)
        adj[v].add(u)

    comp_size: dict[int | str, int] = {}
    for start in list(node_weight.keys()):
        if start in comp_size:
            continue
        seen: set[int | str] = {start}
        queue: collections.deque[int | str] = collections.deque([start])
        while queue:
            curr = queue.popleft()
            for nb in adj[curr]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        total = sum(node_weight.get(n, 1) for n in seen)
        for n in seen:
            comp_size[n] = total
    return comp_size


def get_locked_z_diff(
    dag: ClusterPositionalDAG,
    node_a: NodeRef,
    off_a: int,
    node_b: NodeRef,
    off_b: int,
) -> int | None:
    """Return fixed integer (z_a + off_a) - (z_b + off_b) if locked by spatial_dag, else None."""
    if node_a == node_b:
        return off_a - off_b
    lb_b_minus_a = dag.get_lower_bound(AXIS_Z, node_a, node_b)
    lb_a_minus_b = dag.get_lower_bound(AXIS_Z, node_b, node_a)
    if lb_b_minus_a is not None and lb_a_minus_b is not None and lb_a_minus_b == -lb_b_minus_a:
        return lb_a_minus_b + off_a - off_b
    return None


def filter_box_directions(
    dag: ClusterPositionalDAG,
    bounds_a: EntitySpatialBounds,
    bounds_b: EntitySpatialBounds,
    candidate_dirs: Sequence[str],
) -> list[str]:
    """Filter candidate separation directions using DAG bounds for an arbitrary box pair."""
    valid_dirs: list[str] = []
    for d_name in candidate_dirs:
        axis, sense = DIRECTION_AXIS_MAP[d_name]
        is_valid, _ = dag.check_candidate_direction(
            axis, sense, bounds_a[axis], bounds_b[axis]
        )
        if is_valid:
            valid_dirs.append(d_name)
    return valid_dirs


@dataclass(frozen=True)
class GrownClearanceCut:
    """Descriptor for an enlarged, provably collision-equivalent separation cut."""

    kind: str
    norm_key: tuple[Any, ...]
    entity_a: Any
    target_child_id: str
    obstacle_box: BoundingBox3D
    valid_dirs: tuple[str, ...]


def _unwrap_child_ref(item: Any) -> tuple[str, int] | None:
    """Extract (child_id, box_index) from a raw tuple or ChildSubClearanceBox."""
    if hasattr(item, "child_id") and hasattr(item, "box_index"):
        return (str(item.child_id), int(item.box_index))
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], int)
        and item[0] != "edge"
    ):
        return (item[0], item[1])
    return None


def _unwrap_room_vnum(item: Any) -> int | None:
    """Extract integer room ID from a raw int or RoomClearanceBox."""
    if isinstance(item, int):
        return item
    if hasattr(item, "vnum") and isinstance(item.vnum, int):
        return item.vnum
    return None


def _child_offset_bounds(cid: str, box: BoundingBox3D) -> EntitySpatialBounds:
    """Construct DAG affine bounds for child cid translated by constant box."""
    return (
        SpatialBoundExpr(cid, int(box.min_x), cid, int(box.max_x)),
        SpatialBoundExpr(cid, int(box.min_y), cid, int(box.max_y)),
        SpatialBoundExpr(cid, int(box.min_z), cid, int(box.max_z)),
    )


def _point_node_bounds(node: NodeRef) -> EntitySpatialBounds:
    """Construct DAG affine point bounds for a zero-offset variable node."""
    return (
        SpatialBoundExpr(node, 0, node, 0),
        SpatialBoundExpr(node, 0, node, 0),
        SpatialBoundExpr(node, 0, node, 0),
    )


class ExactCollisionCutGrower:
    """Synthesizes exact configuration-space and hole-filled maximal separation boxes on collision."""

    def __init__(
        self,
        inventory: ClusterInventory,
        spatial_dag: ClusterPositionalDAG,
    ) -> None:
        self.inventory = inventory
        self.spatial_dag = spatial_dag
        self._child_cells: dict[str, set[tuple[int, int, int]]] = {}
        self._child_cells_by_z: dict[str, dict[int, set[tuple[int, int]]]] = {}
        for cid, sbs in inventory.child_sub_boxes.items():
            cells = sub_boxes_to_cells(sbs)
            self._child_cells[cid] = cells
            by_z: dict[int, set[tuple[int, int]]] = collections.defaultdict(set)
            for x, y, z in cells:
                by_z[z].add((x, y))
            self._child_cells_by_z[cid] = by_z

        self._planar_comp_sizes = compute_planar_component_sizes(inventory)
        self._hole_filled_cache: dict[tuple[str, int, int], set[tuple[int, int]]] = {}

        self._incident_edge_children: dict[int, set[str]] = {}
        self._incident_room_children: dict[int, set[str]] = {}
        self._child_owned_and_port_rooms: dict[str, set[int]] = {}
        for i, ae in enumerate(inventory.active_edges):
            inc = set(ae.associated_children)
            if isinstance(ae.src, tuple):
                inc.add(ae.src[0])
            if isinstance(ae.dst, tuple):
                inc.add(ae.dst[0])
            self._incident_edge_children[i] = inc
            if isinstance(ae.src, int) and isinstance(ae.dst, tuple):
                self._incident_room_children.setdefault(ae.src, set()).add(ae.dst[0])
            if isinstance(ae.dst, int) and isinstance(ae.src, tuple):
                self._incident_room_children.setdefault(ae.dst, set()).add(ae.src[0])

        for ch in inventory.children:
            c_sol = inventory.child_solutions.get(ch.node_id)
            if c_sol is None:
                continue
            ports: set[int] = set()
            if hasattr(c_sol, "port_coordinates") and c_sol.port_coordinates:
                ports.update(c_sol.port_coordinates.keys())
            if "port_coordinates" in c_sol.metadata:
                ports.update(c_sol.metadata["port_coordinates"].keys())
            owned_or_ports = set(ports)
            if c_sol.local_coordinates:
                owned_or_ports.update(c_sol.local_coordinates.keys())
            self._child_owned_and_port_rooms[ch.node_id] = owned_or_ports
            for port_vnum in ports:
                self._incident_room_children.setdefault(port_vnum, set()).add(ch.node_id)

    def _get_hole_filled_slice(
        self,
        cid: str,
        z_slice: int,
        max_hole_size: int,
    ) -> set[tuple[int, int]]:
        effective_max_hole = min(max_hole_size, 400)
        key = (cid, z_slice, effective_max_hole)
        cached = self._hole_filled_cache.get(key)
        if cached is not None:
            return cached
        raw_slice = self._child_cells_by_z.get(cid, {}).get(z_slice, set())
        if len(raw_slice) > 400:
            self._hole_filled_cache[key] = raw_slice
            return raw_slice
        filled = fill_enclosed_2d_holes(raw_slice, effective_max_hole)
        self._hole_filled_cache[key] = filled
        return filled

    def try_grow_cut(
        self,
        category: str,
        item_a: Any,
        item_b: Any,
        extraction: SolvedClusterExtraction,
    ) -> GrownClearanceCut | None:
        """Attempt to grow an exact maximal obstacle cut for the colliding entity pair."""
        if category == "child_child":
            return self._grow_child_child_cut(item_a, item_b, extraction)
        if category == "corridor_child":
            return self._grow_corridor_child_cut(item_a, item_b, extraction)
        if category == "room_child":
            return self._grow_room_child_cut(item_a, item_b, extraction)
        return None

    def _grow_child_child_cut(
        self,
        item_a: Any,
        item_b: Any,
        extraction: SolvedClusterExtraction,
    ) -> GrownClearanceCut | None:
        ref_a = _unwrap_child_ref(item_a)
        ref_b = _unwrap_child_ref(item_b)
        if ref_a is None or ref_b is None:
            return None
        c1, c2 = ref_a[0], ref_b[0]
        if c1 == c2:
            return None
        if c1 > c2:
            c1, c2 = c2, c1

        o1 = self._child_cells.get(c1)
        o2 = self._child_cells.get(c2)
        if (
            not o1
            or not o2
            or len(o1) * len(o2) > 25_000
            or c1 not in extraction.raw_child_offsets
            or c2 not in extraction.raw_child_offsets
        ):
            return None

        p1 = extraction.raw_child_offsets[c1]
        p2 = extraction.raw_child_offsets[c2]
        dx_cur, dy_cur, dz_cur = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        dz_locked = get_locked_z_diff(self.spatial_dag, c1, 0, c2, 0)

        if dz_locked is not None:
            o1_by_z = self._child_cells_by_z.get(c1, {})
            o2_by_z = self._child_cells_by_z.get(c2, {})
            p_size1 = self._planar_comp_sizes.get(c1, 1)
            p_size2 = self._planar_comp_sizes.get(c2, 1)
            mink_2d: set[tuple[int, int]] = set()
            for z1 in o1_by_z:
                z2 = z1 + dz_locked
                if z2 not in o2_by_z:
                    continue
                s1 = self._get_hole_filled_slice(c1, z1, max(0, p_size2 - 1))
                s2 = self._get_hole_filled_slice(c2, z2, max(0, p_size1 - 1))
                for x2, y2 in s2:
                    for x1, y1 in s1:
                        mink_2d.add((x2 - x1, y2 - y1))
            if (dx_cur, dy_cur) not in mink_2d:
                return None
            rx0, rx1, ry0, ry1 = max_rect_2d_containing(mink_2d, dx_cur, dy_cur)
            box = BoundingBox3D(rx0, rx1, ry0, ry1, dz_locked, dz_locked)
            candidate_dirs: Sequence[str] = _PLANAR_DIRS
        else:
            mink_3d = {
                (x2 - x1, y2 - y1, z2 - z1)
                for (x2, y2, z2) in o2
                for (x1, y1, z1) in o1
            }
            if (dx_cur, dy_cur, dz_cur) not in mink_3d:
                return None
            box = max_box_3d_containing(mink_3d, dx_cur, dy_cur, dz_cur)
            candidate_dirs = _ALL_DIRS

        valid_dirs = filter_box_directions(
            self.spatial_dag,
            _point_node_bounds(c1),
            _child_offset_bounds(c2, box),
            candidate_dirs,
        )
        if not valid_dirs:
            return None

        return GrownClearanceCut(
            kind="mink_child_child",
            norm_key=("mink_child_child", c1, c2, box),
            entity_a=c1,
            target_child_id=c2,
            obstacle_box=box,
            valid_dirs=tuple(valid_dirs),
        )

    def _grow_corridor_child_cut(
        self,
        item_a: Any,
        item_b: Any,
        extraction: SolvedClusterExtraction,
    ) -> GrownClearanceCut | None:
        child_ref = _unwrap_child_ref(item_b)
        if child_ref is None:
            return None
        cid, box_idx = child_ref
        if cid not in extraction.raw_child_offsets:
            return None
        sbs = self.inventory.child_sub_boxes.get(cid, ())
        if box_idx < 0 or box_idx >= len(sbs):
            return None
        sb = _as_bbox3d(sbs[box_idx])

        edge_idx: int | None = None
        if isinstance(item_a, int):
            edge_idx = item_a
        elif hasattr(item_a, "edge_index"):
            edge_idx = int(item_a.edge_index)

        if edge_idx is not None:
            return self._grow_active_edge_child_cut(
                item_a, edge_idx, cid, sb, extraction
            )
        return None

    def _grow_active_edge_child_cut(
        self,
        raw_corridor: Any,
        edge_idx: int,
        cid: str,
        sb: BoundingBox3D,
        extraction: SolvedClusterExtraction,
    ) -> GrownClearanceCut | None:
        if edge_idx < 0 or edge_idx >= len(self.inventory.active_edges):
            return None
        ae: ActiveEdge = self.inventory.active_edges[edge_idx]
        if cid in self._incident_edge_children.get(edge_idx, set()):
            return None
        src_vnum = ae.src[1] if isinstance(ae.src, tuple) else ae.src
        dst_vnum = ae.dst[1] if isinstance(ae.dst, tuple) else ae.dst
        owned_or_ports = self._child_owned_and_port_rooms.get(cid, set())
        if src_vnum in owned_or_ports or dst_vnum in owned_or_ports:
            return None
        if ae.direction[2] != 0 or sum(1 for d in ae.direction if d != 0) != 1:
            return None

        src_node, src_off = resolve_endpoint_node_offset(ae.src, self.inventory)
        dz_locked = get_locked_z_diff(self.spatial_dag, src_node, src_off[2], cid, 0)
        if dz_locked is None:
            return None
        if dz_locked not in self._child_cells_by_z.get(cid, {}):
            return None

        try:
            p1 = extract_entity_coord(ae.src, extraction, self.inventory)
            p2 = extract_entity_coord(ae.dst, extraction, self.inventory)
        except KeyError:
            return None

        cx, cy, _cz = extraction.raw_child_offsets[cid]
        src_lo, src_hi = getattr(ae, "src_extent", (0, 0))
        dst_lo, dst_hi = getattr(ae, "dst_extent", (0, 0))
        t_lo, t_hi = getattr(ae, "transverse_extent", (0, 0))
        dx, dy, _dz = ae.direction
        if dx != 0:
            if dx > 0:
                c_x0 = p1[0] + src_hi + 1 - cx
                c_x1 = p2[0] - dst_lo - 1 - cx
            else:
                c_x0 = p2[0] + dst_hi + 1 - cx
                c_x1 = p1[0] - src_lo - 1 - cx
            c_y0 = p1[1] - t_lo - cy
            c_y1 = p1[1] + t_hi - cy
            axis = "x"
        else:
            if dy > 0:
                c_y0 = p1[1] + src_hi + 1 - cy
                c_y1 = p2[1] - dst_lo - 1 - cy
            else:
                c_y0 = p2[1] + dst_hi + 1 - cy
                c_y1 = p1[1] - src_lo - 1 - cy
            c_x0 = p1[0] - t_lo - cx
            c_x1 = p1[0] + t_hi - cx
            axis = "y"

        ov_x0, ov_x1 = max(sb.min_x, c_x0), min(sb.max_x, c_x1)
        ov_y0, ov_y1 = max(sb.min_y, c_y0), min(sb.max_y, c_y1)
        if ov_x0 > ov_x1 or ov_y0 > ov_y1:
            return None
        sx, sy = ov_x0, ov_y0

        p_size = self._planar_comp_sizes.get(src_node, ae.min_length + 1)
        max_hole = max(0, p_size - 1)
        oc_f = self._get_hole_filled_slice(cid, dz_locked, max_hole)
        has_scaled_extents = any((src_lo, src_hi, dst_lo, dst_hi, t_lo, t_hi))
        if not has_scaled_extents:
            if not ae.allow_transverse_slack:
                # Neither endpoint of a non-incident orthogonal corridor can occupy O_C,
                # so any point within +-1 along the travel axis cannot be a corridor interior cell.
                oc_f = dilate_2d_along_axis(oc_f, axis, 1, 1)
                oc_f = fill_narrow_gaps_along_axis(oc_f, axis, max(1, ae.min_length - 1))
            else:
                # For transverse-slack corridors aligned to src's transverse coordinate, any non-empty
                # interior cell inside a 1-cell travel-axis gap requires |c_src - c_dst| >= 2, placing
                # src in O_C or stretching the interior across O_C; pass min_clear_span=2 (fills gap == 1).
                oc_f = fill_narrow_gaps_along_axis(oc_f, axis, 2)

        if (sx, sy) not in oc_f:
            return None
        rx0, rx1, ry0, ry1 = max_rect_2d_containing(oc_f, sx, sy)
        box = BoundingBox3D(rx0, rx1, ry0, ry1, dz_locked, dz_locked)

        valid_dirs = filter_box_directions(
            self.spatial_dag,
            _get_corridor_bounds(ae, self.inventory, full=False),
            _child_offset_bounds(cid, box),
            _PLANAR_DIRS,
        )
        if not valid_dirs:
            return None

        return GrownClearanceCut(
            kind="grown_corridor_child",
            norm_key=("grown_corridor_child", ("edge", edge_idx), cid, box),
            entity_a=raw_corridor,
            target_child_id=cid,
            obstacle_box=box,
            valid_dirs=tuple(valid_dirs),
        )

    def _grow_room_child_cut(
        self,
        item_a: Any,
        item_b: Any,
        extraction: SolvedClusterExtraction,
    ) -> GrownClearanceCut | None:
        u = _unwrap_room_vnum(item_a)
        child_ref = _unwrap_child_ref(item_b)
        if u is None or child_ref is None:
            return None
        cid = child_ref[0]
        if (
            cid in self._incident_room_children.get(u, set())
            or cid not in extraction.raw_child_offsets
            or (u not in extraction.raw_parent_coords and u not in extraction.raw_port_coords)
        ):
            return None

        dz_locked = get_locked_z_diff(self.spatial_dag, u, 0, cid, 0)
        if dz_locked is None or dz_locked not in self._child_cells_by_z.get(cid, {}):
            return None

        u_coord = (
            extraction.raw_parent_coords[u]
            if u in extraction.raw_parent_coords
            else extraction.raw_port_coords.get(u)
        )
        if u_coord is None:
            return None
        ux, uy, _uz = u_coord
        cx, cy, _cz = extraction.raw_child_offsets[cid]
        room_extents = getattr(self.inventory, "room_extents", {})
        lo, hi = room_extents.get(u, (0, 0)) if isinstance(room_extents, Mapping) else (0, 0)

        p_size = self._planar_comp_sizes.get(u, 1)
        oc_f = self._get_hole_filled_slice(cid, dz_locked, max(0, p_size - 1))
        sx, sy = ux - cx, uy - cy
        if (sx, sy) not in oc_f:
            found_seed: tuple[int, int] | None = None
            for tx in range(ux - lo - cx, ux + hi - cx + 1):
                for ty in range(uy - lo - cy, uy + hi - cy + 1):
                    if (tx, ty) in oc_f:
                        found_seed = (tx, ty)
                        break
                if found_seed is not None:
                    break
            if found_seed is None:
                return None
            sx, sy = found_seed
        rx0, rx1, ry0, ry1 = max_rect_2d_containing(oc_f, sx, sy)
        box = BoundingBox3D(rx0, rx1, ry0, ry1, dz_locked, dz_locked)

        valid_dirs = filter_box_directions(
            self.spatial_dag,
            _room_spatial_bounds(u, self.inventory),
            _child_offset_bounds(cid, box),
            _PLANAR_DIRS,
        )
        if not valid_dirs:
            return None

        return GrownClearanceCut(
            kind="grown_room_child",
            norm_key=("grown_room_child", u, cid, box),
            entity_a=item_a,
            target_child_id=cid,
            obstacle_box=box,
            valid_dirs=tuple(valid_dirs),
        )


__all__ = [
    "ExactCollisionCutGrower",
    "GrownClearanceCut",
    "compute_planar_component_sizes",
    "dilate_2d_along_axis",
    "extract_entity_coord",
    "fill_enclosed_2d_holes",
    "fill_narrow_gaps_along_axis",
    "filter_box_directions",
    "get_locked_z_diff",
    "max_box_3d_containing",
    "max_rect_2d_containing",
    "sub_boxes_to_cells",
]
