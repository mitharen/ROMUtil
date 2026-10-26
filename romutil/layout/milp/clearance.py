"""Stage 3: Geometric clearance and separation constraints for layout cluster MILP."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from romutil.layout.geometry import (
    BoundingBox3D,
    Segment3D,
    segment_room_penetration,
    segments_intersect_3d,
)
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.inventory import (
    ActiveEdge,
    EndpointRef,
    direction_to_displacement,
    get_child_coordinate,
)
from romutil.types import Coord3D

if TYPE_CHECKING:
    from romutil.layout.ir import NodeSolution, SolveTreeNode
    from romutil.layout.milp.formulation import ClusterFormulation
    from romutil.layout.milp.inventory import ClusterInventory
    from romutil.layout.milp.solver import SolvedClusterExtraction


@dataclass(frozen=True)
class ClearanceViolation:
    """Detected geometric collision or clearance violation between cluster layout entities."""

    category: str
    item_a: Any
    item_b: Any
    description: str = ""


VIOLATION_TIER_PRIORITY: Mapping[str, int] = {
    "parent_room": 0,
    "child_child": 1,
    "room_child": 1,
    "doorway_room": 2,
    "doorway_child": 2,
    "corridor_room": 3,
    "corridor_child": 3,
    "corridor_crossing": 4,
}


class ClearanceConstraintBuilder:
    """Builder enforcing disjunctive Big-M geometric clearance and separation constraints."""

    def __init__(self, formulation: ClusterFormulation) -> None:
        self.formulation = formulation
        self.separated_pairs: set[tuple[str, Any, Any]] = set()

    def get_doorway_box(
        self, d_key: tuple[int, tuple[int, int, int]]
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for an external boundary doorway vector."""
        u, disp = d_key
        x, y, z = self.formulation.get_point(u)
        dx, dy, dz = disp
        return (x + dx, x + dx, y + dy, y + dy, z + dz, z + dz)

    @property
    def local_rooms(self) -> tuple[int, ...]:
        """Return the local rooms and external boundary ports instantiated in formulation.model.x."""
        if hasattr(self.formulation, "local_rooms"):
            return self.formulation.local_rooms
        if hasattr(self.formulation.model, "x"):
            return tuple(sorted(self.formulation.model.x.keys()))
        return self.formulation.inventory.rooms

    def add_parent_room_separation(self) -> None:
        """Enforce pairwise room-room non-overlap separation among parent rooms and boundary ports."""
        rooms = self.local_rooms
        if len(rooms) > 1:
            parent_pairs = [
                (rooms[i], rooms[j])
                for i in range(len(rooms))
                for j in range(i + 1, len(rooms))
            ]
            for p in parent_pairs:
                self.separated_pairs.add(
                    ("parent_room", min(p[0], p[1]), max(p[0], p[1]))
                )
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=parent_pairs,
                get_box_a=self.formulation.get_room_box,
                get_box_b=self.formulation.get_room_box,
                prefix="parent_room",
                m_big=self.formulation.m_big,
            )

    def add_child_child_separation(self) -> None:
        """Enforce pairwise child-child sub-box separation across all child clusters."""
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        child_sub_boxes = self.formulation.inventory.child_sub_boxes

        equated = self.formulation.inventory.equated_children
        child_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1 = children[i]
                c2 = children[j]
                if c1.node_id == c2.node_id:
                    continue
                if frozenset({c1.node_id, c2.node_id}) in equated:
                    continue
                sol1 = child_solutions.get(c1.node_id)
                sol2 = child_solutions.get(c2.node_id)
                sb1 = child_sub_boxes.get(c1.node_id, ())
                sb2 = child_sub_boxes.get(c2.node_id, ())
                if sol1 and sol2 and sb1 and sb2:
                    for k1 in range(len(sb1)):
                        for k2 in range(len(sb2)):
                            pair = ((c1.node_id, k1), (c2.node_id, k2))
                            child_pairs.append(pair)
                            self.separated_pairs.add(
                                (
                                    "child_child",
                                    min(pair[0], pair[1]),
                                    max(pair[0], pair[1]),
                                )
                            )

        if child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=child_pairs,
                get_box_a=self.formulation.get_child_sub_box,
                get_box_b=self.formulation.get_child_sub_box,
                prefix="child_child",
                m_big=self.formulation.m_big,
            )

    def add_room_child_separation(self) -> None:
        """Enforce parent room vs. child sub-box clearance."""
        rooms = self.local_rooms
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        gateway_anchors = self.formulation.inventory.gateway_anchors
        child_sub_boxes = self.formulation.inventory.child_sub_boxes

        room_child_pairs: list[tuple[int, tuple[str, int]]] = []
        if rooms:
            for child in children:
                c_sol = child_solutions.get(child.node_id)
                sb = child_sub_boxes.get(child.node_id, ())
                if c_sol and sb:
                    anchors = gateway_anchors.get(child.node_id, frozenset())
                    c_ports = getattr(c_sol, "port_coordinates", {})
                    for u in rooms:
                        if (
                            u not in anchors
                            and u not in c_sol.local_coordinates
                            and u not in c_ports
                        ):
                            for k in range(len(sb)):
                                pair = (u, (child.node_id, k))
                                room_child_pairs.append(pair)
                                self.separated_pairs.add(("room_child", pair[0], pair[1]))

        if room_child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=room_child_pairs,
                get_box_a=self.formulation.get_room_box,
                get_box_b=self.formulation.get_child_sub_box,
                prefix="room_child",
                m_big=self.formulation.m_big,
            )

    def add_corridor_room_separation(self) -> None:
        """Enforce corridor segment non-penetration against third-party parent rooms."""
        rooms = self.local_rooms
        active_edges = self.formulation.inventory.active_edges
        child_solutions = self.formulation.inventory.child_solutions
        edge_indices = list(range(len(active_edges)))

        corridor_room_pairs: list[tuple[int, int]] = []
        if rooms and edge_indices:
            for i, ae in enumerate(active_edges):
                if sum(1 for d in ae.direction if d != 0) != 1:
                    continue
                for w in rooms:
                    if w in ae.associated_rooms:
                        continue
                    if any(
                        w in child_solutions[cid].local_coordinates
                        or w in getattr(child_solutions[cid], "port_coordinates", {})
                        for cid in ae.associated_children
                        if cid in child_solutions
                    ):
                        continue
                    pair = (i, w)
                    corridor_room_pairs.append(pair)
                    self.separated_pairs.add(("corridor_room", pair[0], pair[1]))

        add_disjunctive_box_separation(
            model=self.formulation.model,
            pairs=corridor_room_pairs,
            get_box_a=lambda i: self.formulation.get_corridor_interior_box(
                active_edges[i]
            ),
            get_box_b=self.formulation.get_room_box,
            prefix="corridor_room",
            m_big=self.formulation.m_big,
        )

    def add_corridor_child_separation(self) -> None:
        """Enforce corridor segment non-penetration against sibling child sub-boxes."""
        active_edges = self.formulation.inventory.active_edges
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        child_sub_boxes = self.formulation.inventory.child_sub_boxes
        edge_indices = list(range(len(active_edges)))

        b_ports = self.formulation.inventory.boundary_ports
        corridor_child_pairs: list[tuple[int, tuple[str, int]]] = []
        if edge_indices and len(children) > 0:
            for i, ae in enumerate(active_edges):
                if ae.edge_id.startswith("sp_"):
                    continue
                if sum(1 for d in ae.direction if d != 0) != 1:
                    continue
                for s_child in children:
                    s_id = s_child.node_id
                    s_sol = child_solutions.get(s_id)
                    sb = child_sub_boxes.get(s_id, ())
                    if not s_sol or not sb:
                        continue

                    s_ports = getattr(s_sol, "port_coordinates", {})
                    s_child_rooms = set(s_sol.local_coordinates.keys()) | set(s_ports.keys())
                    if (
                        (
                            isinstance(ae.src, int)
                            and ae.src in b_ports
                            and (
                                ae.dst in s_child_rooms
                                or (isinstance(ae.dst, tuple) and ae.dst[0] == s_id)
                            )
                        )
                        or (
                            isinstance(ae.dst, int)
                            and ae.dst in b_ports
                            and (
                                ae.src in s_child_rooms
                                or (isinstance(ae.src, tuple) and ae.src[0] == s_id)
                            )
                        )
                    ):
                        continue

                    exempt_boxes = get_incident_child_sub_box_indices(
                        s_id, s_sol, sb, ae, child_solutions
                    )
                    for k in range(len(sb)):
                        if k in exempt_boxes:
                            continue
                        pair = (i, (s_id, k))
                        corridor_child_pairs.append(pair)
                        self.separated_pairs.add(("corridor_child", pair[0], pair[1]))

        if corridor_child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=corridor_child_pairs,
                get_box_a=lambda i: self.formulation.get_corridor_interior_box(
                    active_edges[i]
                ),
                get_box_b=self.formulation.get_child_sub_box,
                prefix="corridor_child",
                m_big=self.formulation.m_big,
            )

    def add_corridor_crossing_separation(self) -> None:
        """Enforce corridor crossing separation between non-incident orthogonal corridor segments."""
        active_edges = self.formulation.inventory.active_edges

        crossing_pairs: list[tuple[int, int]] = []
        for i1 in range(len(active_edges)):
            ae1 = active_edges[i1]
            if ae1.edge_id.startswith("sp_"):
                continue
            if sum(1 for d in ae1.direction if d != 0) != 1:
                continue
            act1 = (
                "x"
                if ae1.direction[0] != 0
                else ("y" if ae1.direction[1] != 0 else "z")
            )
            for i2 in range(i1 + 1, len(active_edges)):
                ae2 = active_edges[i2]
                if ae2.edge_id.startswith("sp_"):
                    continue
                if sum(1 for d in ae2.direction if d != 0) != 1:
                    continue
                act2 = (
                    "x"
                    if ae2.direction[0] != 0
                    else ("y" if ae2.direction[1] != 0 else "z")
                )
                if act1 == act2:
                    continue
                incident = bool(
                    ae1.associated_rooms & ae2.associated_rooms
                ) or bool({ae1.src, ae1.dst} & {ae2.src, ae2.dst})
                if not incident:
                    pair = (i1, i2)
                    crossing_pairs.append(pair)
                    self.separated_pairs.add(
                        ("corridor_crossing", min(pair[0], pair[1]), max(pair[0], pair[1]))
                    )

        add_disjunctive_box_separation(
            model=self.formulation.model,
            pairs=crossing_pairs,
            get_box_a=lambda i: self.formulation.get_corridor_full_box(
                active_edges[i]
            ),
            get_box_b=lambda i: self.formulation.get_corridor_full_box(
                active_edges[i]
            ),
            prefix="corridor_crossing",
            m_big=self.formulation.m_big,
        )

    def add_doorway_clearance(self) -> None:
        """Enforce external doorway clearance for boundary ports against child sub-boxes and rooms."""
        node = getattr(self.formulation.inventory, "node", None)
        rooms = self.formulation.inventory.rooms
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        child_sub_boxes = self.formulation.inventory.child_sub_boxes
        graph = self.formulation.inventory.graph
        b_ports = self.formulation.inventory.boundary_ports
        gateway_anchors = getattr(self.formulation.inventory, "gateway_anchors", {})

        if not (node and b_ports and graph):
            return

        sub_rooms: set[int] = set(node.rooms)

        def _collect_subtree(n: SolveTreeNode) -> None:
            sub_rooms.update(n.rooms)
            for ch in n.children:
                _collect_subtree(ch)

        for ch in node.children:
            _collect_subtree(ch)

        active_dirs: dict[int, set[tuple[int, int, int]]] = collections.defaultdict(set)
        for ae in self.formulation.inventory.active_edges:
            if isinstance(ae.src, int):
                active_dirs[ae.src].add(ae.direction)
            if isinstance(ae.dst, int):
                rev = (-ae.direction[0], -ae.direction[1], -ae.direction[2])
                active_dirs[ae.dst].add(rev)

        all_parent_rooms = set(self.local_rooms)
        doorway_child_pairs: list[
            tuple[tuple[int, tuple[int, int, int]], tuple[str, int]]
        ] = []
        doorway_room_pairs: list[
            tuple[tuple[int, tuple[int, int, int]], int]
        ] = []

        for u in b_ports:
            if u not in all_parent_rooms or u not in graph.rooms:
                continue
            for ex in graph.rooms[u].exits:
                if ex.dst_vnum not in sub_rooms:
                    try:
                        disp = direction_to_displacement(ex.direction)
                    except ValueError:
                        continue
                    if disp in active_dirs[u]:
                        continue
                    d_key = (u, disp)
                    for child in children:
                        c_sol = child_solutions.get(child.node_id)
                        sb = child_sub_boxes.get(child.node_id, ())
                        if c_sol and sb:
                            for k in range(len(sb)):
                                pair = (d_key, (child.node_id, k))
                                doorway_child_pairs.append(pair)
                                self.separated_pairs.add(
                                    ("doorway_child", pair[0], pair[1])
                                )

                    for w in all_parent_rooms:
                        if w != u:
                            pair_r = (d_key, w)
                            doorway_room_pairs.append(pair_r)
                            self.separated_pairs.add(
                                ("doorway_room", pair_r[0], pair_r[1])
                            )

        if doorway_child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=doorway_child_pairs,
                get_box_a=self.get_doorway_box,
                get_box_b=self.formulation.get_child_sub_box,
                prefix="doorway_child",
                m_big=self.formulation.m_big,
            )

        if doorway_room_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=doorway_room_pairs,
                get_box_a=self.get_doorway_box,
                get_box_b=self.formulation.get_room_box,
                prefix="doorway_room",
                m_big=self.formulation.m_big,
            )

    def inject_clearance_cuts(
        self,
        violations: Sequence[ClearanceViolation],
        max_cuts_per_pass: int | None = None,
    ) -> int:
        """Inject disjunctive Big-M separation cuts for detected collision violations.

        Sorts candidate un-separated violations according to priority tier and limits
        newly injected cuts to at most `max_cuts_per_pass` (defaulting to
        `formulation.options.max_cuts_per_pass` or 20) per pass to prevent branch-and-bound
        explosion.

        Args:
            violations: Sequence of ClearanceViolation instances detected in the current extraction.
            max_cuts_per_pass: Optional override for the maximum number of cuts to inject in this pass.

        Returns:
            Number of newly injected separation constraints.
        """
        if not violations:
            return 0

        if max_cuts_per_pass is not None:
            budget = max_cuts_per_pass
        else:
            opts = getattr(self.formulation, "options", None)
            budget = getattr(opts, "max_cuts_per_pass", 20) if opts is not None else 20

        if budget <= 0:
            return 0

        candidates: list[tuple[str, tuple[str, Any, Any], tuple[Any, Any]]] = []
        seen: set[tuple[str, Any, Any]] = set()

        for viol in violations:
            cat = viol.category
            a = viol.item_a
            b = viol.item_b

            if cat in ("parent_room", "child_child", "corridor_crossing"):
                norm_key = (cat, min(a, b), max(a, b))
                pair: tuple[Any, Any] = (min(a, b), max(a, b))
            else:
                norm_key = (cat, a, b)
                pair = (a, b)

            if norm_key in self.separated_pairs or norm_key in seen:
                continue
            seen.add(norm_key)
            candidates.append((cat, norm_key, pair))

        if not candidates:
            return 0

        # Sort candidate un-separated violations according to priority tier (stable sort)
        candidates.sort(key=lambda item: VIOLATION_TIER_PRIORITY.get(item[0], 5))

        # Limit newly injected cuts to at most max_cuts_per_pass
        selected = candidates[:budget]

        new_pairs_by_cat: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
        for cat, norm_key, pair in selected:
            self.separated_pairs.add(norm_key)
            new_pairs_by_cat[cat].append(pair)

        for cat, pairs in new_pairs_by_cat.items():
            if cat == "parent_room":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.formulation.get_room_box,
                    get_box_b=self.formulation.get_room_box,
                    prefix="parent_room",
                    m_big=self.formulation.m_big,
                )
            elif cat == "room_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.formulation.get_room_box,
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="room_child",
                    m_big=self.formulation.m_big,
                )
            elif cat == "child_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.formulation.get_child_sub_box,
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="child_child",
                    m_big=self.formulation.m_big,
                )
            elif cat == "corridor_room":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=lambda i: self.formulation.get_corridor_interior_box(
                        self.formulation.inventory.active_edges[i]
                    ),
                    get_box_b=self.formulation.get_room_box,
                    prefix="corridor_room",
                    m_big=self.formulation.m_big,
                )
            elif cat == "corridor_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=lambda i: self.formulation.get_corridor_interior_box(
                        self.formulation.inventory.active_edges[i]
                    ),
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="corridor_child",
                    m_big=self.formulation.m_big,
                )
            elif cat == "corridor_crossing":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=lambda i: self.formulation.get_corridor_full_box(
                        self.formulation.inventory.active_edges[i]
                    ),
                    get_box_b=lambda i: self.formulation.get_corridor_full_box(
                        self.formulation.inventory.active_edges[i]
                    ),
                    prefix="corridor_crossing",
                    m_big=self.formulation.m_big,
                )
            elif cat == "doorway_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.get_doorway_box,
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="doorway_child",
                    m_big=self.formulation.m_big,
                )
            elif cat == "doorway_room":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.get_doorway_box,
                    get_box_b=self.formulation.get_room_box,
                    prefix="doorway_room",
                    m_big=self.formulation.m_big,
                )

        return len(selected)

    def build(self) -> None:
        """Apply all disjunctive Big-M geometric clearance constraints."""
        self.add_parent_room_separation()
        self.add_child_child_separation()
        self.add_room_child_separation()
        self.add_doorway_clearance()
        self.add_corridor_room_separation()
        self.add_corridor_child_separation()
        self.add_corridor_crossing_separation()


def _extract_entity_coord(
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


def _extract_child_sub_box_bounds(
    key: tuple[str, int],
    extraction: SolvedClusterExtraction,
    inventory: ClusterInventory,
) -> tuple[int, int, int, int, int, int]:
    """Retrieve absolute bounding box (min_x, max_x, min_y, max_y, min_z, max_z) for a child sub-box."""
    cid, k = key
    cx, cy, cz = extraction.raw_child_offsets[cid]
    raw_sb = inventory.child_sub_boxes[cid][k]
    sb = (
        raw_sb
        if isinstance(raw_sb, BoundingBox3D)
        else BoundingBox3D.from_coordinates([raw_sb[0], raw_sb[1]])
    )
    return (
        cx + sb.min_x,
        cx + sb.max_x,
        cy + sb.min_y,
        cy + sb.max_y,
        cz + sb.min_z,
        cz + sb.max_z,
    )


def get_incident_child_sub_box_indices(
    child_id: str,
    child_solution: NodeSolution,
    sub_boxes: Sequence[BoundingBox3D],
    ae: ActiveEdge,
    child_solutions: Mapping[str, NodeSolution] | None = None,
) -> set[int]:
    """Identify sub-box indices of child_id enclosing boundary ports incident to active edge ae.

    For an active edge incident to child_id at one or more boundary ports with relative
    coordinate (x, y, z) within the child coordinate frame, identify the specific sub-box
    index b_i of child child_id whose closed bounds enclose or are adjacent (L_inf distance <= 1)
    to that port coordinate. Exempt only those specific incident sub-boxes from collision
    detection and Big-M separation against ae, while keeping all other sub-boxes active obstacles.

    Args:
        child_id: Identifier of the child cluster.
        child_solution: Solved NodeSolution for child_id.
        sub_boxes: Sequence of keep-out sub-boxes for child_id.
        ae: ActiveEdge corridor segment being checked.
        child_solutions: Optional mapping of child IDs to NodeSolutions for resolving
            inter-child shared boundary ports.

    Returns:
        Set of integer indices into sub_boxes that enclose or are adjacent to incident port coordinates.
    """
    if not sub_boxes:
        return set()

    incident_coords: list[Coord3D] = []

    def _collect_vnum(vnum: int) -> None:
        coord = get_child_coordinate(child_solution, vnum)
        if coord is not None:
            incident_coords.append(coord)

    # 1. Inspect active edge endpoints
    if isinstance(ae.src, tuple):
        src_cid, src_v = ae.src
        if src_cid == child_id:
            _collect_vnum(src_v)
        elif child_solutions and src_cid in child_solutions:
            _collect_vnum(src_v)
    elif isinstance(ae.src, int):
        _collect_vnum(ae.src)

    if isinstance(ae.dst, tuple):
        dst_cid, dst_v = ae.dst
        if dst_cid == child_id:
            _collect_vnum(dst_v)
        elif child_solutions and dst_cid in child_solutions:
            _collect_vnum(dst_v)
    elif isinstance(ae.dst, int):
        _collect_vnum(ae.dst)

    # 2. Inspect rooms associated with the edge
    for r in ae.associated_rooms:
        _collect_vnum(r)

    if not incident_coords:
        return set()

    norm_boxes: list[BoundingBox3D] = [
        raw_b
        if isinstance(raw_b, BoundingBox3D)
        else BoundingBox3D.from_coordinates([raw_b[0], raw_b[1]])
        for raw_b in sub_boxes
    ]

    exempt: set[int] = set()
    # Pass 1: Strict enclosure
    for k, box in enumerate(norm_boxes):
        for pt in incident_coords:
            if (
                box.min_x <= pt.x <= box.max_x
                and box.min_y <= pt.y <= box.max_y
                and box.min_z <= pt.z <= box.max_z
            ):
                exempt.add(k)
                break

    # Pass 2: Boundary port carve-out adjacent fallback (L_inf distance <= 1)
    for pt in incident_coords:
        pt_covered = any(
            box.min_x <= pt.x <= box.max_x
            and box.min_y <= pt.y <= box.max_y
            and box.min_z <= pt.z <= box.max_z
            for box in norm_boxes
        )
        if not pt_covered:
            min_dist = float("inf")
            for box in norm_boxes:
                dist = max(
                    max(0, box.min_x - pt.x, pt.x - box.max_x),
                    max(0, box.min_y - pt.y, pt.y - box.max_y),
                    max(0, box.min_z - pt.z, pt.z - box.max_z),
                )
                if dist < min_dist:
                    min_dist = dist
            if min_dist <= 1:
                for k, box in enumerate(norm_boxes):
                    dist = max(
                        max(0, box.min_x - pt.x, pt.x - box.max_x),
                        max(0, box.min_y - pt.y, pt.y - box.max_y),
                        max(0, box.min_z - pt.z, pt.z - box.max_z),
                    )
                    if dist == min_dist:
                        exempt.add(k)

    return exempt


def detect_cluster_collisions(
    formulation: ClusterFormulation,
    extraction: SolvedClusterExtraction,
) -> list[ClearanceViolation]:
    """Detect geometric collisions across all clearance categories from solved cluster coordinates.

    Evaluates exact coordinates and child offsets to identify active collisions across
    all seven clearance categories:
      1. Room–Room: Coordinate equality between parent rooms and boundary ports.
      2. Room–Child: Parent room coordinate penetrating a child keep-out sub-box.
      3. Child–Child: 3D AABB intersection between non-shared child keep-out sub-boxes.
      4. Corridor–Room: Corridor interior points penetrating a third-party parent room.
      5. Corridor–Child: Corridor interior segment penetrating a sibling child keep-out sub-box.
      6. Corridor–Corridor: Non-incident orthogonal corridor segments crossing in 3D.
      7. Doorway: Boundary doorway vector obstructed by non-incident child sub-box or room.

    Args:
        formulation: ClusterFormulation containing inventory, graph, and model metadata.
        extraction: SolvedClusterExtraction containing solved parent/port coords and child offsets.

    Returns:
        List of ClearanceViolation objects representing the detected geometric collisions.
    """
    violations: list[ClearanceViolation] = []
    inventory = formulation.inventory
    rooms = tuple(
        sorted(set(inventory.rooms) | set(inventory.boundary_ports))
    )
    parent_coords: dict[int, tuple[int, int, int]] = {}
    parent_coords.update(extraction.raw_parent_coords)
    parent_coords.update(extraction.raw_port_coords)

    children = inventory.children
    child_solutions = inventory.child_solutions
    child_sub_boxes = inventory.child_sub_boxes
    gateway_anchors = inventory.gateway_anchors
    active_edges = inventory.active_edges
    b_ports = inventory.boundary_ports

    # Category 1: Room-Room Collisions
    for i in range(len(rooms)):
        u = rooms[i]
        coord_u = parent_coords.get(u)
        if coord_u is None:
            continue
        for j in range(i + 1, len(rooms)):
            v = rooms[j]
            coord_v = parent_coords.get(v)
            if coord_v is None:
                continue
            if coord_u == coord_v:
                violations.append(
                    ClearanceViolation(
                        category="parent_room",
                        item_a=u,
                        item_b=v,
                        description=(
                            f"Parent room/port {u} and {v} collide at coordinate {coord_u}."
                        ),
                    )
                )

    # Category 2: Room-Child Collisions
    for child in children:
        cid = child.node_id
        c_sol = child_solutions.get(cid)
        sb = child_sub_boxes.get(cid, ())
        if not c_sol or not sb or cid not in extraction.raw_child_offsets:
            continue
        anchors = gateway_anchors.get(cid, frozenset())
        c_ports = getattr(c_sol, "port_coordinates", {})
        for u in rooms:
            if u in anchors or u in c_sol.local_coordinates or u in c_ports:
                continue
            u_coord = parent_coords.get(u)
            if u_coord is None:
                continue
            ux, uy, uz = u_coord
            for k in range(len(sb)):
                min_x, max_x, min_y, max_y, min_z, max_z = _extract_child_sub_box_bounds(
                    (cid, k), extraction, inventory
                )
                if min_x <= ux <= max_x and min_y <= uy <= max_y and min_z <= uz <= max_z:
                    violations.append(
                        ClearanceViolation(
                            category="room_child",
                            item_a=u,
                            item_b=(cid, k),
                            description=(
                                f"Room {u} at {u_coord} penetrates child '{cid}' sub-box {k} "
                                f"bounds ({min_x}..{max_x}, {min_y}..{max_y}, {min_z}..{max_z})."
                            ),
                        )
                    )

    # Category 3: Child-Child Collisions
    equated = inventory.equated_children
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            c1 = children[i]
            c2 = children[j]
            if c1.node_id == c2.node_id:
                continue
            if frozenset({c1.node_id, c2.node_id}) in equated:
                continue
            sol1 = child_solutions.get(c1.node_id)
            sol2 = child_solutions.get(c2.node_id)
            sb1 = child_sub_boxes.get(c1.node_id, ())
            sb2 = child_sub_boxes.get(c2.node_id, ())
            if not sol1 or not sol2 or not sb1 or not sb2:
                continue
            if (
                c1.node_id not in extraction.raw_child_offsets
                or c2.node_id not in extraction.raw_child_offsets
            ):
                continue
            for k1 in range(len(sb1)):
                box1 = _extract_child_sub_box_bounds(
                    (c1.node_id, k1), extraction, inventory
                )
                for k2 in range(len(sb2)):
                    box2 = _extract_child_sub_box_bounds(
                        (c2.node_id, k2), extraction, inventory
                    )
                    if (
                        max(box1[0], box2[0]) <= min(box1[1], box2[1])
                        and max(box1[2], box2[2]) <= min(box1[3], box2[3])
                        and max(box1[4], box2[4]) <= min(box1[5], box2[5])
                    ):
                        violations.append(
                            ClearanceViolation(
                                category="child_child",
                                item_a=(c1.node_id, k1),
                                item_b=(c2.node_id, k2),
                                description=(
                                    f"Child '{c1.node_id}' sub-box {k1} overlaps "
                                    f"child '{c2.node_id}' sub-box {k2}."
                                ),
                            )
                        )

    # Category 4: Corridor-Room Collisions
    for i, ae in enumerate(active_edges):
        if sum(1 for d in ae.direction if d != 0) != 1:
            continue
        try:
            p1 = _extract_entity_coord(ae.src, extraction, inventory)
            p2 = _extract_entity_coord(ae.dst, extraction, inventory)
        except KeyError:
            continue
        seg = Segment3D(Coord3D(*p1), Coord3D(*p2))
        for w in rooms:
            if w in ae.associated_rooms:
                continue
            if any(
                w in child_solutions[cid].local_coordinates
                or w in getattr(child_solutions[cid], "port_coordinates", {})
                for cid in ae.associated_children
                if cid in child_solutions
            ):
                continue
            w_coord = parent_coords.get(w)
            if w_coord is None:
                continue
            if segment_room_penetration(seg, Coord3D(*w_coord)):
                violations.append(
                    ClearanceViolation(
                        category="corridor_room",
                        item_a=i,
                        item_b=w,
                        description=(
                            f"Corridor '{ae.edge_id}' interior penetrates parent room {w} at {w_coord}."
                        ),
                    )
                )

    # Category 5: Corridor-Child Collisions
    for i, ae in enumerate(active_edges):
        if ae.edge_id.startswith("sp_"):
            continue
        if sum(1 for d in ae.direction if d != 0) != 1:
            continue
        try:
            p1 = _extract_entity_coord(ae.src, extraction, inventory)
            p2 = _extract_entity_coord(ae.dst, extraction, inventory)
        except KeyError:
            continue
        dx, dy, dz = ae.direction
        if dx > 0:
            c_min_x, c_max_x = p1[0] + 1, p2[0] - 1
            c_min_y, c_max_y = p1[1], p1[1]
            c_min_z, c_max_z = p1[2], p1[2]
        elif dx < 0:
            c_min_x, c_max_x = p2[0] + 1, p1[0] - 1
            c_min_y, c_max_y = p1[1], p1[1]
            c_min_z, c_max_z = p1[2], p1[2]
        elif dy > 0:
            c_min_x, c_max_x = p1[0], p1[0]
            c_min_y, c_max_y = p1[1] + 1, p2[1] - 1
            c_min_z, c_max_z = p1[2], p1[2]
        elif dy < 0:
            c_min_x, c_max_x = p1[0], p1[0]
            c_min_y, c_max_y = p2[1] + 1, p1[1] - 1
            c_min_z, c_max_z = p1[2], p1[2]
        elif dz > 0:
            c_min_x, c_max_x = p1[0], p1[0]
            c_min_y, c_max_y = p1[1], p1[1]
            c_min_z, c_max_z = p1[2] + 1, p2[2] - 1
        else:
            c_min_x, c_max_x = p1[0], p1[0]
            c_min_y, c_max_y = p1[1], p1[1]
            c_min_z, c_max_z = p2[2] + 1, p1[2] - 1

        if c_min_x > c_max_x or c_min_y > c_max_y or c_min_z > c_max_z:
            continue

        for s_child in children:
            s_id = s_child.node_id
            s_sol = child_solutions.get(s_id)
            sb = child_sub_boxes.get(s_id, ())
            if not s_sol or not sb or s_id not in extraction.raw_child_offsets:
                continue

            s_ports = getattr(s_sol, "port_coordinates", {})
            s_child_rooms = set(s_sol.local_coordinates.keys()) | set(s_ports.keys())
            if (
                (
                    isinstance(ae.src, int)
                    and ae.src in b_ports
                    and (
                        ae.dst in s_child_rooms
                        or (isinstance(ae.dst, tuple) and ae.dst[0] == s_id)
                    )
                )
                or (
                    isinstance(ae.dst, int)
                    and ae.dst in b_ports
                    and (
                        ae.src in s_child_rooms
                        or (isinstance(ae.src, tuple) and ae.src[0] == s_id)
                    )
                )
            ):
                continue

            exempt_boxes = get_incident_child_sub_box_indices(
                s_id, s_sol, sb, ae, child_solutions
            )

            for k in range(len(sb)):
                if k in exempt_boxes:
                    continue
                box = _extract_child_sub_box_bounds(
                    (s_id, k), extraction, inventory
                )
                if (
                    max(c_min_x, box[0]) <= min(c_max_x, box[1])
                    and max(c_min_y, box[2]) <= min(c_max_y, box[3])
                    and max(c_min_z, box[4]) <= min(c_max_z, box[5])
                ):
                    violations.append(
                        ClearanceViolation(
                            category="corridor_child",
                            item_a=i,
                            item_b=(s_id, k),
                            description=(
                                f"Corridor '{ae.edge_id}' interior intersects sibling child '{s_id}' sub-box {k}."
                            ),
                        )
                    )

    # Category 6: Corridor-Corridor Collisions
    for i1 in range(len(active_edges)):
        ae1 = active_edges[i1]
        if ae1.edge_id.startswith("sp_"):
            continue
        if sum(1 for d in ae1.direction if d != 0) != 1:
            continue
        act1 = (
            "x" if ae1.direction[0] != 0 else ("y" if ae1.direction[1] != 0 else "z")
        )
        try:
            p1_a = _extract_entity_coord(ae1.src, extraction, inventory)
            p1_b = _extract_entity_coord(ae1.dst, extraction, inventory)
        except KeyError:
            continue
        seg1 = Segment3D(Coord3D(*p1_a), Coord3D(*p1_b))

        for i2 in range(i1 + 1, len(active_edges)):
            ae2 = active_edges[i2]
            if ae2.edge_id.startswith("sp_"):
                continue
            if sum(1 for d in ae2.direction if d != 0) != 1:
                continue
            act2 = (
                "x"
                if ae2.direction[0] != 0
                else ("y" if ae2.direction[1] != 0 else "z")
            )
            if act1 == act2:
                continue
            incident = bool(
                ae1.associated_rooms & ae2.associated_rooms
            ) or bool({ae1.src, ae1.dst} & {ae2.src, ae2.dst})
            if incident:
                continue
            try:
                p2_a = _extract_entity_coord(ae2.src, extraction, inventory)
                p2_b = _extract_entity_coord(ae2.dst, extraction, inventory)
            except KeyError:
                continue
            seg2 = Segment3D(Coord3D(*p2_a), Coord3D(*p2_b))
            if segments_intersect_3d(seg1, seg2):
                violations.append(
                    ClearanceViolation(
                        category="corridor_crossing",
                        item_a=i1,
                        item_b=i2,
                        description=(
                            f"Corridors '{ae1.edge_id}' and '{ae2.edge_id}' cross in 3D."
                        ),
                    )
                )

    # Category 7: Doorway Clearance
    node = getattr(inventory, "node", None)
    b_ports = inventory.boundary_ports
    graph = inventory.graph

    if node and b_ports and graph:
        sub_rooms: set[int] = set(node.rooms)

        def _collect_subtree(n: SolveTreeNode) -> None:
            sub_rooms.update(n.rooms)
            for ch in n.children:
                _collect_subtree(ch)

        for ch in node.children:
            _collect_subtree(ch)

        active_dirs: dict[int, set[tuple[int, int, int]]] = collections.defaultdict(set)
        for ae in active_edges:
            if isinstance(ae.src, int):
                active_dirs[ae.src].add(ae.direction)
            if isinstance(ae.dst, int):
                rev = (-ae.direction[0], -ae.direction[1], -ae.direction[2])
                active_dirs[ae.dst].add(rev)

        all_parent_rooms = set(rooms)
        for u in b_ports:
            if u not in all_parent_rooms or u not in graph.rooms:
                continue
            u_pt = parent_coords.get(u)
            if u_pt is None:
                continue
            for ex in graph.rooms[u].exits:
                if ex.dst_vnum not in sub_rooms:
                    try:
                        disp = direction_to_displacement(ex.direction)
                    except ValueError:
                        continue
                    if disp in active_dirs[u]:
                        continue
                    d_key = (u, disp)
                    door_pt = (u_pt[0] + disp[0], u_pt[1] + disp[1], u_pt[2] + disp[2])

                    # Check against child sub-boxes
                    for child in children:
                        cid = child.node_id
                        c_sol = child_solutions.get(cid)
                        sb = child_sub_boxes.get(cid, ())
                        if (
                            not c_sol
                            or not sb
                            or cid not in extraction.raw_child_offsets
                        ):
                            continue
                        for k in range(len(sb)):
                            box = _extract_child_sub_box_bounds(
                                (cid, k), extraction, inventory
                            )
                            if (
                                box[0] <= door_pt[0] <= box[1]
                                and box[2] <= door_pt[1] <= box[3]
                                and box[4] <= door_pt[2] <= box[5]
                            ):
                                violations.append(
                                    ClearanceViolation(
                                        category="doorway_child",
                                        item_a=d_key,
                                        item_b=(cid, k),
                                        description=(
                                            f"Doorway from port {u} in direction {disp} at {door_pt} "
                                            f"obstructed by child '{cid}' sub-box {k}."
                                        ),
                                    )
                                )

                    # Check against third-party parent rooms
                    for w in rooms:
                        if w == u:
                            continue
                        w_pt = parent_coords.get(w)
                        if w_pt == door_pt:
                            violations.append(
                                ClearanceViolation(
                                    category="doorway_room",
                                    item_a=d_key,
                                    item_b=w,
                                    description=(
                                        f"Doorway from port {u} in direction {disp} at {door_pt} "
                                        f"obstructed by parent room {w}."
                                    ),
                                )
                            )

    return violations


def apply_clearance_constraints(formulation: ClusterFormulation) -> None:
    """Add disjunctive Big-M clearance and non-overlap constraints to the cluster model.

    Delegates to formulation.apply_clearances() for backward compatibility.

    Args:
        formulation: ClusterFormulation containing the Pyomo model and inventory.
    """
    formulation.apply_clearances()


__all__ = [
    "ClearanceConstraintBuilder",
    "ClearanceViolation",
    "VIOLATION_TIER_PRIORITY",
    "_extract_child_sub_box_bounds",
    "_extract_entity_coord",
    "apply_clearance_constraints",
    "detect_cluster_collisions",
    "get_incident_child_sub_box_indices",
]
