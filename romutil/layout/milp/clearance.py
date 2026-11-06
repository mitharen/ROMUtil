"""Stage 3: Geometric clearance and separation constraints for layout cluster MILP."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
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
    BoundaryExit,
    EndpointRef,
    direction_to_displacement,
    get_child_coordinate,
)
from romutil.layout.milp.spatial_dag import PositionalConstraintDAG
from romutil.layout.hierarchy.spqr import SplitPairCut
from romutil.types import Coord3D

logger = logging.getLogger(__name__)

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
    "corridor_room": 2,
    "corridor_child": 2,
    "corridor_crossing": 3,
}


class ClearanceConstraintBuilder:
    """Builder enforcing disjunctive Big-M geometric clearance and separation constraints."""

    def __init__(self, formulation: ClusterFormulation) -> None:
        self.formulation = formulation
        self.separated_pairs: set[tuple[str, Any, Any]] = set()
        self.spatial_dag = PositionalConstraintDAG.from_formulation(formulation)
        self.unsatisfiable_collisions: list[ClearanceViolation] = []

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

        child_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1 = children[i]
                c2 = children[j]
                if c1.node_id == c2.node_id:
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
                    for u in rooms:
                        if u not in c_sol.local_coordinates:
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

    def get_corridor_interior_box(self, item: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for active edge or boundary exit interior segment."""
        if isinstance(item, int):
            return self.formulation.get_corridor_interior_box(
                self.formulation.inventory.active_edges[item]
            )
        elif isinstance(item, (tuple, list)) and len(item) == 2 and item[0] == "bexit":
            return self.formulation.get_boundary_exit_interior_box(
                self.formulation.inventory.boundary_exits[item[1]]
            )
        raise ValueError(f"Unknown corridor item reference: {item}")

    def get_corridor_full_box(self, item: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for active edge or boundary exit entire segment."""
        if isinstance(item, int):
            return self.formulation.get_corridor_full_box(
                self.formulation.inventory.active_edges[item]
            )
        elif isinstance(item, (tuple, list)) and len(item) == 2 and item[0] == "bexit":
            return self.formulation.get_boundary_exit_full_box(
                self.formulation.inventory.boundary_exits[item[1]]
            )
        raise ValueError(f"Unknown corridor item reference: {item}")

    def add_corridor_room_separation(self) -> None:
        """Enforce corridor segment non-penetration against third-party parent rooms."""
        rooms = tuple(sorted(self.formulation.inventory.rooms))
        active_edges = self.formulation.inventory.active_edges
        boundary_exits = getattr(self.formulation.inventory, "boundary_exits", ())
        child_solutions = self.formulation.inventory.child_solutions
        edge_indices = list(range(len(active_edges)))

        corridor_room_pairs: list[tuple[Any, int]] = []
        if rooms:
            if edge_indices:
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

            for j, be in enumerate(boundary_exits):
                if sum(1 for d in be.direction if d != 0) != 1:
                    continue
                for w in rooms:
                    if w == be.port:
                        continue
                    pair = (("bexit", j), w)
                    corridor_room_pairs.append(pair)
                    self.separated_pairs.add(("corridor_room", pair[0], pair[1]))

        if corridor_room_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=corridor_room_pairs,
                get_box_a=self.get_corridor_interior_box,
                get_box_b=self.formulation.get_room_box,
                prefix="corridor_room",
                m_big=self.formulation.m_big,
            )

    def add_corridor_child_separation(self) -> None:
        """Enforce corridor segment non-penetration against sibling child sub-boxes."""
        active_edges = self.formulation.inventory.active_edges
        boundary_exits = getattr(self.formulation.inventory, "boundary_exits", ())
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        child_sub_boxes = self.formulation.inventory.child_sub_boxes
        edge_indices = list(range(len(active_edges)))

        corridor_child_pairs: list[tuple[Any, tuple[str, int]]] = []
        if len(children) > 0:
            if edge_indices:
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

                        for k in range(len(sb)):
                            pair = (i, (s_id, k))
                            corridor_child_pairs.append(pair)
                            self.separated_pairs.add(("corridor_child", pair[0], pair[1]))

            for j, be in enumerate(boundary_exits):
                if sum(1 for d in be.direction if d != 0) != 1:
                    continue
                for s_child in children:
                    s_id = s_child.node_id
                    s_sol = child_solutions.get(s_id)
                    sb = child_sub_boxes.get(s_id, ())
                    if not s_sol or not sb:
                        continue
                    if be.port in s_sol.local_coordinates or be.port in getattr(s_sol, "port_coordinates", {}):
                        continue

                    for k in range(len(sb)):
                        pair = (("bexit", j), (s_id, k))
                        corridor_child_pairs.append(pair)
                        self.separated_pairs.add(("corridor_child", pair[0], pair[1]))

        if corridor_child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=corridor_child_pairs,
                get_box_a=self.get_corridor_interior_box,
                get_box_b=self.formulation.get_child_sub_box,
                prefix="corridor_child",
                m_big=self.formulation.m_big,
            )

    def add_corridor_crossing_separation(self) -> None:
        """Enforce corridor crossing separation between non-incident orthogonal corridor segments."""
        active_edges = self.formulation.inventory.active_edges
        boundary_exits = getattr(self.formulation.inventory, "boundary_exits", ())

        crossing_pairs: list[tuple[Any, Any]] = []
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
                        ("corridor_crossing", pair[0], pair[1])
                    )

            for j in range(len(boundary_exits)):
                be = boundary_exits[j]
                if sum(1 for d in be.direction if d != 0) != 1:
                    continue
                act_be = (
                    "x"
                    if be.direction[0] != 0
                    else ("y" if be.direction[1] != 0 else "z")
                )
                if act1 == act_be:
                    continue
                incident = (
                    be.port in ae1.associated_rooms
                    or be.port in (ae1.src, ae1.dst)
                )
                if not incident:
                    pair = (i1, ("bexit", j))
                    crossing_pairs.append(pair)
                    self.separated_pairs.add(
                        ("corridor_crossing", pair[0], pair[1])
                    )

        for j1 in range(len(boundary_exits)):
            be1 = boundary_exits[j1]
            if sum(1 for d in be1.direction if d != 0) != 1:
                continue
            act1 = (
                "x"
                if be1.direction[0] != 0
                else ("y" if be1.direction[1] != 0 else "z")
            )
            for j2 in range(j1 + 1, len(boundary_exits)):
                be2 = boundary_exits[j2]
                if sum(1 for d in be2.direction if d != 0) != 1:
                    continue
                act2 = (
                    "x"
                    if be2.direction[0] != 0
                    else ("y" if be2.direction[1] != 0 else "z")
                )
                if act1 == act2:
                    continue
                if be1.port == be2.port:
                    continue
                pair = (("bexit", j1), ("bexit", j2))
                crossing_pairs.append(pair)
                self.separated_pairs.add(
                    ("corridor_crossing", pair[0], pair[1])
                )

        if crossing_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=crossing_pairs,
                get_box_a=self.get_corridor_full_box,
                get_box_b=self.get_corridor_full_box,
                prefix="corridor_crossing",
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

        Filters candidate separation directions through the positional constraint DAG to
        prune contradicted half-spaces. If all separation directions for a candidate collision
        are contradicted by active edges or topological alignments, the collision is classified
        as an unresolvable geometric deadlock (UNSATISFIABLE_COLLISION) and impossible cuts are omitted.

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

        candidates: list[tuple[str, tuple[str, Any, Any], tuple[Any, Any], ClearanceViolation]] = []
        seen: set[tuple[str, Any, Any]] = set()

        for viol in violations:
            cat = viol.category
            a = viol.item_a
            b = viol.item_b

            if cat in ("parent_room", "child_child", "corridor_crossing"):
                key_a = (0, a) if isinstance(a, int) else (1, a) if isinstance(a, tuple) else (2, str(a))
                key_b = (0, b) if isinstance(b, int) else (1, b) if isinstance(b, tuple) else (2, str(b))
                first, second = (a, b) if key_a <= key_b else (b, a)
                norm_key = (cat, first, second)
                pair = (first, second)
            else:
                norm_key = (cat, a, b)
                pair = (a, b)

            if norm_key in self.separated_pairs or norm_key in seen:
                continue
            seen.add(norm_key)
            candidates.append((cat, norm_key, pair, viol))

        if not candidates:
            return 0

        # Sort candidate un-separated violations according to priority tier (stable sort)
        candidates.sort(key=lambda item: VIOLATION_TIER_PRIORITY.get(item[0], 5))

        # Filter candidate separation directions with the PositionalConstraintDAG
        inventory = getattr(self.formulation, "inventory", None)
        resolvable: list[
            tuple[str, tuple[str, Any, Any], tuple[Any, Any], list[str]]
        ] = []
        for cat, norm_key, pair, viol in candidates:
            item_a, item_b = pair
            valid_dirs, pruned_reasons = self.spatial_dag.filter_candidate_directions(
                item_a, item_b, cat, inventory
            )
            if not valid_dirs:
                # All separation directions contradicted: mark as unresolvable geometric deadlock
                self.unsatisfiable_collisions.append(viol)
                self.separated_pairs.add(norm_key)
                logger.warning(
                    "Detected unresolvable geometric deadlock (UNSATISFIABLE_COLLISION) between %s and %s in category '%s'. "
                    "All 6 separation directions contradicted by positional constraint DAG: %s",
                    item_a,
                    item_b,
                    cat,
                    pruned_reasons,
                )
                continue

            if pruned_reasons:
                logger.debug(
                    "Pruned %d contradicted separation directions for (%s, %s) in category '%s': %s",
                    len(pruned_reasons),
                    item_a,
                    item_b,
                    cat,
                    pruned_reasons,
                )

            resolvable.append((cat, norm_key, pair, valid_dirs))

        if not resolvable:
            return 0

        # Limit newly injected cuts to at most max_cuts_per_pass
        selected = resolvable[:budget]

        new_pairs_by_cat: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
        allowed_dirs_by_cat: dict[str, dict[tuple[Any, Any], list[str]]] = collections.defaultdict(dict)

        for cat, norm_key, pair, valid_dirs in selected:
            self.separated_pairs.add(norm_key)
            new_pairs_by_cat[cat].append(pair)
            allowed_dirs_by_cat[cat][pair] = valid_dirs

        for cat, pairs in new_pairs_by_cat.items():
            allowed = allowed_dirs_by_cat[cat]
            if cat == "parent_room":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.formulation.get_room_box,
                    get_box_b=self.formulation.get_room_box,
                    prefix="parent_room",
                    m_big=self.formulation.m_big,
                    allowed_directions=allowed,
                )
            elif cat == "room_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.formulation.get_room_box,
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="room_child",
                    m_big=self.formulation.m_big,
                    allowed_directions=allowed,
                )
            elif cat == "child_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.formulation.get_child_sub_box,
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="child_child",
                    m_big=self.formulation.m_big,
                    allowed_directions=allowed,
                )
            elif cat == "corridor_room":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.get_corridor_interior_box,
                    get_box_b=self.formulation.get_room_box,
                    prefix="corridor_room",
                    m_big=self.formulation.m_big,
                    allowed_directions=allowed,
                )
            elif cat == "corridor_child":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.get_corridor_interior_box,
                    get_box_b=self.formulation.get_child_sub_box,
                    prefix="corridor_child",
                    m_big=self.formulation.m_big,
                    allowed_directions=allowed,
                )
            elif cat == "corridor_crossing":
                add_disjunctive_box_separation(
                    model=self.formulation.model,
                    pairs=pairs,
                    get_box_a=self.get_corridor_full_box,
                    get_box_b=self.get_corridor_full_box,
                    prefix="corridor_crossing",
                    m_big=self.formulation.m_big,
                    allowed_directions=allowed,
                )

        return len(selected)

    def build(self) -> None:
        """Apply all disjunctive Big-M geometric clearance constraints."""
        self.add_parent_room_separation()
        self.add_child_child_separation()
        self.add_room_child_separation()
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
    boundary_exits = getattr(inventory, "boundary_exits", ())
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
        for u in rooms:
            if u in c_sol.local_coordinates:
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
                                f"Room {u} at {u_coord} overlaps child '{cid}' sub-box {k} "
                                f"bounds ({min_x}..{max_x}, {min_y}..{max_y}, {min_z}..{max_z})."
                            ),
                        )
                    )

    # Category 3: Child-Child Collisions
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            c1 = children[i]
            c2 = children[j]
            if c1.node_id == c2.node_id:
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
    parent_rooms = tuple(sorted(inventory.rooms))
    for i, ae in enumerate(active_edges):
        if sum(1 for d in ae.direction if d != 0) != 1:
            continue
        try:
            p1 = _extract_entity_coord(ae.src, extraction, inventory)
            p2 = _extract_entity_coord(ae.dst, extraction, inventory)
        except KeyError:
            continue
        seg = Segment3D(Coord3D(*p1), Coord3D(*p2))
        for w in parent_rooms:
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
                            f"Corridor '{ae.edge_id}' interior overlaps parent room {w} at {w_coord}."
                        ),
                    )
                )

    for j, be in enumerate(boundary_exits):
        if sum(1 for d in be.direction if d != 0) != 1:
            continue
        try:
            p1 = _extract_entity_coord(be.port, extraction, inventory)
        except KeyError:
            continue
        dx, dy, dz = be.direction
        length = be.length
        p2 = (p1[0] + dx * length, p1[1] + dy * length, p1[2] + dz * length)
        b_min_x, b_max_x = min(p1[0] + dx, p2[0]), max(p1[0] + dx, p2[0])
        b_min_y, b_max_y = min(p1[1] + dy, p2[1]), max(p1[1] + dy, p2[1])
        b_min_z, b_max_z = min(p1[2] + dz, p2[2]), max(p1[2] + dz, p2[2])

        for w in parent_rooms:
            if w == be.port:
                continue
            w_coord = parent_coords.get(w)
            if w_coord is None:
                continue
            if (
                b_min_x <= w_coord[0] <= b_max_x
                and b_min_y <= w_coord[1] <= b_max_y
                and b_min_z <= w_coord[2] <= b_max_z
            ):
                violations.append(
                    ClearanceViolation(
                        category="corridor_room",
                        item_a=("bexit", j),
                        item_b=w,
                        description=(
                            f"Boundary exit '{be.edge_id}' interior overlaps parent room {w} at {w_coord}."
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

            for k in range(len(sb)):
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

    for j, be in enumerate(boundary_exits):
        if sum(1 for d in be.direction if d != 0) != 1:
            continue
        try:
            p1 = _extract_entity_coord(be.port, extraction, inventory)
        except KeyError:
            continue
        dx, dy, dz = be.direction
        length = be.length
        p2 = (p1[0] + dx * length, p1[1] + dy * length, p1[2] + dz * length)
        c_min_x, c_max_x = min(p1[0] + dx, p2[0]), max(p1[0] + dx, p2[0])
        c_min_y, c_max_y = min(p1[1] + dy, p2[1]), max(p1[1] + dy, p2[1])
        c_min_z, c_max_z = min(p1[2] + dz, p2[2]), max(p1[2] + dz, p2[2])

        for s_child in children:
            s_id = s_child.node_id
            s_sol = child_solutions.get(s_id)
            sb = child_sub_boxes.get(s_id, ())
            if not s_sol or not sb or s_id not in extraction.raw_child_offsets:
                continue
            if be.port in s_sol.local_coordinates or be.port in getattr(s_sol, "port_coordinates", {}):
                continue

            for k in range(len(sb)):
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
                            item_a=("bexit", j),
                            item_b=(s_id, k),
                            description=(
                                f"Boundary exit '{be.edge_id}' interior intersects sibling child '{s_id}' sub-box {k}."
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

        for j in range(len(boundary_exits)):
            be = boundary_exits[j]
            if sum(1 for d in be.direction if d != 0) != 1:
                continue
            act_be = (
                "x"
                if be.direction[0] != 0
                else ("y" if be.direction[1] != 0 else "z")
            )
            if act1 == act_be:
                continue
            incident = (
                be.port in ae1.associated_rooms
                or be.port in (ae1.src, ae1.dst)
            )
            if incident:
                continue
            try:
                p_be_1 = _extract_entity_coord(be.port, extraction, inventory)
            except KeyError:
                continue
            p_be_2 = (
                p_be_1[0] + be.direction[0] * be.length,
                p_be_1[1] + be.direction[1] * be.length,
                p_be_1[2] + be.direction[2] * be.length,
            )
            seg_be = Segment3D(Coord3D(*p_be_1), Coord3D(*p_be_2))
            if segments_intersect_3d(seg1, seg_be):
                violations.append(
                    ClearanceViolation(
                        category="corridor_crossing",
                        item_a=i1,
                        item_b=("bexit", j),
                        description=(
                            f"Corridor '{ae1.edge_id}' and boundary exit '{be.edge_id}' cross in 3D."
                        ),
                    )
                )

    for j1 in range(len(boundary_exits)):
        be1 = boundary_exits[j1]
        if sum(1 for d in be1.direction if d != 0) != 1:
            continue
        act1 = (
            "x"
            if be1.direction[0] != 0
            else ("y" if be1.direction[1] != 0 else "z")
        )
        try:
            p1_a = _extract_entity_coord(be1.port, extraction, inventory)
        except KeyError:
            continue
        p1_b = (
            p1_a[0] + be1.direction[0] * be1.length,
            p1_a[1] + be1.direction[1] * be1.length,
            p1_a[2] + be1.direction[2] * be1.length,
        )
        seg1 = Segment3D(Coord3D(*p1_a), Coord3D(*p1_b))

        for j2 in range(j1 + 1, len(boundary_exits)):
            be2 = boundary_exits[j2]
            if sum(1 for d in be2.direction if d != 0) != 1:
                continue
            act2 = (
                "x"
                if be2.direction[0] != 0
                else ("y" if be2.direction[1] != 0 else "z")
            )
            if act1 == act2:
                continue
            if be1.port == be2.port:
                continue
            try:
                p2_a = _extract_entity_coord(be2.port, extraction, inventory)
            except KeyError:
                continue
            p2_b = (
                p2_a[0] + be2.direction[0] * be2.length,
                p2_a[1] + be2.direction[1] * be2.length,
                p2_a[2] + be2.direction[2] * be2.length,
            )
            seg2 = Segment3D(Coord3D(*p2_a), Coord3D(*p2_b))
            if segments_intersect_3d(seg1, seg2):
                violations.append(
                    ClearanceViolation(
                        category="corridor_crossing",
                        item_a=("bexit", j1),
                        item_b=("bexit", j2),
                        description=(
                            f"Boundary exits '{be1.edge_id}' and '{be2.edge_id}' cross in 3D."
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
]
