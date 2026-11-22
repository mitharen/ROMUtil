"""Stage 3: Geometric clearance and separation constraints for layout cluster MILP."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from romutil.layout.geometry import (
    BoundingBox3D,
    Segment3D,
    segment_room_penetration,
    segments_intersect_3d,
)
from romutil.layout.milp.clearance_boxes import (
    ExactCollisionCutGrower,
    GrownClearanceCut,
    extract_entity_coord as _extract_entity_coord,
)
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.inventory import (
    ActiveEdge,
    direction_to_displacement,
    get_child_coordinate,
)
from romutil.layout.milp.spatial_dag import ClusterPositionalDAG
from romutil.layout.hierarchy.spqr import SplitPairCut
from romutil.types import Coord3D

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from romutil.layout.ir import NodeSolution, SolveTreeNode
    from romutil.layout.milp.formulation import ClusterFormulation
    from romutil.layout.milp.inventory import ClusterInventory
    from romutil.layout.milp.solver import SolvedClusterExtraction


@dataclass(frozen=True)
class RoomClearanceBox:
    """Clearance box representing a local parent room or boundary port."""

    vnum: int


@dataclass(frozen=True)
class ChildSubClearanceBox:
    """Clearance box representing a child cluster keep-out sub-box."""

    child_id: str
    box_index: int


@dataclass(frozen=True)
class CorridorClearanceBox:
    """Clearance box representing an active corridor edge segment."""

    edge_index: int
    edge_id: str = ""
    kind: str = "internal"


ClearanceEntity = (
    RoomClearanceBox
    | ChildSubClearanceBox
    | CorridorClearanceBox
)


@dataclass(frozen=True)
class ClearanceViolation:
    """Detected geometric collision or clearance violation between cluster layout entities."""

    category: str
    item_a: Any
    item_b: Any
    description: str = ""
    overlap_volume: int = 1


VIOLATION_TIER_PRIORITY: Mapping[str, int] = {
    "child_child": 0,
    "room_child": 1,
    "parent_room": 2,
    "corridor_child": 3,
    "corridor_room": 4,
    "corridor_crossing": 5,
}


@dataclass(frozen=True)
class _PendingCut:
    """Unified cut candidate for tier-budgeted Big-M constraint injection."""

    category: str
    norm_key: tuple[Any, ...]
    pair: tuple[Any, Any]
    valid_dirs: tuple[str, ...]
    grown: GrownClearanceCut | None = None


def _item_sort_key(it: Any) -> tuple[int, Any]:
    """Return a deterministic ordering key for symmetric collision pair canonicalization."""
    if isinstance(it, int):
        return (0, it)
    if isinstance(it, RoomClearanceBox):
        return (0, it.vnum)
    if isinstance(it, ChildSubClearanceBox):
        return (1, (it.child_id, it.box_index))
    if isinstance(it, CorridorClearanceBox):
        return (2, (0, it.edge_index))
    if isinstance(it, tuple):
        return (3, it)
    return (4, str(it))


def _normalize_violation_pair(
    viol: ClearanceViolation,
) -> tuple[str, tuple[Any, ...], tuple[Any, Any]]:
    """Return canonical (category, norm_key, pair) for deduplication."""
    cat = viol.category
    a = viol.item_a
    b = viol.item_b
    if cat in ("parent_room", "child_child", "corridor_crossing"):
        first, second = (a, b) if _item_sort_key(a) <= _item_sort_key(b) else (b, a)
        return cat, (cat, first, second), (first, second)
    return cat, (cat, a, b), (a, b)


def _macro_entity_key(it: Any) -> tuple[str, Any]:
    """Return macro-level entity key for per-pass deduplication and hashing."""
    if isinstance(it, int):
        return ("room", it)
    if isinstance(it, RoomClearanceBox):
        return ("room", it.vnum)
    if isinstance(it, ChildSubClearanceBox):
        return ("child", it.child_id)
    if isinstance(it, CorridorClearanceBox):
        return ("corridor", ("edge", it.edge_index))
    if (
        isinstance(it, tuple)
        and len(it) == 2
        and isinstance(it[0], str)
        and isinstance(it[1], int)
    ):
        if it[0] == "edge":
            return ("corridor", it)
        return ("child", it[0])
    return ("other", str(it))


class ClearanceConstraintBuilder:
    """Builder enforcing disjunctive Big-M geometric clearance and separation constraints."""

    def __init__(self, formulation: ClusterFormulation) -> None:
        self.formulation = formulation
        self.separated_pairs: set[tuple[Any, ...]] = set()
        self.spatial_dag = ClusterPositionalDAG.from_formulation(formulation)
        self.unsatisfiable_collisions: list[ClearanceViolation] = []
        self._cut_grower: ExactCollisionCutGrower | None = None

    def _get_cut_grower(self) -> ExactCollisionCutGrower | None:
        """Lazily initialize the exact collision cut grower when cluster inventory supports it."""
        if self._cut_grower is not None:
            return self._cut_grower
        inventory = getattr(self.formulation, "inventory", None)
        child_sub_boxes = getattr(inventory, "child_sub_boxes", None)
        active_edges = getattr(inventory, "active_edges", None)
        rooms = getattr(inventory, "rooms", None)
        if (
            inventory is None
            or not isinstance(child_sub_boxes, Mapping)
            or not isinstance(active_edges, (list, tuple))
            or not isinstance(rooms, (list, tuple))
        ):
            return None
        self._cut_grower = ExactCollisionCutGrower(inventory, self.spatial_dag)
        return self._cut_grower

    def _get_child_origin_box(self, cid: str) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return zero-width point box for child cid's local origin variable."""
        m = self.formulation.model
        return (m.cx[cid], m.cx[cid], m.cy[cid], m.cy[cid], m.cz[cid], m.cz[cid])

    def _get_child_offset_box(
        self, target: tuple[str, BoundingBox3D]
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return symbolic 3D box for child cid translated by a grown obstacle box."""
        m = self.formulation.model
        cid, box = target
        return (
            m.cx[cid] + box.min_x,
            m.cx[cid] + box.max_x,
            m.cy[cid] + box.min_y,
            m.cy[cid] + box.max_y,
            m.cz[cid] + box.min_z,
            m.cz[cid] + box.max_z,
        )

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
        """Return 3D bounding box for active edge interior segment."""
        if isinstance(item, CorridorClearanceBox):
            return self.formulation.get_corridor_interior_box(
                self.formulation.inventory.active_edges[item.edge_index]
            )
        elif isinstance(item, int):
            return self.formulation.get_corridor_interior_box(
                self.formulation.inventory.active_edges[item]
            )
        raise ValueError(f"Unknown corridor item reference: {item}")

    def get_corridor_room_box(self, item: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D interior box for corridor-vs-room separation, including transverse slack."""
        if isinstance(item, CorridorClearanceBox):
            ae = self.formulation.inventory.active_edges[item.edge_index]
        elif isinstance(item, int):
            ae = self.formulation.inventory.active_edges[item]
        else:
            raise ValueError(f"Unknown corridor item reference: {item}")

        if hasattr(type(self.formulation), "get_corridor_envelope_box"):
            return self.formulation.get_corridor_envelope_box(ae, interior=True)
        return self.formulation.get_corridor_interior_box(ae)

    def get_corridor_full_box(self, item: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return 3D bounding box for active edge entire segment."""
        if isinstance(item, CorridorClearanceBox):
            return self.formulation.get_corridor_full_box(
                self.formulation.inventory.active_edges[item.edge_index]
            )
        elif isinstance(item, int):
            return self.formulation.get_corridor_full_box(
                self.formulation.inventory.active_edges[item]
            )
        raise ValueError(f"Unknown corridor item reference: {item}")

    def add_corridor_room_separation(self) -> None:
        """Enforce corridor segment non-penetration against third-party parent rooms."""
        rooms = tuple(sorted(self.formulation.inventory.rooms))
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

        if corridor_room_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=corridor_room_pairs,
                get_box_a=self.get_corridor_room_box,
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

        corridor_child_pairs: list[tuple[int, tuple[str, int]]] = []
        if len(children) > 0 and edge_indices:
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

    def _resolve_group_cut(
        self,
        items: Sequence[ClearanceViolation],
        grower: ExactCollisionCutGrower | None,
        extraction: SolvedClusterExtraction | None,
        inventory: Any,
        batch_seen: set[tuple[Any, ...]],
    ) -> _PendingCut | None:
        """Pick the highest-penetration resolvable cut for one macro entity pair."""
        for viol in items:
            if grower is not None and extraction is not None:
                grown = grower.try_grow_cut(
                    viol.category, viol.item_a, viol.item_b, extraction
                )
                if (
                    grown is not None
                    and grown.norm_key not in self.separated_pairs
                    and grown.norm_key not in batch_seen
                ):
                    pair = (grown.entity_a, (grown.target_child_id, grown.obstacle_box))
                    return _PendingCut(
                        category=viol.category,
                        norm_key=grown.norm_key,
                        pair=pair,
                        valid_dirs=grown.valid_dirs,
                        grown=grown,
                    )

            cat, norm_key, pair = _normalize_violation_pair(viol)
            if norm_key in self.separated_pairs or norm_key in batch_seen:
                continue
            valid_dirs, pruned_reasons = self.spatial_dag.filter_candidate_directions(
                pair[0], pair[1], cat, inventory
            )
            if not valid_dirs:
                self.unsatisfiable_collisions.append(viol)
                self.separated_pairs.add(norm_key)
                logger.warning(
                    "Detected unresolvable geometric deadlock (UNSATISFIABLE_COLLISION) between %s and %s in category '%s'. "
                    "All 6 separation directions contradicted by positional constraint DAG: %s",
                    pair[0],
                    pair[1],
                    cat,
                    pruned_reasons,
                )
                continue

            if pruned_reasons:
                logger.debug(
                    "Pruned %d contradicted separation directions for (%s, %s) in category '%s': %s",
                    len(pruned_reasons),
                    pair[0],
                    pair[1],
                    cat,
                    pruned_reasons,
                )

            return _PendingCut(
                category=cat,
                norm_key=norm_key,
                pair=pair,
                valid_dirs=tuple(valid_dirs),
                grown=None,
            )
        return None

    def inject_clearance_cuts(
        self,
        violations: Sequence[ClearanceViolation],
        max_cuts_per_pass: int | None = None,
        *,
        extraction: SolvedClusterExtraction | None = None,
    ) -> int:
        """Inject disjunctive Big-M separation cuts for detected collision violations.

        Sorts candidate un-separated violations according to priority tier and limits
        newly injected cuts to at most `max_cuts_per_pass` (defaulting to
        `formulation.options.max_cuts_per_pass` or 20) per pass to prevent branch-and-bound
        explosion. When `extraction` is supplied, synthesizes exact collision-time
        configuration-space and hole-filled obstacle boxes when valid; otherwise uses
        the exact detected entity sub-boxes.

        Args:
            violations: Sequence of ClearanceViolation instances detected in the current extraction.
            max_cuts_per_pass: Optional override for the maximum number of cuts to inject in this pass.
            extraction: Optional current iteration extraction for collision-time cut growth.

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

        inventory = getattr(self.formulation, "inventory", None)
        grower = self._get_cut_grower() if extraction is not None else None

        macro_groups: dict[
            tuple[int, tuple[tuple[str, Any], tuple[str, Any]]],
            list[ClearanceViolation],
        ] = collections.defaultdict(list)
        for viol in violations:
            tier = VIOLATION_TIER_PRIORITY.get(viol.category, 5)
            m_a = _macro_entity_key(viol.item_a)
            m_b = _macro_entity_key(viol.item_b)
            m_pair = (m_a, m_b) if m_a <= m_b else (m_b, m_a)
            macro_groups[(tier, m_pair)].append(viol)

        pass_iteration = getattr(self, "_pass_iteration", 0)
        self._pass_iteration = pass_iteration + 1

        candidates_by_tier: dict[int, list[tuple[int, int, _PendingCut]]] = collections.defaultdict(list)
        batch_seen: set[tuple[Any, ...]] = set()
        for (tier, m_pair), group_viols in sorted(macro_groups.items(), key=lambda kv: kv[0][0]):
            h = int(
                hashlib.sha256(f"{m_pair}_{pass_iteration}".encode("utf-8")).hexdigest(),
                16,
            )
            group_viols.sort(key=lambda v: v.overlap_volume, reverse=True)
            pending = self._resolve_group_cut(
                group_viols, grower, extraction, inventory, batch_seen
            )
            if pending is not None:
                batch_seen.add(pending.norm_key)
                candidates_by_tier[tier].append((tier, h, pending))

        if not candidates_by_tier:
            return 0

        for tier_list in candidates_by_tier.values():
            tier_list.sort(key=lambda x: x[1])

        macro_tiers = (0, 1, 2)
        corridor_tiers = (3, 4, 5)
        has_macro = any(t in candidates_by_tier for t in macro_tiers)
        has_corridor = any(t in candidates_by_tier for t in corridor_tiers)

        selected: list[_PendingCut] = []
        if has_macro and has_corridor and budget >= 4:
            corridor_quota = max(1, budget // 3)
            macro_quota = budget - corridor_quota

            macro_pool = [cand for t in macro_tiers for cand in candidates_by_tier.get(t, ())]
            corridor_pool = [cand for t in corridor_tiers for cand in candidates_by_tier.get(t, ())]

            selected_macro = [x[2] for x in macro_pool[:macro_quota]]
            selected.extend(selected_macro)

            actual_corridor_budget = budget - len(selected_macro)
            selected.extend([x[2] for x in corridor_pool[:actual_corridor_budget]])

            if len(selected) < budget and len(macro_pool) > len(selected_macro):
                remainder = budget - len(selected)
                selected.extend(
                    [x[2] for x in macro_pool[len(selected_macro) : len(selected_macro) + remainder]]
                )
        else:
            all_cands = [cand for t in sorted(candidates_by_tier) for cand in candidates_by_tier[t]]
            selected = [x[2] for x in all_cands[:budget]]

        if not selected:
            return 0

        std_pairs_by_cat: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
        std_dirs_by_cat: dict[str, dict[tuple[Any, Any], list[str]]] = collections.defaultdict(dict)
        grown_pairs_by_kind: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
        grown_dirs_by_kind: dict[str, dict[tuple[Any, Any], list[str]]] = collections.defaultdict(dict)

        for cut in selected:
            self.separated_pairs.add(cut.norm_key)
            if cut.grown is not None:
                grown_pairs_by_kind[cut.grown.kind].append(cut.pair)
                grown_dirs_by_kind[cut.grown.kind][cut.pair] = list(cut.valid_dirs)
            else:
                std_pairs_by_cat[cut.category].append(cut.pair)
                std_dirs_by_cat[cut.category][cut.pair] = list(cut.valid_dirs)

        std_box_getters: Mapping[str, tuple[Any, Any]] = {
            "parent_room": (self.formulation.get_room_box, self.formulation.get_room_box),
            "room_child": (self.formulation.get_room_box, self.formulation.get_child_sub_box),
            "child_child": (self.formulation.get_child_sub_box, self.formulation.get_child_sub_box),
            "corridor_room": (self.get_corridor_room_box, self.formulation.get_room_box),
            "corridor_child": (self.get_corridor_interior_box, self.formulation.get_child_sub_box),
            "corridor_crossing": (self.get_corridor_full_box, self.get_corridor_full_box),
        }
        for cat, pairs in std_pairs_by_cat.items():
            getters = std_box_getters.get(cat)
            if getters is None:
                continue
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=pairs,
                get_box_a=getters[0],
                get_box_b=getters[1],
                prefix=cat,
                m_big=self.formulation.m_big,
                allowed_directions=std_dirs_by_cat[cat],
            )

        grown_box_getters: Mapping[str, tuple[Any, Any]] = {
            "mink_child_child": (self._get_child_origin_box, self._get_child_offset_box),
            "grown_corridor_child": (self.get_corridor_interior_box, self._get_child_offset_box),
            "grown_room_child": (self.formulation.get_room_box, self._get_child_offset_box),
        }
        for kind, pairs in grown_pairs_by_kind.items():
            getters = grown_box_getters[kind]
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=pairs,
                get_box_a=getters[0],
                get_box_b=getters[1],
                prefix=kind,
                m_big=self.formulation.m_big,
                allowed_directions=grown_dirs_by_kind[kind],
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


def _extract_child_sub_box_bounds(
    key: tuple[str, int] | ChildSubClearanceBox,
    extraction: SolvedClusterExtraction,
    inventory: ClusterInventory,
) -> tuple[int, int, int, int, int, int]:
    """Retrieve absolute bounding box (min_x, max_x, min_y, max_y, min_z, max_z) for a child sub-box."""
    if isinstance(key, ChildSubClearanceBox):
        cid, k = key.child_id, key.box_index
    else:
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


def _corridor_interior_aabb(
    ae: ActiveEdge,
    p1: tuple[int, int, int],
    p2: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Compute the axis-aligned interior box of corridor ae excluding endpoint footprints."""
    dx, dy, dz = ae.direction
    src_lo, src_hi = getattr(ae, "src_extent", (0, 0))
    dst_lo, dst_hi = getattr(ae, "dst_extent", (0, 0))
    t_lo, t_hi = getattr(ae, "transverse_extent", (0, 0))
    if dx > 0:
        return (p1[0] + src_hi + 1, p2[0] - dst_lo - 1, p1[1] - t_lo, p1[1] + t_hi, p1[2], p1[2])
    if dx < 0:
        return (p2[0] + dst_hi + 1, p1[0] - src_lo - 1, p1[1] - t_lo, p1[1] + t_hi, p1[2], p1[2])
    if dy > 0:
        return (p1[0] - t_lo, p1[0] + t_hi, p1[1] + src_hi + 1, p2[1] - dst_lo - 1, p1[2], p1[2])
    if dy < 0:
        return (p1[0] - t_lo, p1[0] + t_hi, p2[1] + dst_hi + 1, p1[1] - src_lo - 1, p1[2], p1[2])
    if dz > 0:
        return (p1[0] - t_lo, p1[0] + t_hi, p1[1] - t_lo, p1[1] + t_hi, p1[2] + 1, p2[2] - 1)
    return (p1[0] - t_lo, p1[0] + t_hi, p1[1] - t_lo, p1[1] + t_hi, p2[2] + 1, p1[2] - 1)


def _corridor_full_aabb(
    ae: ActiveEdge,
    p1: tuple[int, int, int],
    p2: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Compute the axis-aligned full box of corridor ae (endpoints + transverse extent)."""
    dx, dy, dz = ae.direction
    t_lo, t_hi = getattr(ae, "transverse_extent", (0, 0))
    if dx != 0:
        return (min(p1[0], p2[0]), max(p1[0], p2[0]), p1[1] - t_lo, p1[1] + t_hi, p1[2], p1[2])
    if dy != 0:
        return (p1[0] - t_lo, p1[0] + t_hi, min(p1[1], p2[1]), max(p1[1], p2[1]), p1[2], p1[2])
    return (p1[0] - t_lo, p1[0] + t_hi, p1[1] - t_lo, p1[1] + t_hi, min(p1[2], p2[2]), max(p1[2], p2[2]))


def detect_cluster_collisions(
    formulation: ClusterFormulation,
    extraction: SolvedClusterExtraction,
) -> list[ClearanceViolation]:
    """Detect geometric collisions across all clearance categories from solved cluster coordinates.

    Evaluates exact coordinates and child offsets to identify active collisions across
    all six clearance categories:
      1. Room–Room: 3D footprint overlap between parent rooms and boundary ports.
      2. Room–Child: Parent room footprint penetrating a child keep-out sub-box.
      3. Child–Child: 3D AABB intersection between non-shared child keep-out sub-boxes.
      4. Corridor–Room: Corridor interior overlapping a third-party parent room footprint.
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

    room_extents = getattr(inventory, "room_extents", {})
    children = inventory.children
    child_solutions = inventory.child_solutions
    child_sub_boxes = inventory.child_sub_boxes
    active_edges = inventory.active_edges

    # Category 1: Room-Room Collisions
    for i in range(len(rooms)):
        u = rooms[i]
        coord_u = parent_coords.get(u)
        if coord_u is None:
            continue
        u_lo, u_hi = room_extents.get(u, (0, 0))
        u_x0, u_x1 = coord_u[0] - u_lo, coord_u[0] + u_hi
        u_y0, u_y1 = coord_u[1] - u_lo, coord_u[1] + u_hi
        for j in range(i + 1, len(rooms)):
            v = rooms[j]
            coord_v = parent_coords.get(v)
            if coord_v is None or coord_u[2] != coord_v[2]:
                continue
            v_lo, v_hi = room_extents.get(v, (0, 0))
            v_x0, v_x1 = coord_v[0] - v_lo, coord_v[0] + v_hi
            v_y0, v_y1 = coord_v[1] - v_lo, coord_v[1] + v_hi
            if max(u_x0, v_x0) <= min(u_x1, v_x1) and max(u_y0, v_y0) <= min(u_y1, v_y1):
                ov_x = min(u_x1, v_x1) - max(u_x0, v_x0) + 1
                ov_y = min(u_y1, v_y1) - max(u_y0, v_y0) + 1
                violations.append(
                    ClearanceViolation(
                        category="parent_room",
                        item_a=u,
                        item_b=v,
                        description=(
                            f"Parent room/port {u} at {coord_u} and {v} at {coord_v} overlap."
                        ),
                        overlap_volume=ov_x * ov_y,
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
            u_lo, u_hi = room_extents.get(u, (0, 0))
            u_x0, u_x1 = ux - u_lo, ux + u_hi
            u_y0, u_y1 = uy - u_lo, uy + u_hi
            for k in range(len(sb)):
                min_x, max_x, min_y, max_y, min_z, max_z = _extract_child_sub_box_bounds(
                    (cid, k), extraction, inventory
                )
                if (
                    max(u_x0, min_x) <= min(u_x1, max_x)
                    and max(u_y0, min_y) <= min(u_y1, max_y)
                    and min_z <= uz <= max_z
                ):
                    ov_x = min(u_x1, max_x) - max(u_x0, min_x) + 1
                    ov_y = min(u_y1, max_y) - max(u_y0, min_y) + 1
                    vol = ov_x * ov_y
                    violations.append(
                        ClearanceViolation(
                            category="room_child",
                            item_a=u,
                            item_b=(cid, k),
                            description=(
                                f"Room {u} at {u_coord} overlaps child '{cid}' sub-box {k} "
                                f"bounds ({min_x}..{max_x}, {min_y}..{max_y}, {min_z}..{max_z})."
                            ),
                            overlap_volume=vol,
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
                        ov_x = min(box1[1], box2[1]) - max(box1[0], box2[0]) + 1
                        ov_y = min(box1[3], box2[3]) - max(box1[2], box2[2]) + 1
                        ov_z = min(box1[5], box2[5]) - max(box1[4], box2[4]) + 1
                        vol = ov_x * ov_y * ov_z
                        violations.append(
                            ClearanceViolation(
                                category="child_child",
                                item_a=(c1.node_id, k1),
                                item_b=(c2.node_id, k2),
                                description=(
                                    f"Child '{c1.node_id}' sub-box {k1} overlaps "
                                    f"child '{c2.node_id}' sub-box {k2}."
                                ),
                                overlap_volume=vol,
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
        dx, dy, dz = ae.direction
        c_min_x, c_max_x, c_min_y, c_max_y, c_min_z, c_max_z = _corridor_interior_aabb(ae, p1, p2)
        if c_min_x > c_max_x or c_min_y > c_max_y or c_min_z > c_max_z:
            continue
        is_aligned = (
            (dx != 0 and p1[1] == p2[1] and p1[2] == p2[2])
            or (dy != 0 and p1[0] == p2[0] and p1[2] == p2[2])
            or (dz != 0 and p1[0] == p2[0] and p1[1] == p2[1])
        )
        seg = None if is_aligned else Segment3D(Coord3D(*p1), Coord3D(*p2))

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
            wx, wy, wz = w_coord
            w_lo, w_hi = room_extents.get(w, (0, 0))
            if is_aligned:
                if (
                    c_min_z <= wz <= c_max_z
                    and max(c_min_x, wx - w_lo) <= min(c_max_x, wx + w_hi)
                    and max(c_min_y, wy - w_lo) <= min(c_max_y, wy + w_hi)
                ):
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
            elif seg is not None:
                if dz != 0 and not (c_min_z <= wz <= c_max_z):
                    continue
                if dx != 0:
                    rx_min, rx_max = max(wx - w_lo, c_min_x), min(wx + w_hi, c_max_x)
                    ry_min, ry_max = wy - w_lo, wy + w_hi
                elif dy != 0:
                    rx_min, rx_max = wx - w_lo, wx + w_hi
                    ry_min, ry_max = max(wy - w_lo, c_min_y), min(wy + w_hi, c_max_y)
                else:
                    rx_min, rx_max = wx - w_lo, wx + w_hi
                    ry_min, ry_max = wy - w_lo, wy + w_hi
                if any(
                    segment_room_penetration(seg, Coord3D(rx, ry, wz))
                    for rx in range(rx_min, rx_max + 1)
                    for ry in range(ry_min, ry_max + 1)
                ):
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
        c_min_x, c_max_x, c_min_y, c_max_y, c_min_z, c_max_z = _corridor_interior_aabb(ae, p1, p2)

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
                    ov_x = min(c_max_x, box[1]) - max(c_min_x, box[0]) + 1
                    ov_y = min(c_max_y, box[3]) - max(c_min_y, box[2]) + 1
                    ov_z = min(c_max_z, box[5]) - max(c_min_z, box[4]) + 1
                    vol = ov_x * ov_y * ov_z
                    violations.append(
                        ClearanceViolation(
                            category="corridor_child",
                            item_a=i,
                            item_b=(s_id, k),
                            description=(
                                f"Corridor '{ae.edge_id}' interior intersects sibling child '{s_id}' sub-box {k}."
                            ),
                            overlap_volume=vol,
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
        t1 = getattr(ae1, "transverse_extent", (0, 0))
        corr_box1 = _corridor_full_aabb(ae1, p1_a, p1_b) if seg1.is_axis_aligned() else None

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
            crosses = segments_intersect_3d(seg1, seg2)
            if (
                not crosses
                and corr_box1 is not None
                and seg2.is_axis_aligned()
                and (t1 != (0, 0) or getattr(ae2, "transverse_extent", (0, 0)) != (0, 0))
            ):
                corr_box2 = _corridor_full_aabb(ae2, p2_a, p2_b)
                crosses = (
                    max(corr_box1[0], corr_box2[0]) <= min(corr_box1[1], corr_box2[1])
                    and max(corr_box1[2], corr_box2[2]) <= min(corr_box1[3], corr_box2[3])
                    and max(corr_box1[4], corr_box2[4]) <= min(corr_box1[5], corr_box2[5])
                )
            if crosses:
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

    return violations


__all__ = [
    "ChildSubClearanceBox",
    "ClearanceConstraintBuilder",
    "ClearanceEntity",
    "ClearanceViolation",
    "CorridorClearanceBox",
    "RoomClearanceBox",
    "VIOLATION_TIER_PRIORITY",
    "_extract_child_sub_box_bounds",
    "_extract_entity_coord",
    "detect_cluster_collisions",
]
