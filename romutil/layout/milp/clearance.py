"""Stage 3: Geometric clearance and separation constraints for layout cluster MILP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from romutil.layout.milp.constraints import add_disjunctive_box_separation

if TYPE_CHECKING:
    from romutil.layout.milp.formulation import ClusterFormulation


class ClearanceConstraintBuilder:
    """Builder enforcing disjunctive Big-M geometric clearance and separation constraints."""

    def __init__(self, formulation: ClusterFormulation) -> None:
        self.formulation = formulation

    def add_parent_room_separation(self) -> None:
        """Enforce pairwise room-room non-overlap separation among parent rooms."""
        rooms = self.formulation.inventory.rooms
        if len(rooms) > 1:
            parent_pairs = [
                (rooms[i], rooms[j])
                for i in range(len(rooms))
                for j in range(i + 1, len(rooms))
            ]
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=parent_pairs,
                get_box_a=self.formulation.get_room_box,
                get_box_b=self.formulation.get_room_box,
                prefix="pr",
                m_big=self.formulation.m_big,
            )

    def add_child_child_separation(self) -> None:
        """Enforce pairwise child-child sub-box separation among non-shared child clusters."""
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        child_sub_boxes = self.formulation.inventory.child_sub_boxes

        child_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1 = children[i]
                c2 = children[j]
                sol1 = child_solutions.get(c1.node_id)
                sol2 = child_solutions.get(c2.node_id)
                sb1 = child_sub_boxes.get(c1.node_id, ())
                sb2 = child_sub_boxes.get(c2.node_id, ())
                if sol1 and sol2 and sb1 and sb2:
                    shared = set(sol1.local_coordinates.keys()) & set(sol2.local_coordinates.keys())
                    if not shared:
                        for k1 in range(len(sb1)):
                            for k2 in range(len(sb2)):
                                child_pairs.append(((c1.node_id, k1), (c2.node_id, k2)))

        if child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=child_pairs,
                get_box_a=self.formulation.get_child_sub_box,
                get_box_b=self.formulation.get_child_sub_box,
                prefix="cb",
                m_big=self.formulation.m_big,
            )

    def add_room_child_separation(self) -> None:
        """Enforce parent room vs. child sub-box clearance."""
        rooms = self.formulation.inventory.rooms
        children = self.formulation.inventory.children
        child_solutions = self.formulation.inventory.child_solutions
        gateway_anchors = self.formulation.inventory.gateway_anchors
        child_sub_boxes = self.formulation.inventory.child_sub_boxes

        rc_pairs: list[tuple[int, tuple[str, int]]] = []
        if rooms:
            for child in children:
                c_sol = child_solutions.get(child.node_id)
                sb = child_sub_boxes.get(child.node_id, ())
                if c_sol and sb:
                    anchors = gateway_anchors.get(child.node_id, frozenset())
                    for u in rooms:
                        if u not in anchors and u not in c_sol.local_coordinates:
                            for k in range(len(sb)):
                                rc_pairs.append((u, (child.node_id, k)))

        if rc_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=rc_pairs,
                get_box_a=self.formulation.get_room_box,
                get_box_b=self.formulation.get_child_sub_box,
                prefix="rc",
                m_big=self.formulation.m_big,
            )

    def add_corridor_room_separation(self) -> None:
        """Enforce corridor segment non-penetration against third-party parent rooms."""
        rooms = self.formulation.inventory.rooms
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
                        for cid in ae.associated_children
                        if cid in child_solutions
                    ):
                        continue
                    corridor_room_pairs.append((i, w))

        add_disjunctive_box_separation(
            model=self.formulation.model,
            pairs=corridor_room_pairs,
            get_box_a=lambda i: self.formulation.get_corridor_interior_box(active_edges[i]),
            get_box_b=self.formulation.get_room_box,
            prefix="cwr",
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
                    sb = child_sub_boxes.get(s_id, ())
                    if not s_sol or not sb:
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
                    for k in range(len(sb)):
                        corridor_child_pairs.append((i, (s_id, k)))

        if corridor_child_pairs:
            add_disjunctive_box_separation(
                model=self.formulation.model,
                pairs=corridor_child_pairs,
                get_box_a=lambda i: self.formulation.get_corridor_full_box(active_edges[i]),
                get_box_b=self.formulation.get_child_sub_box,
                prefix="cws",
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
            model=self.formulation.model,
            pairs=crossing_pairs,
            get_box_a=lambda i: self.formulation.get_corridor_full_box(active_edges[i]),
            get_box_b=lambda i: self.formulation.get_corridor_full_box(active_edges[i]),
            prefix="cc",
            m_big=self.formulation.m_big,
        )

    def build(self) -> None:
        """Apply all disjunctive Big-M geometric clearance constraints."""
        self.add_parent_room_separation()
        self.add_child_child_separation()
        self.add_room_child_separation()
        self.add_corridor_room_separation()
        self.add_corridor_child_separation()
        self.add_corridor_crossing_separation()


def apply_clearance_constraints(formulation: ClusterFormulation) -> None:
    """Add disjunctive Big-M clearance and non-overlap constraints to the cluster model.

    Delegates to formulation.apply_clearances() for backward compatibility.

    Args:
        formulation: ClusterFormulation containing the Pyomo model and inventory.
    """
    formulation.apply_clearances()


__all__ = [
    "ClearanceConstraintBuilder",
    "apply_clearance_constraints",
]
