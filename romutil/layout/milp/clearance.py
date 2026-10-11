"""Stage 3: Geometric clearance and separation constraints for layout cluster MILP."""

from __future__ import annotations

from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.formulation import ClusterFormulation


def apply_clearance_constraints(formulation: ClusterFormulation) -> None:
    """Add disjunctive Big-M clearance and non-overlap constraints to the cluster model.

    Enforces pairwise 3D axis-aligned separation for:
      - Room-room non-overlap among parent rooms.
      - Child-child macro-box and sub-box separation among non-shared child clusters.
      - Room-child non-overlap between parent rooms and non-anchored child sub-boxes.
      - Corridor segment non-penetration against third-party parent rooms.
      - Corridor segment non-penetration against sibling child sub-boxes.
      - Corridor crossing separation between non-incident orthogonal corridor segments.

    Args:
        formulation: ClusterFormulation containing the Pyomo model, inventory, and
            coordinate box accessors.
    """
    model = formulation.model
    inventory = formulation.inventory
    rooms = inventory.rooms
    children = inventory.children
    child_solutions = inventory.child_solutions
    active_edges = inventory.active_edges
    gateway_anchors = inventory.gateway_anchors
    m_big = formulation.m_big
    edge_indices = list(range(len(active_edges)))

    # Parent room non-overlap separation
    if len(rooms) > 1:
        parent_pairs = [
            (rooms[i], rooms[j])
            for i in range(len(rooms))
            for j in range(i + 1, len(rooms))
        ]
        add_disjunctive_box_separation(
            model=model,
            pairs=parent_pairs,
            get_box_a=formulation.get_room_box,
            get_box_b=formulation.get_room_box,
            prefix="pr",
            m_big=m_big,
        )

    # Pairwise child-child sub-box separation
    child_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            c1 = children[i]
            c2 = children[j]
            sol1 = child_solutions.get(c1.node_id)
            sol2 = child_solutions.get(c2.node_id)
            if sol1 and sol2 and sol1.sub_boxes and sol2.sub_boxes:
                shared = set(sol1.local_coordinates.keys()) & set(sol2.local_coordinates.keys())
                if not shared:
                    for k1 in range(len(sol1.sub_boxes)):
                        for k2 in range(len(sol2.sub_boxes)):
                            child_pairs.append(((c1.node_id, k1), (c2.node_id, k2)))

    if child_pairs:
        add_disjunctive_box_separation(
            model=model,
            pairs=child_pairs,
            get_box_a=formulation.get_child_sub_box,
            get_box_b=formulation.get_child_sub_box,
            prefix="cb",
            m_big=m_big,
        )

    # Parent room vs child sub-box clearance
    rc_pairs: list[tuple[int, tuple[str, int]]] = []
    if rooms:
        for child in children:
            c_sol = child_solutions.get(child.node_id)
            if c_sol and c_sol.sub_boxes:
                anchors = gateway_anchors.get(child.node_id, frozenset())
                for u in rooms:
                    if u not in anchors and u not in c_sol.local_coordinates:
                        for k in range(len(c_sol.sub_boxes)):
                            rc_pairs.append((u, (child.node_id, k)))

    if rc_pairs:
        add_disjunctive_box_separation(
            model=model,
            pairs=rc_pairs,
            get_box_a=formulation.get_room_box,
            get_box_b=formulation.get_child_sub_box,
            prefix="rc",
            m_big=m_big,
        )

    # Corridor vs third-party room non-penetration
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
        model=model,
        pairs=corridor_room_pairs,
        get_box_a=lambda i: formulation.get_corridor_interior_box(active_edges[i]),
        get_box_b=formulation.get_room_box,
        prefix="cwr",
        m_big=m_big,
    )

    # Corridor vs sibling child sub-box non-penetration
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
                if not s_sol or not s_sol.sub_boxes:
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
                for k in range(len(s_sol.sub_boxes)):
                    corridor_child_pairs.append((i, (s_id, k)))

    if corridor_child_pairs:
        add_disjunctive_box_separation(
            model=model,
            pairs=corridor_child_pairs,
            get_box_a=lambda i: formulation.get_corridor_full_box(active_edges[i]),
            get_box_b=formulation.get_child_sub_box,
            prefix="cws",
            m_big=m_big,
        )

    # Corridor crossing separation
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
        model=model,
        pairs=crossing_pairs,
        get_box_a=lambda i: formulation.get_corridor_full_box(active_edges[i]),
        get_box_b=lambda i: formulation.get_corridor_full_box(active_edges[i]),
        prefix="cc",
        m_big=m_big,
    )


__all__ = [
    "apply_clearance_constraints",
]
