"""Stage 1: Inventory and interface classification for layout cluster MILP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from romutil.layout.geometry import BoundingBox3D, decompose_into_orthogonal_sub_boxes
from romutil.layout.ir import (
    ArticulationPointCut,
    BridgeCut,
    CutType,
    LayoutGraph,
    NodeSolution,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.types import Coord3D, LayoutOptions

EndpointRef = int | tuple[str, int]


@dataclass(frozen=True)
class ActiveEdge:
    """Active layout edge between room or boundary port endpoints."""

    edge_id: str
    src: EndpointRef
    dst: EndpointRef
    direction: tuple[int, int, int]
    min_length: int
    associated_rooms: frozenset[int]
    associated_children: frozenset[str]
    allow_transverse_slack: bool = False


@dataclass(frozen=True)
class RoomChildAlignment:
    """Alignment equality between a parent room and a child boundary port."""

    room_vnum: int
    child_id: str
    child_coord: Coord3D


@dataclass(frozen=True)
class ChildChildAlignment:
    """Alignment equality between two sibling child boundary ports."""

    base_child_id: str
    target_child_id: str
    base_coord: Coord3D
    target_coord: Coord3D


@dataclass(frozen=True)
class ClusterInventory:
    """Inventory of rooms, children, interfaces, and active edges for cluster layout."""

    node: SolveTreeNode
    rooms: tuple[int, ...]
    children: tuple[SolveTreeNode, ...]
    child_solutions: Mapping[str, NodeSolution]
    boundary_ports: frozenset[int]
    active_edges: tuple[ActiveEdge, ...]
    room_alignments: tuple[RoomChildAlignment, ...]
    child_alignments: tuple[ChildChildAlignment, ...]
    gateway_anchors: Mapping[str, frozenset[int]]
    aligned_children: frozenset[str]
    equated_children: frozenset[frozenset[str]]
    child_sub_boxes: Mapping[str, tuple[BoundingBox3D, ...]] = field(default_factory=dict)


def get_child_boundary_ports(child: SolveTreeNode) -> set[int]:
    """Return the set of room IDs that act as boundary ports for this child."""
    ports: set[int] = set()
    contract = child.cut_contract
    if isinstance(contract, SplitPairCut):
        ports.update(contract.poles)
    elif (
        child.cut_type == CutType.SPLIT_PAIR
        or child.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair")
        or "poles" in child.metadata
    ):
        raw_poles = child.metadata.get("poles")
        if raw_poles and len(raw_poles) == 2:
            ports.add(int(raw_poles[0]))
            ports.add(int(raw_poles[1]))
    return ports


_get_child_boundary_ports = get_child_boundary_ports


def derive_child_sub_boxes(
    child: SolveTreeNode,
    child_solution: NodeSolution,
    boundary_ports: frozenset[int] | None = None,
) -> tuple[BoundingBox3D, ...]:
    """Derive orthogonal keep-out sub-boxes for a child component within parent context.

    Boundary ports connecting the child to parent rooms or sibling split pairs are
    excluded from the keep-out footprint to allow external docking without triggering
    Big-M non-overlap collisions. For leaf children, non-convex collections of interior
    cells are partitioned into minimal disjoint orthogonal boxes. For composite children,
    the enclosing bounding box is returned.

    Args:
        child: SolveTreeNode representing the child cluster.
        child_solution: Solved NodeSolution for the child cluster.
        boundary_ports: Set of boundary port room IDs to exclude. If None, derived
            from child cut contracts and metadata.

    Returns:
        Tuple of disjoint BoundingBox3D instances representing keep-out obstacles.
    """
    if child_solution.sub_boxes:
        return child_solution.sub_boxes

    if boundary_ports is None:
        boundary_ports = frozenset(get_child_boundary_ports(child))

    if not child.children:
        interior_cells: set[Coord3D] = {
            coord
            for u, coord in child_solution.local_coordinates.items()
            if u not in boundary_ports
        }
        if not interior_cells:
            return ()
        return decompose_into_orthogonal_sub_boxes(interior_cells)

    bbox = child_solution.bounding_box
    if bbox is not None:
        return (bbox,)
    return ()


def inventory_cluster(
    node: SolveTreeNode,
    child_solutions: Mapping[str, NodeSolution],
    graph: LayoutGraph,
    options: LayoutOptions | None = None,
) -> ClusterInventory:
    """Inventory rooms, child components, boundary ports, alignments, and active edges.

    Ingests the cluster node, solved child clusters, layout graph, and options,
    classifying all boundary interfaces and building active edges including internal
    room edges, gateway bridges, articulation points, split-pair boundary port edges,
    and inter-child bridges.

    Args:
        node: SolveTreeNode representing the cluster to lay out.
        child_solutions: Mapping of child node_id to already-solved NodeSolution.
        graph: Underlying LayoutGraph with room definitions and exit edges.
        options: Optional layout configuration parameters.

    Returns:
        ClusterInventory containing classified interfaces, alignments, and active edges.
    """
    rooms = tuple(sorted(node.rooms))
    children = tuple(node.children)
    child_ids = [c.node_id for c in children]
    room_set = set(rooms)
    boundary_ports = frozenset(get_child_boundary_ports(node))

    gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in children}
    aligned_children: set[str] = set()
    if not rooms and child_ids:
        aligned_children.add(child_ids[0])

    active_edges: list[ActiveEdge] = []
    room_alignments: list[RoomChildAlignment] = []
    child_alignments: list[ChildChildAlignment] = []
    equated_children: set[frozenset[str]] = set()

    # Split-pair alignment (rigid boundary ports)
    split_children_by_poles: dict[tuple[int, int], list[SolveTreeNode]] = {}
    for child in children:
        if child.node_id not in child_solutions:
            continue
        poles: tuple[int, int] | None = None
        contract = child.cut_contract
        if isinstance(contract, SplitPairCut):
            poles = contract.poles
        elif contract is None:
            if (
                child.cut_type == CutType.SPLIT_PAIR
                or child.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair")
                or "poles" in child.metadata
            ):
                raw_poles = child.metadata.get("poles")
                if raw_poles and len(raw_poles) == 2:
                    poles = (int(raw_poles[0]), int(raw_poles[1]))

        if poles is not None:
            p0, p1 = poles[0], poles[1]
            c_coords = child_solutions[child.node_id].local_coordinates
            if p0 in c_coords and p1 in c_coords:
                can_pair = (min(p0, p1), max(p0, p1))
                split_children_by_poles.setdefault(can_pair, []).append(child)

    # Support container-level split metadata
    if node.cut_type in (CutType.SPLIT_PAIR, "split_pair") or (
        node.node_type in (TreeNodeType.SPLIT_PAIR_CONTAINER, "split_pair_container") and "poles" in node.metadata
    ):
        raw_node_poles = node.metadata.get("poles")
        if raw_node_poles and len(raw_node_poles) == 2:
            np0, np1 = int(raw_node_poles[0]), int(raw_node_poles[1])
            can_pair = (min(np0, np1), max(np0, np1))
            for child in children:
                if child.node_id in child_solutions:
                    c_coords = child_solutions[child.node_id].local_coordinates
                    if np0 in c_coords and np1 in c_coords:
                        if child not in split_children_by_poles.get(can_pair, []):
                            split_children_by_poles.setdefault(can_pair, []).append(child)

    split_pole_pairs = set(split_children_by_poles.keys())

    is_split_pair_container = (
        node.node_type in (TreeNodeType.SPLIT_PAIR_CONTAINER, "split_pair_container")
        or (node.node_type == "container" and "poles" in node.metadata)
    )

    split_min_lengths: dict[tuple[int, int], int] = {}
    split_directions: dict[tuple[int, int], tuple[int, int, int]] = {}
    room_anchored_children: dict[int, list[str]] = {}
    child_anchor_poles: dict[str, int] = {}

    for (p0, p1), split_children in split_children_by_poles.items():
        if not split_children:
            continue

        for child in split_children:
            c_sol = child_solutions[child.node_id]
            anchor_pole = p0 if p0 in room_set else (p1 if p1 in room_set else p0)
            if (
                p0 in room_set
                and p1 in room_set
                and p0 in c_sol.local_coordinates
                and p1 in c_sol.local_coordinates
            ):
                pc0 = c_sol.local_coordinates[p0]
                pc1 = c_sol.local_coordinates[p1]
                pdx = pc1.x - pc0.x
                pdy = pc1.y - pc0.y
                pdz = pc1.z - pc0.z
                max_d = max(abs(pdx), abs(pdy), abs(pdz))
                primary_axis = 0 if abs(pdx) == max_d else (1 if abs(pdy) == max_d else 2)

                def _has_transverse_exit(p_check: int) -> bool:
                    for ge in graph.edges:
                        if ge.cut or {ge.src, ge.dst} == {p0, p1}:
                            continue
                        if ge.src == p_check and ge.dst in c_sol.local_coordinates and ge.dst not in (p0, p1):
                            try:
                                disp = direction_to_displacement(ge.direction)
                                if disp[primary_axis] == 0:
                                    return True
                            except ValueError:
                                pass
                        elif ge.dst == p_check and ge.src in c_sol.local_coordinates and ge.src not in (p0, p1):
                            try:
                                disp = direction_to_displacement(ge.direction)
                                if disp[primary_axis] == 0:
                                    return True
                            except ValueError:
                                pass
                    return False

                p0_trans = _has_transverse_exit(p0)
                p1_trans = _has_transverse_exit(p1)
                if p1_trans and not p0_trans:
                    anchor_pole = p1
                elif p0_trans and not p1_trans:
                    anchor_pole = p0

            child_anchor_poles[child.node_id] = anchor_pole
            pc_anchor = c_sol.local_coordinates[anchor_pole]
            if anchor_pole in room_set:
                room_alignments.append(
                    RoomChildAlignment(
                        room_vnum=anchor_pole,
                        child_id=child.node_id,
                        child_coord=pc_anchor,
                    )
                )
                gateway_anchors[child.node_id].add(anchor_pole)
                aligned_children.add(child.node_id)
                room_anchored_children.setdefault(anchor_pole, []).append(child.node_id)
            elif len(split_children) > 1:
                base_child = split_children[0]
                base_pc0 = child_solutions[base_child.node_id].local_coordinates[p0]
                pc0 = c_sol.local_coordinates[p0]
                child_alignments.append(
                    ChildChildAlignment(
                        base_child_id=base_child.node_id,
                        target_child_id=child.node_id,
                        base_coord=base_pc0,
                        target_coord=pc0,
                    )
                )
                aligned_children.add(child.node_id)
                equated_children.add(frozenset({base_child.node_id, child.node_id}))

        if len(split_children) > 1 and p0 not in room_set and p1 not in room_set:
            for idx1 in range(len(split_children)):
                for idx2 in range(idx1 + 1, len(split_children)):
                    equated_children.add(frozenset({split_children[idx1].node_id, split_children[idx2].node_id}))

    for p_anchor, c_list in room_anchored_children.items():
        for idx1 in range(len(c_list)):
            for idx2 in range(idx1 + 1, len(c_list)):
                equated_children.add(frozenset({c_list[idx1], c_list[idx2]}))

        def _child_span(c: SolveTreeNode) -> int:
            c_coords = child_solutions[c.node_id].local_coordinates
            if p0 not in c_coords or p1 not in c_coords:
                return 0
            return (
                abs(c_coords[p1].x - c_coords[p0].x)
                + abs(c_coords[p1].y - c_coords[p0].y)
                + abs(c_coords[p1].z - c_coords[p0].z)
            )

        prim_child = max(split_children, key=_child_span)
        prim_sol = child_solutions[prim_child.node_id]
        pc0_prim = prim_sol.local_coordinates[p0]
        pc1_prim = prim_sol.local_coordinates[p1]
        pdx = pc1_prim.x - pc0_prim.x
        pdy = pc1_prim.y - pc0_prim.y
        pdz = pc1_prim.z - pc0_prim.z

        is_axis_aligned = (sum(1 for d in (pdx, pdy, pdz) if d != 0) == 1)
        d_vec: tuple[int, int, int] | None = None

        if is_split_pair_container:
            if is_axis_aligned:
                d_vec = (
                    1 if pdx > 0 else (-1 if pdx < 0 else 0),
                    1 if pdy > 0 else (-1 if pdy < 0 else 0),
                    1 if pdz > 0 else (-1 if pdz < 0 else 0),
                )
            else:
                abs_deltas = [
                    (abs(pdx), (1 if pdx > 0 else (-1 if pdx < 0 else 0), 0, 0)),
                    (abs(pdy), (0, 1 if pdy > 0 else (-1 if pdy < 0 else 0), 0)),
                    (abs(pdz), (0, 0, 1 if pdz > 0 else (-1 if pdz < 0 else 0))),
                ]
                max_delta, d_cand = max(abs_deltas, key=lambda t: t[0])
                d_vec = (1, 0, 0) if max_delta == 0 else d_cand

        l_min = max(
            1,
            max(
                max(
                    abs(child_solutions[c.node_id].local_coordinates[p1].x - child_solutions[c.node_id].local_coordinates[p0].x),
                    abs(child_solutions[c.node_id].local_coordinates[p1].y - child_solutions[c.node_id].local_coordinates[p0].y),
                    abs(child_solutions[c.node_id].local_coordinates[p1].z - child_solutions[c.node_id].local_coordinates[p0].z),
                )
                for c in split_children
            ),
        )

        for ge in graph.edges:
            if not ge.cut and {ge.src, ge.dst} == {p0, p1}:
                l_min = max(l_min, max(1, ge.weight))

        can_pair = (min(p0, p1), max(p0, p1))
        split_min_lengths[can_pair] = l_min
        if d_vec is not None:
            split_directions[can_pair] = d_vec

    # External parent edges and inter-child edges connecting to split-pair boundary ports
    seen_split_corridors: set[frozenset[EndpointRef]] = set()
    for (p0, p1), split_children in split_children_by_poles.items():
        for sc in split_children:
            c_coords = child_solutions[sc.node_id].local_coordinates
            anchor_p = child_anchor_poles.get(sc.node_id, p0)
            for edge in graph.edges:
                if edge.cut:
                    continue
                if {edge.src, edge.dst} == {p0, p1}:
                    continue
                if edge.src in room_set and edge.dst in room_set:
                    continue
                try:
                    disp = direction_to_displacement(edge.direction)
                except ValueError:
                    continue

                # Parent room to split-pair child room
                if edge.src in room_set and edge.dst in c_coords and edge.dst not in room_set:
                    u, v = edge.src, edge.dst
                    if u != anchor_p:
                        corridor_key = frozenset({u, (sc.node_id, v)})
                        if corridor_key not in seen_split_corridors:
                            seen_split_corridors.add(corridor_key)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"ext_{sc.node_id}_{u}_{v}",
                                    src=u,
                                    dst=(sc.node_id, v),
                                    direction=disp,
                                    min_length=max(1, edge.weight),
                                    associated_rooms=frozenset({u}),
                                    associated_children=frozenset({sc.node_id}),
                                    allow_transverse_slack=True,
                                )
                            )
                            gateway_anchors[sc.node_id].add(u)
                            aligned_children.add(sc.node_id)
                elif edge.dst in room_set and edge.src in c_coords and edge.src not in room_set:
                    u, v = edge.dst, edge.src
                    if u != anchor_p:
                        corridor_key = frozenset({u, (sc.node_id, v)})
                        if corridor_key not in seen_split_corridors:
                            seen_split_corridors.add(corridor_key)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"ext_{sc.node_id}_{v}_{u}",
                                    src=(sc.node_id, v),
                                    dst=u,
                                    direction=disp,
                                    min_length=max(1, edge.weight),
                                    associated_rooms=frozenset({u}),
                                    associated_children=frozenset({sc.node_id}),
                                    allow_transverse_slack=True,
                                )
                            )
                            gateway_anchors[sc.node_id].add(u)
                            aligned_children.add(sc.node_id)
                else:
                    # Inter-child edge to split-pair boundary ports
                    for other in children:
                        if other.node_id == sc.node_id:
                            continue
                        if frozenset({other.node_id, sc.node_id}) in equated_children:
                            continue
                        if edge.src == edge.dst:
                            continue
                        other_coords = (
                            child_solutions[other.node_id].local_coordinates
                            if other.node_id in child_solutions
                            else {}
                        )
                        # Skip if either child fully contains both endpoints (internal edge of that child)
                        if (edge.src in c_coords and edge.dst in c_coords) or (
                            edge.src in other_coords and edge.dst in other_coords
                        ):
                            continue

                        other_ports = get_child_boundary_ports(other)
                        if (
                            edge.src in other_ports
                            and edge.dst in c_coords
                            and edge.src not in c_coords
                            and edge.dst not in other_coords
                        ):
                            corridor_key = frozenset({(other.node_id, edge.src), (sc.node_id, edge.dst)})
                            if corridor_key not in seen_split_corridors:
                                seen_split_corridors.add(corridor_key)
                                active_edges.append(
                                    ActiveEdge(
                                        edge_id=f"ext_{other.node_id}_{sc.node_id}_{edge.src}_{edge.dst}",
                                        src=(other.node_id, edge.src),
                                        dst=(sc.node_id, edge.dst),
                                        direction=disp,
                                        min_length=max(1, edge.weight),
                                        associated_rooms=frozenset(),
                                        associated_children=frozenset({other.node_id, sc.node_id}),
                                        allow_transverse_slack=True,
                                    )
                                )
                                aligned_children.add(sc.node_id)
                        elif (
                            edge.dst in other_ports
                            and edge.src in c_coords
                            and edge.dst not in c_coords
                            and edge.src not in other_coords
                        ):
                            corridor_key = frozenset({(sc.node_id, edge.src), (other.node_id, edge.dst)})
                            if corridor_key not in seen_split_corridors:
                                seen_split_corridors.add(corridor_key)
                                active_edges.append(
                                    ActiveEdge(
                                        edge_id=f"ext_{sc.node_id}_{other.node_id}_{edge.src}_{edge.dst}",
                                        src=(sc.node_id, edge.src),
                                        dst=(other.node_id, edge.dst),
                                        direction=disp,
                                        min_length=max(1, edge.weight),
                                        associated_rooms=frozenset(),
                                        associated_children=frozenset({sc.node_id, other.node_id}),
                                        allow_transverse_slack=True,
                                    )
                                )
                                aligned_children.add(sc.node_id)

    # 1-cut articulation point alignment
    for child in children:
        if child.node_id in aligned_children:
            continue
        c_sol = child_solutions.get(child.node_id)
        if not c_sol:
            continue
        c_coords = c_sol.local_coordinates
        a: int | None = None
        contract = child.cut_contract
        if isinstance(contract, ArticulationPointCut):
            a = contract.anchor_vnum
        elif (
            child.cut_type == CutType.ARTICULATION_POINT
            or child.metadata.get("cut_type") in (CutType.ARTICULATION_POINT, "articulation_point")
            or "anchor_vnum" in child.metadata
        ):
            raw_a = child.metadata.get("anchor_vnum")
            if raw_a is not None:
                a = int(raw_a)

        if a is not None and a in room_set and a in c_coords:
            pc = c_coords[a]
            room_alignments.append(
                RoomChildAlignment(
                    room_vnum=a,
                    child_id=child.node_id,
                    child_coord=pc,
                )
            )
            gateway_anchors[child.node_id].add(a)
            aligned_children.add(child.node_id)
            for other_cid in room_anchored_children.get(a, []):
                equated_children.add(frozenset({other_cid, child.node_id}))
            room_anchored_children.setdefault(a, []).append(child.node_id)

    # 1-cut doorway / bridge edge alignment
    for child in children:
        if child.node_id in aligned_children:
            continue
        c_sol = child_solutions.get(child.node_id)
        if not c_sol:
            continue
        c_coords = c_sol.local_coordinates

        contract = child.cut_contract
        if isinstance(contract, BridgeCut):
            u, v = contract.bridge_edge
            dir_val = contract.direction
            weight = contract.weight
            if u in room_set and v in c_coords:
                try:
                    disp = direction_to_displacement(dir_val)
                    active_edges.append(
                        ActiveEdge(
                            edge_id=f"gw_{child.node_id}_{u}_{v}",
                            src=u,
                            dst=(child.node_id, v),
                            direction=disp,
                            min_length=max(1, weight),
                            associated_rooms=frozenset({u}),
                            associated_children=frozenset({child.node_id}),
                        )
                    )
                    gateway_anchors[child.node_id].add(u)
                    aligned_children.add(child.node_id)
                    continue
                except ValueError:
                    pass
            elif v in room_set and u in c_coords:
                try:
                    disp = direction_to_displacement(dir_val)
                    active_edges.append(
                        ActiveEdge(
                            edge_id=f"gw_{child.node_id}_{u}_{v}",
                            src=(child.node_id, u),
                            dst=v,
                            direction=disp,
                            min_length=max(1, weight),
                            associated_rooms=frozenset({v}),
                            associated_children=frozenset({child.node_id}),
                        )
                    )
                    gateway_anchors[child.node_id].add(v)
                    aligned_children.add(child.node_id)
                    continue
                except ValueError:
                    pass

        for edge in graph.edges:
            if edge.cut:
                continue
            if edge.src in room_set and edge.dst in c_coords:
                try:
                    disp = direction_to_displacement(edge.direction)
                    active_edges.append(
                        ActiveEdge(
                            edge_id=f"gw_{child.node_id}_{edge.src}_{edge.dst}",
                            src=edge.src,
                            dst=(child.node_id, edge.dst),
                            direction=disp,
                            min_length=max(1, edge.weight),
                            associated_rooms=frozenset({edge.src}),
                            associated_children=frozenset({child.node_id}),
                        )
                    )
                    gateway_anchors[child.node_id].add(edge.src)
                    aligned_children.add(child.node_id)
                    break
                except ValueError:
                    pass
            elif edge.dst in room_set and edge.src in c_coords:
                try:
                    disp = direction_to_displacement(edge.direction)
                    active_edges.append(
                        ActiveEdge(
                            edge_id=f"gw_{child.node_id}_{edge.src}_{edge.dst}",
                            src=(child.node_id, edge.src),
                            dst=edge.dst,
                            direction=disp,
                            min_length=max(1, edge.weight),
                            associated_rooms=frozenset({edge.dst}),
                            associated_children=frozenset({child.node_id}),
                        )
                    )
                    gateway_anchors[child.node_id].add(edge.dst)
                    aligned_children.add(child.node_id)
                    break
                except ValueError:
                    pass

    # Child-child bridge alignment
    if len(children) > 1:
        for child in children:
            if child.node_id in aligned_children:
                continue
            c_sol = child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates

            contract = child.cut_contract
            if isinstance(contract, BridgeCut):
                u, v = contract.bridge_edge
                for other in children:
                    if other.node_id == child.node_id or other.node_id not in aligned_children:
                        continue
                    if frozenset({other.node_id, child.node_id}) in equated_children:
                        continue
                    other_coords = (
                        child_solutions[other.node_id].local_coordinates
                        if other.node_id in child_solutions
                        else {}
                    )
                    if u in other_coords and v in c_coords:
                        try:
                            disp = direction_to_displacement(contract.direction)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"ccb_{other.node_id}_{child.node_id}_{u}_{v}",
                                    src=(other.node_id, u),
                                    dst=(child.node_id, v),
                                    direction=disp,
                                    min_length=max(1, contract.weight),
                                    associated_rooms=frozenset(),
                                    associated_children=frozenset({other.node_id, child.node_id}),
                                )
                            )
                            aligned_children.add(child.node_id)
                            break
                        except ValueError:
                            pass
                    elif v in other_coords and u in c_coords:
                        try:
                            disp = direction_to_displacement(contract.direction)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"ccb_{child.node_id}_{other.node_id}_{u}_{v}",
                                    src=(child.node_id, u),
                                    dst=(other.node_id, v),
                                    direction=disp,
                                    min_length=max(1, contract.weight),
                                    associated_rooms=frozenset(),
                                    associated_children=frozenset({child.node_id, other.node_id}),
                                )
                            )
                            aligned_children.add(child.node_id)
                            break
                        except ValueError:
                            pass
            if child.node_id in aligned_children:
                continue
            for edge in graph.edges:
                if edge.cut:
                    continue
                for other in children:
                    if other.node_id == child.node_id or other.node_id not in aligned_children:
                        continue
                    if frozenset({other.node_id, child.node_id}) in equated_children:
                        continue
                    other_coords = (
                        child_solutions[other.node_id].local_coordinates
                        if other.node_id in child_solutions
                        else {}
                    )
                    if edge.src in other_coords and edge.dst in c_coords:
                        try:
                            disp = direction_to_displacement(edge.direction)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"ccb_{other.node_id}_{child.node_id}_{edge.src}_{edge.dst}",
                                    src=(other.node_id, edge.src),
                                    dst=(child.node_id, edge.dst),
                                    direction=disp,
                                    min_length=max(1, edge.weight),
                                    associated_rooms=frozenset(),
                                    associated_children=frozenset({other.node_id, child.node_id}),
                                )
                            )
                            aligned_children.add(child.node_id)
                            break
                        except ValueError:
                            pass
                    elif edge.dst in other_coords and edge.src in c_coords:
                        try:
                            disp = direction_to_displacement(edge.direction)
                            active_edges.append(
                                ActiveEdge(
                                    edge_id=f"ccb_{child.node_id}_{other.node_id}_{edge.src}_{edge.dst}",
                                    src=(child.node_id, edge.src),
                                    dst=(other.node_id, edge.dst),
                                    direction=disp,
                                    min_length=max(1, edge.weight),
                                    associated_rooms=frozenset(),
                                    associated_children=frozenset({child.node_id, other.node_id}),
                                )
                            )
                            aligned_children.add(child.node_id)
                            break
                        except ValueError:
                            pass
                if child.node_id in aligned_children:
                    break

    # Internal room-room edges
    pole_edges_handled: set[tuple[int, int]] = set()
    if rooms:
        for e in graph.edges:
            if not e.cut and e.src in room_set and e.dst in room_set:
                can_p = (min(e.src, e.dst), max(e.src, e.dst))
                if can_p in split_pole_pairs:
                    if can_p in pole_edges_handled:
                        continue
                    pole_edges_handled.add(can_p)
                    edge_min_len = max(e.weight, split_min_lengths.get(can_p, 1))
                    try:
                        disp = direction_to_displacement(e.direction)
                        active_edges.append(
                            ActiveEdge(
                                edge_id=f"int_{e.src}_{e.dst}",
                                src=e.src,
                                dst=e.dst,
                                direction=disp,
                                min_length=max(1, edge_min_len),
                                associated_rooms=frozenset({e.src, e.dst}),
                                associated_children=frozenset(
                                    c.node_id for c in split_children_by_poles.get(can_p, [])
                                ),
                            )
                        )
                    except ValueError:
                        pass
                    continue
                try:
                    disp = direction_to_displacement(e.direction)
                    active_edges.append(
                        ActiveEdge(
                            edge_id=f"int_{e.src}_{e.dst}",
                            src=e.src,
                            dst=e.dst,
                            direction=disp,
                            min_length=max(1, e.weight),
                            associated_rooms=frozenset({e.src, e.dst}),
                            associated_children=frozenset(),
                        )
                    )
                except ValueError:
                    continue

    for (p0, p1), split_children in split_children_by_poles.items():
        if p0 in room_set and p1 in room_set:
            can_p = (min(p0, p1), max(p0, p1))
            if can_p in pole_edges_handled:
                continue
            if can_p not in split_directions:
                continue
            d_vec = split_directions[can_p]
            pole_edges_handled.add(can_p)
            active_edges.append(
                ActiveEdge(
                    edge_id=f"sp_{p0}_{p1}",
                    src=p0,
                    dst=p1,
                    direction=d_vec,
                    min_length=split_min_lengths[can_p],
                    associated_rooms=frozenset({p0, p1}),
                    associated_children=frozenset({c.node_id for c in split_children}),
                )
            )

    child_sub_boxes: dict[str, tuple[BoundingBox3D, ...]] = {}
    for child in children:
        if child.node_id in child_solutions:
            child_ports = frozenset(get_child_boundary_ports(child))
            child_sub_boxes[child.node_id] = derive_child_sub_boxes(
                child=child,
                child_solution=child_solutions[child.node_id],
                boundary_ports=child_ports,
            )
        else:
            child_sub_boxes[child.node_id] = ()

    return ClusterInventory(
        node=node,
        rooms=rooms,
        children=children,
        child_solutions=child_solutions,
        boundary_ports=boundary_ports,
        active_edges=tuple(active_edges),
        room_alignments=tuple(room_alignments),
        child_alignments=tuple(child_alignments),
        gateway_anchors={k: frozenset(v) for k, v in gateway_anchors.items()},
        aligned_children=frozenset(aligned_children),
        equated_children=frozenset(equated_children),
        child_sub_boxes=child_sub_boxes,
    )


__all__ = [
    "ActiveEdge",
    "ChildChildAlignment",
    "ClusterInventory",
    "EndpointRef",
    "RoomChildAlignment",
    "derive_child_sub_boxes",
    "get_child_boundary_ports",
    "inventory_cluster",
]
