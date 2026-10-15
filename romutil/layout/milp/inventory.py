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
class SplitPairResult:
    """Classification results produced by SplitPairHandler."""

    split_children_by_poles: dict[tuple[int, int], list[SolveTreeNode]]
    room_alignments: tuple[RoomChildAlignment, ...]
    child_alignments: tuple[ChildChildAlignment, ...]
    equated_children: frozenset[frozenset[str]]
    corridor_edges: tuple[ActiveEdge, ...]
    gateway_anchors: Mapping[str, frozenset[int]]
    aligned_children: frozenset[str]

    @property
    def active_edges(self) -> tuple[ActiveEdge, ...]:
        """All active edges produced by the split-pair handler."""
        return self.corridor_edges


@dataclass(frozen=True)
class GatewayCutResult:
    """Classification results produced by GatewayCutHandler."""

    room_alignments: tuple[RoomChildAlignment, ...]
    active_edges: tuple[ActiveEdge, ...]
    gateway_anchors: Mapping[str, frozenset[int]]
    aligned_children: frozenset[str]
    equated_children: frozenset[frozenset[str]]


@dataclass(frozen=True)
class InterChildBridgeResult:
    """Classification results produced by InterChildBridgeHandler."""

    active_edges: tuple[ActiveEdge, ...]
    aligned_children: frozenset[str]


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
    graph: LayoutGraph | None = None

    @classmethod
    def from_node(
        cls,
        node: SolveTreeNode,
        child_solutions: Mapping[str, NodeSolution],
        graph: LayoutGraph,
        options: LayoutOptions | None = None,
    ) -> ClusterInventory:
        """Construct ClusterInventory from cluster node, child solutions, and layout graph."""
        return inventory_cluster(node, child_solutions, graph, options)


def get_child_boundary_ports(child: SolveTreeNode) -> set[int]:
    """Return the set of room IDs that act as boundary ports for this child."""
    ports: set[int] = set()
    contract = child.cut_contract
    if isinstance(contract, SplitPairCut):
        ports.update(contract.poles)
    elif isinstance(contract, (ArticulationPointCut, BridgeCut)):
        anchor = getattr(contract, "anchor_vnum", None)
        if anchor is not None:
            ports.add(int(anchor))
        elif "anchor_vnum" in child.metadata:
            raw_anchor = child.metadata.get("anchor_vnum")
            if raw_anchor is not None:
                ports.add(int(raw_anchor))
    elif (
        child.cut_type == CutType.SPLIT_PAIR
        or child.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair")
        or "poles" in child.metadata
    ):
        raw_poles = child.metadata.get("poles")
        if raw_poles and len(raw_poles) == 2:
            ports.add(int(raw_poles[0]))
            ports.add(int(raw_poles[1]))
    elif (
        child.cut_type in (CutType.ARTICULATION_POINT, CutType.BRIDGE)
        or child.metadata.get("cut_type")
        in (CutType.ARTICULATION_POINT, "articulation_point", CutType.BRIDGE, "bridge")
        or "anchor_vnum" in child.metadata
    ):
        raw_anchor = child.metadata.get("anchor_vnum")
        if raw_anchor is not None:
            ports.add(int(raw_anchor))
    return ports


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
    constituent parent rooms and translated child keep-out sub-boxes are aggregated and
    decomposed into minimal disjoint orthogonal boxes (excluding boundary ports), preserving
    interior voids and concavities within the cluster footprint.

    Args:
        child: SolveTreeNode representing the child cluster.
        child_solution: Solved NodeSolution for the child cluster.
        boundary_ports: Set of boundary port room IDs to exclude. If None, derived
            from child cut contracts and metadata.

    Returns:
        Tuple of disjoint BoundingBox3D instances representing keep-out obstacles.
    """
    if boundary_ports is None:
        boundary_ports = frozenset(get_child_boundary_ports(child))

    boundary_port_coords: set[Coord3D] = {
        coord
        for u, coord in child_solution.local_coordinates.items()
        if u in boundary_ports
    }
    if "all_room_coordinates" in child_solution.metadata:
        all_rooms = child_solution.metadata["all_room_coordinates"]
        for u in boundary_ports:
            if u in all_rooms:
                boundary_port_coords.add(all_rooms[u])

    # If the child solution has pre-computed sub-boxes (e.g. from normalization or explicit test fixture)
    if child_solution.sub_boxes:
        has_boundary_overlap = any(
            Coord3D(x, y, z) in boundary_port_coords
            for sbox in child_solution.sub_boxes
            for x in range(sbox.min_x, sbox.max_x + 1)
            for y in range(sbox.min_y, sbox.max_y + 1)
            for z in range(sbox.min_z, sbox.max_z + 1)
        )
        if not has_boundary_overlap:
            return child_solution.sub_boxes

        # If any sub-box overlaps boundary ports, carve out the boundary port cells and re-decompose
        interior_cells: set[Coord3D] = {
            Coord3D(x, y, z)
            for sbox in child_solution.sub_boxes
            for x in range(sbox.min_x, sbox.max_x + 1)
            for y in range(sbox.min_y, sbox.max_y + 1)
            for z in range(sbox.min_z, sbox.max_z + 1)
            if Coord3D(x, y, z) not in boundary_port_coords
        }
        if not interior_cells:
            return ()
        return decompose_into_orthogonal_sub_boxes(interior_cells)

    # For leaf children without pre-computed sub-boxes
    if not child.children:
        interior_cells = {
            coord
            for u, coord in child_solution.local_coordinates.items()
            if u not in boundary_ports and coord not in boundary_port_coords
        }
        if not interior_cells:
            return ()
        return decompose_into_orthogonal_sub_boxes(interior_cells)

    # Fallback to coarse bounding box if child was not solved with child_sub_boxes metadata
    if "child_sub_boxes" not in child_solution.metadata and child.children:
        bbox = child_solution.bounding_box
        if bbox is not None:
            return (bbox,)
        return ()

    # For composite children, derive occupied cells from all descendant rooms and translated child sub-boxes
    occupied_cells: set[Coord3D] = set()
    if "all_room_coordinates" in child_solution.metadata:
        for u, coord in child_solution.metadata["all_room_coordinates"].items():
            if u not in boundary_ports and coord not in boundary_port_coords:
                occupied_cells.add(coord)
    else:
        for u, coord in child_solution.local_coordinates.items():
            if u not in boundary_ports and coord not in boundary_port_coords:
                occupied_cells.add(coord)

    child_offsets = child_solution.metadata.get("child_offsets", {})
    child_sub_boxes_map = child_solution.metadata.get("child_sub_boxes", {})

    for k in child.children:
        k_id = k.node_id
        offset = child_offsets.get(k_id)
        if offset is None:
            continue

        sboxes = child_sub_boxes_map.get(k_id)
        if sboxes is None:
            continue

        for sbox in sboxes:
            for x in range(sbox.min_x, sbox.max_x + 1):
                for y in range(sbox.min_y, sbox.max_y + 1):
                    for z in range(sbox.min_z, sbox.max_z + 1):
                        cell = Coord3D(x + offset.x, y + offset.y, z + offset.z)
                        if cell not in boundary_port_coords:
                            occupied_cells.add(cell)

    if occupied_cells:
        return decompose_into_orthogonal_sub_boxes(occupied_cells)

    # Fallback to coarse bounding box if no sub-boxes or internal cells could be resolved
    bbox = child_solution.bounding_box
    if bbox is not None:
        return (bbox,)
    return ()


class SplitPairHandler:
    """Classifies split-pair children, anchor alignments, and transverse slack corridors."""

    def __init__(
        self,
        node: SolveTreeNode,
        child_solutions: Mapping[str, NodeSolution],
        graph: LayoutGraph,
    ) -> None:
        self.node = node
        self.child_solutions = child_solutions
        self.graph = graph
        self.rooms = tuple(sorted(node.rooms))
        self.room_set = set(self.rooms)
        self.children = tuple(node.children)

    def group_split_children(self) -> dict[tuple[int, int], list[SolveTreeNode]]:
        """Identify split-pair children and group them by canonical poles (min, max)."""
        split_children_by_poles: dict[tuple[int, int], list[SolveTreeNode]] = {}
        for child in self.children:
            if child.node_id not in self.child_solutions:
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
                c_coords = self.child_solutions[child.node_id].local_coordinates
                if p0 in c_coords and p1 in c_coords:
                    can_pair = (min(p0, p1), max(p0, p1))
                    split_children_by_poles.setdefault(can_pair, []).append(child)

        # Support container-level split metadata
        if self.node.cut_type in (CutType.SPLIT_PAIR, "split_pair") or (
            self.node.node_type in (TreeNodeType.SPLIT_PAIR_CONTAINER, "split_pair_container")
            and "poles" in self.node.metadata
        ):
            raw_node_poles = self.node.metadata.get("poles")
            if raw_node_poles and len(raw_node_poles) == 2:
                np0, np1 = int(raw_node_poles[0]), int(raw_node_poles[1])
                can_pair = (min(np0, np1), max(np0, np1))
                for child in self.children:
                    if child.node_id in self.child_solutions:
                        c_coords = self.child_solutions[child.node_id].local_coordinates
                        if np0 in c_coords and np1 in c_coords:
                            if child not in split_children_by_poles.get(can_pair, []):
                                split_children_by_poles.setdefault(can_pair, []).append(child)

        return split_children_by_poles

    def _find_children_with_parent_corridors(
        self,
        split_children_by_poles: Mapping[tuple[int, int], Sequence[SolveTreeNode]],
    ) -> set[str]:
        """Identify children that connect to parent rooms via boundary port corridor exits."""
        result: set[str] = set()
        for (p0, p1), split_children in split_children_by_poles.items():
            for sc in split_children:
                if sc.node_id not in self.child_solutions:
                    continue
                c_coords = self.child_solutions[sc.node_id].local_coordinates
                for edge in self.graph.edges:
                    if edge.cut or {edge.src, edge.dst} == {p0, p1}:
                        continue
                    if edge.src in self.room_set and edge.dst in self.room_set:
                        continue
                    if (
                        (edge.src in self.room_set and edge.dst in c_coords and edge.dst not in self.room_set)
                        or (edge.dst in self.room_set and edge.src in c_coords and edge.src not in self.room_set)
                    ):
                        try:
                            direction_to_displacement(edge.direction)
                            result.add(sc.node_id)
                            break
                        except ValueError:
                            pass
        return result

    def align_children(
        self,
        split_children_by_poles: Mapping[tuple[int, int], Sequence[SolveTreeNode]],
        room_anchored_children: dict[int, list[str]] | None = None,
        children_with_corridors: set[str] | None = None,
    ) -> tuple[
        list[RoomChildAlignment],
        list[ChildChildAlignment],
        set[frozenset[str]],
        dict[str, set[int]],
        set[str],
    ]:
        """Align split-pair children to parent rooms or sibling children."""
        if room_anchored_children is None:
            room_anchored_children = {}
        if children_with_corridors is None:
            children_with_corridors = self._find_children_with_parent_corridors(split_children_by_poles)

        room_alignments: list[RoomChildAlignment] = []
        child_alignments: list[ChildChildAlignment] = []
        equated_children: set[frozenset[str]] = set()
        gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in self.children}
        aligned_children: set[str] = set()

        for (p0, p1), split_children in split_children_by_poles.items():
            if not split_children:
                continue

            for child in split_children:
                if child.node_id in children_with_corridors:
                    aligned_children.add(child.node_id)

            if len(split_children) > 1 and p0 not in self.room_set and p1 not in self.room_set:
                base_child = split_children[0]
                base_sol = self.child_solutions[base_child.node_id]
                base_pc0 = base_sol.local_coordinates[p0]
                base_pc1 = base_sol.local_coordinates[p1]
                base_vec = (
                    base_pc1.x - base_pc0.x,
                    base_pc1.y - base_pc0.y,
                    base_pc1.z - base_pc0.z,
                )
                for child in split_children[1:]:
                    c_sol = self.child_solutions[child.node_id]
                    pc0 = c_sol.local_coordinates[p0]
                    pc1 = c_sol.local_coordinates[p1]
                    child_vec = (
                        pc1.x - pc0.x,
                        pc1.y - pc0.y,
                        pc1.z - pc0.z,
                    )
                    child_alignments.append(
                        ChildChildAlignment(
                            base_child_id=base_child.node_id,
                            target_child_id=child.node_id,
                            base_coord=base_pc0,
                            target_coord=pc0,
                        )
                    )
                    if base_vec == child_vec:
                        child_alignments.append(
                            ChildChildAlignment(
                                base_child_id=base_child.node_id,
                                target_child_id=child.node_id,
                                base_coord=base_pc1,
                                target_coord=pc1,
                            )
                        )
                    equated_children.add(frozenset({base_child.node_id, child.node_id}))
                for idx1 in range(len(split_children)):
                    for idx2 in range(idx1 + 1, len(split_children)):
                        equated_children.add(
                            frozenset({split_children[idx1].node_id, split_children[idx2].node_id})
                        )

        for _p_anchor, c_list in room_anchored_children.items():
            for idx1 in range(len(c_list)):
                for idx2 in range(idx1 + 1, len(c_list)):
                    equated_children.add(frozenset({c_list[idx1], c_list[idx2]}))

        return (
            room_alignments,
            child_alignments,
            equated_children,
            gateway_anchors,
            aligned_children,
        )

    def build_corridor_active_edges(
        self,
        split_children_by_poles: Mapping[tuple[int, int], Sequence[SolveTreeNode]],
        equated_children: set[frozenset[str]],
    ) -> tuple[list[ActiveEdge], dict[str, set[int]], set[str]]:
        """Build active edges for external parent-to-split, inter-split, and split-pair span corridors."""
        corridor_edges: list[ActiveEdge] = []
        gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in self.children}
        aligned_children: set[str] = set()
        seen_split_corridors: set[frozenset[EndpointRef]] = set()

        for (p0, p1), split_children in split_children_by_poles.items():
            for sc in split_children:
                c_coords = self.child_solutions[sc.node_id].local_coordinates
                for edge in self.graph.edges:
                    if edge.cut:
                        continue
                    if {edge.src, edge.dst} == {p0, p1}:
                        continue
                    if edge.src in self.room_set and edge.dst in self.room_set:
                        continue
                    try:
                        disp = direction_to_displacement(edge.direction)
                    except ValueError:
                        continue

                    # Parent room to split-pair child room
                    if edge.src in self.room_set and edge.dst in c_coords and edge.dst not in self.room_set:
                        u, v = edge.src, edge.dst
                        corridor_key: frozenset[EndpointRef] = frozenset([u, (sc.node_id, v)])
                        if corridor_key not in seen_split_corridors:
                            seen_split_corridors.add(corridor_key)
                            corridor_edges.append(
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
                    elif edge.dst in self.room_set and edge.src in c_coords and edge.src not in self.room_set:
                        u, v = edge.dst, edge.src
                        corridor_key_rev: frozenset[EndpointRef] = frozenset([u, (sc.node_id, v)])
                        if corridor_key_rev not in seen_split_corridors:
                            seen_split_corridors.add(corridor_key_rev)
                            corridor_edges.append(
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
                        for other in self.children:
                            if other.node_id == sc.node_id:
                                continue
                            if not (
                                other.cut_type == CutType.SPLIT_PAIR
                                or isinstance(other.cut_contract, SplitPairCut)
                                or other.metadata.get("cut_type") in (CutType.SPLIT_PAIR, "split_pair")
                            ):
                                continue
                            if frozenset({other.node_id, sc.node_id}) in equated_children:
                                continue
                            if edge.src == edge.dst:
                                continue
                            other_coords = (
                                self.child_solutions[other.node_id].local_coordinates
                                if other.node_id in self.child_solutions
                                else {}
                            )
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
                                corridor_key_ic1: frozenset[EndpointRef] = frozenset([(other.node_id, edge.src), (sc.node_id, edge.dst)])
                                if corridor_key_ic1 not in seen_split_corridors:
                                    seen_split_corridors.add(corridor_key_ic1)
                                    corridor_edges.append(
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
                                corridor_key_ic2: frozenset[EndpointRef] = frozenset([(sc.node_id, edge.src), (other.node_id, edge.dst)])
                                if corridor_key_ic2 not in seen_split_corridors:
                                    seen_split_corridors.add(corridor_key_ic2)
                                    corridor_edges.append(
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

        # Couple split-pair containers whose poles are in parent room set via active edges
        for (p0, p1), split_children in split_children_by_poles.items():
            if p0 in self.room_set and p1 in self.room_set:
                for sc in split_children:
                    if sc.node_id in aligned_children:
                        continue
                    if sc.node_id not in self.child_solutions:
                        continue
                    c_sol = self.child_solutions[sc.node_id]
                    if p0 not in c_sol.local_coordinates or p1 not in c_sol.local_coordinates:
                        continue
                    pc0 = c_sol.local_coordinates[p0]
                    pc1 = c_sol.local_coordinates[p1]
                    pdx = pc1.x - pc0.x
                    pdy = pc1.y - pc0.y
                    pdz = pc1.z - pc0.z
                    max_d = max(abs(pdx), abs(pdy), abs(pdz))
                    L_min = max(1, max_d)
                    if abs(pdx) == max_d:
                        split_dir = (1 if pdx >= 0 else -1, 0, 0)
                    elif abs(pdy) == max_d:
                        split_dir = (0, 1 if pdy >= 0 else -1, 0)
                    else:
                        split_dir = (0, 0, 1 if pdz >= 0 else -1)

                    corridor_edges.append(
                        ActiveEdge(
                            edge_id=f"sp_port_{sc.node_id}_{p0}",
                            src=p0,
                            dst=(sc.node_id, p0),
                            direction=(0, 0, 0),
                            min_length=0,
                            associated_rooms=frozenset({p0}),
                            associated_children=frozenset({sc.node_id}),
                        )
                    )
                    corridor_edges.append(
                        ActiveEdge(
                            edge_id=f"sp_span_{sc.node_id}_{p0}_{p1}",
                            src=p0,
                            dst=p1,
                            direction=split_dir,
                            min_length=L_min,
                            associated_rooms=frozenset({p0, p1}),
                            associated_children=frozenset({sc.node_id}),
                            allow_transverse_slack=True,
                        )
                    )
                    gateway_anchors[sc.node_id].update({p0, p1})
                    aligned_children.add(sc.node_id)

        return corridor_edges, gateway_anchors, aligned_children

    def process(
        self,
        room_anchored_children: dict[int, list[str]] | None = None,
        aligned_children: set[str] | None = None,
    ) -> SplitPairResult:
        """Run split-pair classification, producing alignments, equated sets, and active edges."""
        if room_anchored_children is None:
            room_anchored_children = {}

        grouped = self.group_split_children()
        children_with_corridors = self._find_children_with_parent_corridors(grouped)
        (
            room_alignments,
            child_alignments,
            equated_children,
            gateway_anchors,
            stage_aligned,
        ) = self.align_children(
            split_children_by_poles=grouped,
            room_anchored_children=room_anchored_children,
            children_with_corridors=children_with_corridors,
        )

        corridor_edges, corridor_gw, corridor_aligned = self.build_corridor_active_edges(
            split_children_by_poles=grouped,
            equated_children=equated_children,
        )

        for cid, anchors in corridor_gw.items():
            gateway_anchors[cid].update(anchors)

        all_aligned = set(stage_aligned) | set(corridor_aligned)
        if aligned_children:
            all_aligned.update(aligned_children)

        # Propagate alignment across equated siblings, alignments, and active edges
        changed = True
        while changed:
            changed = False
            for pair in equated_children:
                c1, c2 = tuple(pair)
                if c1 in all_aligned and c2 not in all_aligned:
                    all_aligned.add(c2)
                    changed = True
                elif c2 in all_aligned and c1 not in all_aligned:
                    all_aligned.add(c1)
                    changed = True
            for c_align in child_alignments:
                b_id, t_id = c_align.base_child_id, c_align.target_child_id
                if b_id in all_aligned and t_id not in all_aligned:
                    all_aligned.add(t_id)
                    changed = True
                elif t_id in all_aligned and b_id not in all_aligned:
                    all_aligned.add(b_id)
                    changed = True
            for ce in corridor_edges:
                c_list = list(ce.associated_children)
                if len(c_list) == 2:
                    ca, cb = c_list[0], c_list[1]
                    if ca in all_aligned and cb not in all_aligned:
                        all_aligned.add(cb)
                        changed = True
                    elif cb in all_aligned and ca not in all_aligned:
                        all_aligned.add(ca)
                        changed = True

        return SplitPairResult(
            split_children_by_poles=grouped,
            room_alignments=tuple(room_alignments),
            child_alignments=tuple(child_alignments),
            equated_children=frozenset(equated_children),
            corridor_edges=tuple(corridor_edges),
            gateway_anchors={k: frozenset(v) for k, v in gateway_anchors.items()},
            aligned_children=frozenset(all_aligned),
        )


class GatewayCutHandler:
    """Processes 1-cut children (CutType.ARTICULATION_POINT, CutType.BRIDGE).

    Detects doorway edges connecting parent rooms to child doorway rooms,
    producing gateway room alignments and gateway corridor active edges.
    """

    def __init__(
        self,
        node: SolveTreeNode,
        child_solutions: Mapping[str, NodeSolution],
        graph: LayoutGraph,
    ) -> None:
        self.node = node
        self.child_solutions = child_solutions
        self.graph = graph
        self.rooms = tuple(sorted(node.rooms))
        self.room_set = set(self.rooms)
        self.children = tuple(node.children)

    def process_articulation_points(
        self,
        aligned_children: set[str],
        room_anchored_children: dict[int, list[str]] | None = None,
    ) -> tuple[list[RoomChildAlignment], dict[str, set[int]], set[str], set[frozenset[str]]]:
        """Align 1-cut articulation point children to incident parent rooms."""
        if room_anchored_children is None:
            room_anchored_children = {}

        room_alignments: list[RoomChildAlignment] = []
        gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in self.children}
        new_aligned: set[str] = set()
        equated: set[frozenset[str]] = set()

        for child in self.children:
            if child.node_id in aligned_children or child.node_id in new_aligned:
                continue
            c_sol = self.child_solutions.get(child.node_id)
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

            if a is not None and a in self.room_set and a in c_coords:
                pc = c_coords[a]
                room_alignments.append(
                    RoomChildAlignment(
                        room_vnum=a,
                        child_id=child.node_id,
                        child_coord=pc,
                    )
                )
                gateway_anchors[child.node_id].add(a)
                new_aligned.add(child.node_id)
                for other_cid in room_anchored_children.get(a, []):
                    equated.add(frozenset({other_cid, child.node_id}))
                room_anchored_children.setdefault(a, []).append(child.node_id)

        return room_alignments, gateway_anchors, new_aligned, equated

    def process_doorway_bridges(
        self,
        aligned_children: set[str],
    ) -> tuple[list[ActiveEdge], dict[str, set[int]], set[str]]:
        """Build gateway active edges for 1-cut bridge cuts and doorway corridors."""
        active_edges: list[ActiveEdge] = []
        gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in self.children}
        new_aligned: set[str] = set()

        for child in self.children:
            if child.node_id in aligned_children or child.node_id in new_aligned:
                continue
            c_sol = self.child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates

            contract = child.cut_contract
            if isinstance(contract, BridgeCut):
                u, v = contract.bridge_edge
                dir_val = contract.direction
                weight = contract.weight
                if u in self.room_set and v in c_coords:
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
                        new_aligned.add(child.node_id)
                        continue
                    except ValueError:
                        pass
                elif v in self.room_set and u in c_coords:
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
                        new_aligned.add(child.node_id)
                        continue
                    except ValueError:
                        pass

            for edge in self.graph.edges:
                if edge.cut:
                    continue
                if edge.src in self.room_set and edge.dst in c_coords:
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
                        new_aligned.add(child.node_id)
                        break
                    except ValueError:
                        pass
                elif edge.dst in self.room_set and edge.src in c_coords:
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
                        new_aligned.add(child.node_id)
                        break
                    except ValueError:
                        pass

        return active_edges, gateway_anchors, new_aligned

    def process(
        self,
        aligned_children: set[str] | None = None,
        room_anchored_children: dict[int, list[str]] | None = None,
    ) -> GatewayCutResult:
        """Process both articulation points and doorway bridge connections."""
        if aligned_children is None:
            aligned_children = set()
        if room_anchored_children is None:
            room_anchored_children = {}

        alignments, gw_anchors_art, art_aligned, equated = self.process_articulation_points(
            aligned_children=aligned_children,
            room_anchored_children=room_anchored_children,
        )
        combined_aligned = aligned_children | art_aligned
        edges, gw_anchors_brg, brg_aligned = self.process_doorway_bridges(
            aligned_children=combined_aligned,
        )
        all_gw_anchors: dict[str, set[int]] = {c.node_id: set() for c in self.children}
        for k, v in gw_anchors_art.items():
            all_gw_anchors[k].update(v)
        for k, v in gw_anchors_brg.items():
            all_gw_anchors[k].update(v)

        return GatewayCutResult(
            room_alignments=tuple(alignments),
            active_edges=tuple(edges),
            gateway_anchors={k: frozenset(v) for k, v in all_gw_anchors.items()},
            aligned_children=frozenset(art_aligned | brg_aligned),
            equated_children=frozenset(equated),
        )


class InterChildBridgeHandler:
    """Classifies edges connecting two distinct child clusters across the layout graph."""

    def __init__(
        self,
        node: SolveTreeNode,
        child_solutions: Mapping[str, NodeSolution],
        graph: LayoutGraph,
    ) -> None:
        self.node = node
        self.child_solutions = child_solutions
        self.graph = graph
        self.children = tuple(node.children)

    def process(
        self,
        aligned_children: set[str] | None = None,
        equated_children: set[frozenset[str]] | None = None,
    ) -> InterChildBridgeResult:
        """Identify and construct active edges bridging distinct child clusters."""
        if aligned_children is None:
            aligned_children = set()
        if equated_children is None:
            equated_children = set()

        active_edges: list[ActiveEdge] = []
        new_aligned: set[str] = set()

        if len(self.children) <= 1:
            return InterChildBridgeResult(
                active_edges=(),
                aligned_children=frozenset(),
            )

        current_aligned = set(aligned_children)
        for child in self.children:
            if child.node_id in current_aligned:
                continue
            c_sol = self.child_solutions.get(child.node_id)
            if not c_sol:
                continue
            c_coords = c_sol.local_coordinates

            contract = child.cut_contract
            if isinstance(contract, BridgeCut):
                u, v = contract.bridge_edge
                for other in self.children:
                    if other.node_id == child.node_id or other.node_id not in current_aligned:
                        continue
                    if frozenset({other.node_id, child.node_id}) in equated_children:
                        continue
                    other_coords = (
                        self.child_solutions[other.node_id].local_coordinates
                        if other.node_id in self.child_solutions
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
                            current_aligned.add(child.node_id)
                            new_aligned.add(child.node_id)
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
                            current_aligned.add(child.node_id)
                            new_aligned.add(child.node_id)
                            break
                        except ValueError:
                            pass

            if child.node_id in current_aligned:
                continue

            for edge in self.graph.edges:
                if edge.cut:
                    continue
                for other in self.children:
                    if other.node_id == child.node_id or other.node_id not in current_aligned:
                        continue
                    if frozenset({other.node_id, child.node_id}) in equated_children:
                        continue
                    other_coords = (
                        self.child_solutions[other.node_id].local_coordinates
                        if other.node_id in self.child_solutions
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
                            current_aligned.add(child.node_id)
                            new_aligned.add(child.node_id)
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
                            current_aligned.add(child.node_id)
                            new_aligned.add(child.node_id)
                            break
                        except ValueError:
                            pass
                if child.node_id in current_aligned:
                    break

        return InterChildBridgeResult(
            active_edges=tuple(active_edges),
            aligned_children=frozenset(new_aligned),
        )


def _build_internal_room_edges(
    graph: LayoutGraph,
    rooms: Sequence[int],
) -> list[ActiveEdge]:
    """Build active edges between parent rooms from uncut graph edges."""
    room_set = set(rooms)
    active_edges: list[ActiveEdge] = []
    if not rooms:
        return active_edges

    for e in graph.edges:
        if not e.cut and e.src in room_set and e.dst in room_set and e.src != e.dst:
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

    return active_edges


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
    boundary_ports = frozenset(get_child_boundary_ports(node))

    aligned_children: set[str] = set()
    if not rooms and child_ids:
        aligned_children.add(child_ids[0])

    gateway_anchors: dict[str, set[int]] = {c.node_id: set() for c in children}
    room_anchored_children: dict[int, list[str]] = {}

    # 1. Split-pair children classification & active edges
    split_handler = SplitPairHandler(node=node, child_solutions=child_solutions, graph=graph)
    split_res = split_handler.process(
        room_anchored_children=room_anchored_children,
        aligned_children=aligned_children,
    )

    aligned_children.update(split_res.aligned_children)
    for cid, anchors in split_res.gateway_anchors.items():
        gateway_anchors[cid].update(anchors)

    equated_children: set[frozenset[str]] = set(split_res.equated_children)

    # 2. 1-cut articulation points and gateway doorway bridges
    gateway_handler = GatewayCutHandler(node=node, child_solutions=child_solutions, graph=graph)
    gw_res = gateway_handler.process(
        aligned_children=aligned_children,
        room_anchored_children=room_anchored_children,
    )

    aligned_children.update(gw_res.aligned_children)
    for cid, anchors in gw_res.gateway_anchors.items():
        gateway_anchors[cid].update(anchors)
    equated_children.update(gw_res.equated_children)

    # 3. Inter-child bridges
    bridge_handler = InterChildBridgeHandler(node=node, child_solutions=child_solutions, graph=graph)
    bridge_res = bridge_handler.process(
        aligned_children=aligned_children,
        equated_children=equated_children,
    )
    aligned_children.update(bridge_res.aligned_children)

    # 4. Direct internal room edges
    direct_edges = _build_internal_room_edges(
        graph=graph,
        rooms=rooms,
    )

    # Combine active edges in deterministic order
    active_edges: list[ActiveEdge] = [
        *split_res.active_edges,
        *gw_res.active_edges,
        *bridge_res.active_edges,
        *direct_edges,
    ]

    # 5. Keep-out sub-boxes for children
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
        room_alignments=(*split_res.room_alignments, *gw_res.room_alignments),
        child_alignments=split_res.child_alignments,
        gateway_anchors={k: frozenset(v) for k, v in gateway_anchors.items()},
        aligned_children=frozenset(aligned_children),
        equated_children=frozenset(equated_children),
        child_sub_boxes=child_sub_boxes,
        graph=graph,
    )


SplitPairChildClassifier = SplitPairHandler
ArticulationPointHandler = GatewayCutHandler
BridgeCutHandler = GatewayCutHandler

__all__ = [
    "ActiveEdge",
    "ArticulationPointHandler",
    "BridgeCutHandler",
    "ChildChildAlignment",
    "ClusterInventory",
    "EndpointRef",
    "GatewayCutHandler",
    "GatewayCutResult",
    "InterChildBridgeHandler",
    "InterChildBridgeResult",
    "RoomChildAlignment",
    "SplitPairChildClassifier",
    "SplitPairHandler",
    "SplitPairResult",
    "derive_child_sub_boxes",
    "get_child_boundary_ports",
    "inventory_cluster",
]
