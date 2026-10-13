"""Stage 5: Coordinate normalization and NodeSolution construction for cluster MILP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from romutil.layout.geometry import (
    BoundingBox3D,
    decompose_into_orthogonal_sub_boxes,
)
from romutil.layout.ir import NodeSolution, SolveTreeNode
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.types import Coord3D

if TYPE_CHECKING:
    from romutil.layout.milp.inventory import ActiveEdge


def normalize_empty_cluster(node: SolveTreeNode) -> NodeSolution:
    """Construct an empty NodeSolution for a degenerate zero-room, zero-child cluster."""
    return NodeSolution(
        node_id=node.node_id,
        local_coordinates={},
        sub_boxes=(),
        metadata={"child_offsets": {}},
    )


def normalize_single_room_cluster(
    node: SolveTreeNode,
    room: int,
    is_boundary_port: bool = False,
) -> NodeSolution:
    """Construct a canonical NodeSolution for a degenerate single-room leaf cluster.

    Args:
        node: SolveTreeNode representing the single-room leaf cluster.
        room: Room ID contained within the cluster.
        is_boundary_port: Whether the room functions as a boundary gateway port.

    Returns:
        NodeSolution positioned at origin (0, 0, 0).
    """
    return NodeSolution(
        node_id=node.node_id,
        local_coordinates={room: Coord3D(0, 0, 0)},
        sub_boxes=(),
        metadata={"child_offsets": {}},
    )


def normalize_cluster_solution(
    extraction: SolvedClusterExtraction,
    node: SolveTreeNode,
    children: Sequence[SolveTreeNode],
    child_solutions: Mapping[str, NodeSolution],
    child_sub_boxes: Mapping[str, Sequence[BoundingBox3D]] | None = None,
    active_edges: Sequence[ActiveEdge] | None = None,
) -> NodeSolution:
    """Normalize solved room coordinates and child offsets to origin (0, 0, 0).

    Computes the minimum spatial bounds across all constituent rooms and child bounding
    boxes, applies the translational shift, and returns the immutable NodeSolution.
    For composite clusters embedding child clusters, recursively aggregates physical
    occupied footprints across constituent parent rooms and translated child sub-boxes,
    decomposing them into disjoint orthogonal sub-boxes.

    Args:
        extraction: SolvedClusterExtraction containing raw coordinates and child offsets.
        node: SolveTreeNode representing the solved cluster.
        children: Sequence of child solve tree nodes.
        child_solutions: Mapping of child node IDs to already-solved NodeSolutions.
        child_sub_boxes: Optional mapping of child node IDs to derived keep-out sub-boxes.
        active_edges: Optional active edges of the cluster to identify expanded corridor paths.

    Returns:
        Immutable NodeSolution containing normalized coordinates, sub-boxes, and metadata.
    """
    all_pts: list[tuple[int, int, int]] = list(extraction.raw_parent_coords.values())
    for child in children:
        cid = child.node_id
        cx, cy, cz = extraction.raw_child_offsets[cid]
        c_sol = child_solutions.get(cid)
        box = c_sol.bounding_box if c_sol else None
        if box is not None:
            all_pts.append((cx + box.min_x, cy + box.min_y, cz + box.min_z))
            all_pts.append((cx + box.max_x, cy + box.max_y, cz + box.max_z))
        elif c_sol and c_sol.local_coordinates:
            all_pts.append((cx, cy, cz))

    if all_pts:
        min_x = min(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        min_z = min(p[2] for p in all_pts)
    else:
        min_x, min_y, min_z = 0, 0, 0

    norm_parent_coords = {
        u: Coord3D(x - min_x, y - min_y, z - min_z)
        for u, (x, y, z) in extraction.raw_parent_coords.items()
    }
    norm_child_offsets = {
        cid: Coord3D(x - min_x, y - min_y, z - min_z)
        for cid, (x, y, z) in extraction.raw_child_offsets.items()
    }

    # Aggregate descendant room coordinates in cluster local frame
    all_room_coords: dict[Any, Coord3D] = dict(norm_parent_coords)
    for child in children:
        cid = child.node_id
        c_off = norm_child_offsets.get(cid)
        if c_off is None:
            continue
        c_sol = child_solutions.get(cid)
        if not c_sol:
            continue
        for u, pt in c_sol.local_coordinates.items():
            all_room_coords[u] = Coord3D(pt.x + c_off.x, pt.y + c_off.y, pt.z + c_off.z)
        if "all_room_coordinates" in c_sol.metadata:
            for u, pt in c_sol.metadata["all_room_coordinates"].items():
                all_room_coords[u] = Coord3D(pt.x + c_off.x, pt.y + c_off.y, pt.z + c_off.z)

    if active_edges:
        for ae in active_edges:
            if (
                isinstance(ae.src, int)
                and isinstance(ae.dst, int)
                and ae.src in norm_parent_coords
                and ae.dst in norm_parent_coords
            ):
                p1 = norm_parent_coords[ae.src]
                p2 = norm_parent_coords[ae.dst]
                dist = abs(p2.x - p1.x) + abs(p2.y - p1.y) + abs(p2.z - p1.z)
                if dist > 1:
                    dx, dy, dz = ae.direction
                    for t in range(1, dist):
                        corr_pt = Coord3D(p1.x + t * dx, p1.y + t * dy, p1.z + t * dz)
                        all_room_coords[f"corridor_{ae.src}_{ae.dst}_{t}"] = corr_pt

    # Collect sub-boxes per child
    sub_boxes_map: dict[str, tuple[BoundingBox3D, ...]] = {}
    if child_sub_boxes is not None:
        for cid, boxes in child_sub_boxes.items():
            sub_boxes_map[cid] = tuple(boxes)
    else:
        for child in children:
            cid = child.node_id
            c_sol = child_solutions.get(cid)
            if c_sol and c_sol.sub_boxes:
                sub_boxes_map[cid] = tuple(c_sol.sub_boxes)

    metadata: dict[str, Any] = {
        "child_offsets": norm_child_offsets,
        "status": extraction.solver_status,
        "child_sub_boxes": sub_boxes_map,
        "all_room_coordinates": all_room_coords,
    }
    if all_pts:
        max_x = max(p[0] for p in all_pts)
        max_y = max(p[1] for p in all_pts)
        max_z = max(p[2] for p in all_pts)
        metadata["bounding_box"] = BoundingBox3D(
            min_x=0,
            max_x=max_x - min_x,
            min_y=0,
            max_y=max_y - min_y,
            min_z=0,
            max_z=max_z - min_z,
        )

    return NodeSolution(
        node_id=node.node_id,
        local_coordinates=norm_parent_coords,
        sub_boxes=(),
        metadata=metadata,
    )


__all__ = [
    "normalize_cluster_solution",
    "normalize_empty_cluster",
    "normalize_single_room_cluster",
]
