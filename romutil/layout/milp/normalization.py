"""Stage 5: Coordinate normalization and NodeSolution construction for cluster MILP."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import NodeSolution, SolveTreeNode
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.types import Coord3D


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
) -> NodeSolution:
    """Normalize solved room coordinates and child offsets to origin (0, 0, 0).

    Computes the minimum spatial bounds across all constituent rooms and child bounding
    boxes, applies the translational shift, and returns the immutable NodeSolution.

    Args:
        extraction: SolvedClusterExtraction containing raw coordinates and child offsets.
        node: SolveTreeNode representing the solved cluster.
        children: Sequence of child solve tree nodes.
        child_solutions: Mapping of child node IDs to already-solved NodeSolutions.

    Returns:
        Immutable NodeSolution containing normalized coordinates and child offsets.
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

    metadata: dict[str, Any] = {
        "child_offsets": norm_child_offsets,
        "status": extraction.solver_status,
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
