"""Pure computational geometry primitives and clearance invariants."""

from __future__ import annotations

from romutil.layout.geometry.intersections import (
    point_in_segment_interior,
    segment_box_penetration,
    segment_room_penetration,
    segments_intersect_3d,
)
from romutil.layout.geometry.primitives import (
    BoundingBox3D,
    Segment3D,
)

__all__ = [
    "BoundingBox3D",
    "Segment3D",
    "point_in_segment_interior",
    "segment_box_penetration",
    "segment_room_penetration",
    "segments_intersect_3d",
]
