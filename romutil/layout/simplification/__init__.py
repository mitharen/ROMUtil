"""Layout simplification primitives for the hierarchical layout solver pipeline."""

from __future__ import annotations

from romutil.layout.simplification.hallways import (
    HallwayCompactor,
    expand_corridor_coordinates,
)

__all__ = [
    "HallwayCompactor",
    "expand_corridor_coordinates",
]
