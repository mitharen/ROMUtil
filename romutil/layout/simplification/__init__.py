"""Layout simplification primitives for the hierarchical layout solver pipeline."""

from __future__ import annotations

from romutil.layout.simplification.hallways import (
    HallwayCompactor,
    expand_corridor_coordinates,
)
from romutil.layout.simplification.warps import (
    ChainedSimplifier,
    WarpSeverer,
    direction_to_displacement,
    is_euclidean_satisfiable,
)

__all__ = [
    "ChainedSimplifier",
    "HallwayCompactor",
    "WarpSeverer",
    "direction_to_displacement",
    "expand_corridor_coordinates",
    "is_euclidean_satisfiable",
]
