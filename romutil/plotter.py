"""Backward-compatibility shim re-exporting Plotter from romutil.renderers.svg."""

from __future__ import annotations

from romutil.models import Direction
from romutil.renderers.svg import Plotter, _DynamicPalette

__all__ = [
    "Direction",
    "Plotter",
    "_DynamicPalette",
]
