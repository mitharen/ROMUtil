"""Base protocol definition for map renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from romutil.types import RenderOptions, SolvedLayout, WorldGraph


@runtime_checkable
class BaseRenderer(Protocol):
    """Protocol for renderers outputting solved layouts to visual or structured formats."""

    def render(
        self,
        graph: WorldGraph,
        layout: SolvedLayout,
        output_path: Path | str,
        options: RenderOptions | None = None,
    ) -> Path:
        """Render a solved layout of a WorldGraph to output_path."""
        ...


__all__ = [
    "BaseRenderer",
]
