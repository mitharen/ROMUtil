"""Decoupled component interfaces for parser, solver, and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TextIO, runtime_checkable

from romutil.types import LayoutOptions, RenderOptions, SolvedLayout, WorldGraph


@runtime_checkable
class BaseParser(Protocol):
    """Protocol for area parsers constructing WorldGraph instances."""

    def parse(self, source: Path | str | TextIO) -> WorldGraph:
        """Parse raw MUD area content from a path, string, or stream into a WorldGraph."""
        ...


@runtime_checkable
class BaseLayoutSolver(Protocol):
    """Protocol for layout solvers computing 3D coordinates for a WorldGraph."""

    def solve(self, graph: WorldGraph, options: LayoutOptions | None = None) -> SolvedLayout:
        """Solve 3D spatial layout coordinates for the given WorldGraph."""
        ...


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
