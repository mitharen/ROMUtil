"""Unified map renderer architecture and format dispatchers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from romutil.renderers.base import BaseRenderer
from romutil.renderers.html import (
    HTML_TEMPLATE,
    HTMLRenderer,
    export_html,
    generate_html_viewer,
)
from romutil.renderers.json import JSONRenderer, build_area_json, export_json
from romutil.renderers.svg import SVGRenderer, _DynamicPalette
from romutil.types import RenderOptions, SolvedLayout, WorldGraph

RENDERERS: dict[str, Type[BaseRenderer]] = {
    "svg": SVGRenderer,
    "json": JSONRenderer,
    "html": HTMLRenderer,
}


def get_renderer(format_or_renderer: str | BaseRenderer | Type[BaseRenderer]) -> BaseRenderer:
    """Look up and instantiate a renderer by format string or return an existing renderer.

    Raises ValueError if format string is unsupported.
    """
    if isinstance(format_or_renderer, type):
        if issubclass(format_or_renderer, BaseRenderer):
            return format_or_renderer()
        raise TypeError(f"Class {format_or_renderer.__name__} does not implement BaseRenderer")
    if isinstance(format_or_renderer, BaseRenderer):
        return format_or_renderer
    if isinstance(format_or_renderer, str):
        key = format_or_renderer.lower().lstrip(".")
        if key not in RENDERERS:
            supported = ", ".join(sorted(RENDERERS.keys()))
            raise ValueError(
                f"Unsupported renderer format '{format_or_renderer}'. Supported formats: {supported}"
            )
        renderer_cls = RENDERERS[key]
        return renderer_cls()
    raise TypeError(f"Expected format string or BaseRenderer, got {type(format_or_renderer).__name__}")


def render_map(
    graph: WorldGraph,
    layout: SolvedLayout,
    output_path: Path | str,
    options: RenderOptions | None = None,
    fmt: str | None = None,
    **extra_options: Any,
) -> Path:
    """Dispatch rendering of WorldGraph and SolvedLayout to renderer format.

    If fmt is not specified, it is inferred from output_path's file extension.
    """
    if not isinstance(graph, WorldGraph):
        raise TypeError(f"Expected WorldGraph as first argument, got {type(graph).__name__}")
    if not isinstance(layout, SolvedLayout):
        raise TypeError(f"Expected SolvedLayout as second argument, got {type(layout).__name__}")
    if output_path is None:
        raise TypeError("output_path must be specified")
    out_path = Path(output_path)
    actual_fmt = fmt
    if actual_fmt is None:
        actual_fmt = out_path.suffix.lstrip(".")
        if not actual_fmt:
            actual_fmt = "svg"
    renderer = get_renderer(actual_fmt)
    return renderer.render(graph, layout, out_path, options, **extra_options)


__all__ = [
    "BaseRenderer",
    "SVGRenderer",
    "JSONRenderer",
    "HTMLRenderer",
    "_DynamicPalette",
    "build_area_json",
    "export_json",
    "generate_html_viewer",
    "export_html",
    "HTML_TEMPLATE",
    "render_map",
    "get_renderer",
    "RENDERERS",
]
