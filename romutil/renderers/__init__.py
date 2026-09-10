"""Unified map renderer architecture and format dispatchers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from romutil.models import AreaHeader, Room
from romutil.renderers.base import BaseRenderer
from romutil.renderers.html import (
    HTML_TEMPLATE,
    HTMLRenderer,
    export_html,
    generate_html_viewer,
)
from romutil.renderers.json import JSONRenderer, build_area_json, export_json
from romutil.renderers.svg import Plotter, SVGRenderer, _DynamicPalette

RENDERERS: dict[str, Type[BaseRenderer]] = {
    "svg": SVGRenderer,
    "json": JSONRenderer,
    "html": HTMLRenderer,
}


def get_renderer(format_or_renderer: str | BaseRenderer | Type[BaseRenderer]) -> BaseRenderer:
    """
    Look up and instantiate a renderer by format string or return an existing renderer.

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
    rdb: dict[int, Room],
    output_path: str | Path,
    fmt: str | None = None,
    header: AreaHeader | None = None,
    **options: Any,
) -> Path:
    """
    Dispatch rendering of the room database to the appropriate renderer format.

    If fmt is not specified, it is inferred from output_path's file extension.
    """
    out_path = Path(output_path)
    if fmt is None:
        fmt = out_path.suffix.lstrip(".")
        if not fmt:
            fmt = "svg"
    renderer = get_renderer(fmt)
    return renderer.render(rdb, out_path, header=header, **options)


__all__ = [
    "BaseRenderer",
    "SVGRenderer",
    "JSONRenderer",
    "HTMLRenderer",
    "Plotter",
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
