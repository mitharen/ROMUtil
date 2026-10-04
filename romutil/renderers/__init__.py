"""Unified map renderer architecture and format dispatchers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type, overload

from romutil.models import AreaHeader, Room
from romutil.renderers.base import BaseRenderer, adapt_legacy_rdb
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


@overload
def render_map(
    graph_or_rdb: WorldGraph,
    layout_or_output_path: SolvedLayout,
    output_path_or_fmt: Path | str,
    options_or_header: RenderOptions | None = None,
    fmt: str | None = None,
    *,
    header: AreaHeader | None = None,
    options: RenderOptions | None = None,
    **extra_options: Any,
) -> Path:
    ...


@overload
def render_map(
    graph_or_rdb: dict[int, Any] | list[Any],
    layout_or_output_path: str | Path,
    output_path_or_fmt: Any = None,
    options_or_header: Any = None,
    fmt: str | None = None,
    *,
    header: AreaHeader | None = None,
    options: RenderOptions | None = None,
    **extra_options: Any,
) -> Path:
    ...


def render_map(
    graph_or_rdb: Any,
    layout_or_output_path: Any,
    output_path_or_fmt: Any = None,
    options_or_header: Any = None,
    fmt: str | None = None,
    *,
    header: AreaHeader | None = None,
    options: RenderOptions | None = None,
    **extra_options: Any,
) -> Path:
    """Dispatch rendering of WorldGraph/SolvedLayout or legacy room database to renderer format.

    Primary signature: render_map(graph, layout, output_path, options=None, fmt=None)
    Legacy signature: render_map(rdb, output_path, fmt=None, header=None, **options)

    If fmt is not specified, it is inferred from output_path's file extension.
    """
    if isinstance(graph_or_rdb, WorldGraph):
        graph = graph_or_rdb
        if not isinstance(layout_or_output_path, SolvedLayout):
            raise TypeError(f"Expected SolvedLayout as second argument, got {type(layout_or_output_path).__name__}")
        layout = layout_or_output_path
        if output_path_or_fmt is None:
            raise TypeError("output_path must be specified")
        out_path = Path(output_path_or_fmt)
        render_opts = options if options is not None else (
            options_or_header if isinstance(options_or_header, RenderOptions) else None
        )
        actual_fmt = fmt
        if actual_fmt is None:
            actual_fmt = out_path.suffix.lstrip(".")
            if not actual_fmt:
                actual_fmt = "svg"
        renderer = get_renderer(actual_fmt)
        return renderer.render(graph, layout, out_path, render_opts, **extra_options)
    else:
        # Legacy signature: render_map(rdb, output_path, fmt=None, header=None, **options)
        rdb = graph_or_rdb
        out_path = Path(
            layout_or_output_path
            if isinstance(layout_or_output_path, (str, Path))
            else str(layout_or_output_path)
        )
        actual_fmt = fmt
        if (
            actual_fmt is None
            and isinstance(output_path_or_fmt, str)
            and not output_path_or_fmt.startswith("/")
            and not output_path_or_fmt.startswith(".")
        ):
            actual_fmt = output_path_or_fmt
        if actual_fmt is None:
            actual_fmt = out_path.suffix.lstrip(".")
            if not actual_fmt:
                actual_fmt = "svg"

        legacy_header = header if header is not None else (
            options_or_header
            if isinstance(options_or_header, (AreaHeader, tuple, list))
            or hasattr(options_or_header, "filename")
            or hasattr(options_or_header, "name")
            else (
                output_path_or_fmt
                if isinstance(output_path_or_fmt, (AreaHeader, tuple, list))
                or hasattr(output_path_or_fmt, "filename")
                or hasattr(output_path_or_fmt, "name")
                else None
            )
        )
        renderer = get_renderer(actual_fmt)
        return renderer.render(rdb, out_path, header=legacy_header, **extra_options)


__all__ = [
    "BaseRenderer",
    "SVGRenderer",
    "JSONRenderer",
    "HTMLRenderer",
    "_DynamicPalette",
    "adapt_legacy_rdb",
    "build_area_json",
    "export_json",
    "generate_html_viewer",
    "export_html",
    "HTML_TEMPLATE",
    "render_map",
    "get_renderer",
    "RENDERERS",
]
