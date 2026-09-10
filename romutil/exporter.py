"""Backward-compatibility shim re-exporting exporter functions from romutil.renderers."""

from __future__ import annotations

from romutil.renderers import (
    HTML_TEMPLATE,
    build_area_json,
    export_html,
    export_json,
    generate_html_viewer,
)

__all__ = [
    "HTML_TEMPLATE",
    "build_area_json",
    "export_html",
    "export_json",
    "generate_html_viewer",
]
