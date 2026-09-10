"""Standalone interactive HTML/JS map viewer renderer."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from romutil.models import AreaHeader, Room
from romutil.renderers.base import BaseRenderer
from romutil.renderers.json import build_area_json


def _load_html_template() -> str:
    """Load the standalone HTML viewer template asset."""
    try:
        return (
            importlib.resources.files("romutil.templates")
            .joinpath("viewer.html")
            .read_text(encoding="utf-8")
        )
    except Exception:
        fallback_path = Path(__file__).resolve().parent.parent / "templates" / "viewer.html"
        if fallback_path.is_file():
            return fallback_path.read_text(encoding="utf-8")
        raise


HTML_TEMPLATE: str = _load_html_template()


def generate_html_viewer(data: Dict[str, Any], title: Optional[str] = None) -> str:
    """
    Generates a single, self-contained HTML/JS map viewer that renders the map
    data with viewport auto-centering, dual-end elevation gradients for inter-floor
    transitions, inclusive floor filtering, contextual warning tooltips, and an
    incoming exits inspector without requiring external network access, CDNs, or Node.js.
    """
    area_name = data.get('area', {}).get('name') or 'ROM MUD Area'
    area_file = data.get('area', {}).get('file') or 'area.are'
    doc_title = title or f"{area_name} ({area_file}) - ROMUtil Map Viewer"

    bounds = data.get('bounds', {})
    room_count = len(data.get('rooms', []))
    max_x = bounds.get('max_x', 0)
    max_y = bounds.get('max_y', 0)

    # Escape JSON safely for embedding in HTML script tag
    safe_json = json.dumps(data, indent=2, ensure_ascii=False).replace('</', '<\\/')

    content = HTML_TEMPLATE
    content = content.replace('__TITLE__', doc_title)
    content = content.replace('__AREA_NAME__', area_name)
    content = content.replace('__AREA_FILE__', area_file)
    content = content.replace('__ROOM_COUNT__', str(room_count))
    content = content.replace('__MAX_X__', str(max_x))
    content = content.replace('__MAX_Y__', str(max_y))
    content = content.replace('__JSON_DATA__', safe_json)
    return content


def export_html(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    title: Optional[str] = None,
) -> Path:
    """
    Generates and saves the standalone HTML/JS map viewer to disk.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_html_viewer(data, title=title)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


class HTMLRenderer:
    """Renderer generating standalone interactive HTML/JS map viewers conforming to BaseRenderer."""

    def render(
        self,
        rdb: dict[int, Room],
        output_path: str | Path,
        header: AreaHeader | None = None,
        **options: Any,
    ) -> Path:
        title = options.get('title')
        bounds = options.get('bounds')
        data = build_area_json(rdb, area_meta=header, bounds=bounds)
        return export_html(data, output_path, title=title)
