"""Base protocol definition for map renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from romutil.models import AreaHeader, Room


@runtime_checkable
class BaseRenderer(Protocol):
    """Protocol defining the interface for map renderers."""

    def render(
        self,
        rdb: dict[int, Room],
        output_path: str | Path,
        header: AreaHeader | None = None,
        **options: Any,
    ) -> Path:
        """Render the room database to the specified output path."""
        ...
