"""Decoupled domain value objects and data contracts for ROMUtil."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from romutil.models import AreaData, AreaHeader, ExitDef, RoomDef


@dataclass(frozen=True, slots=True)
class Coord3D:
    """Immutable 3D integer coordinate."""

    x: int
    y: int
    z: int


class SolverStatus(str, Enum):
    """Status outcomes from a layout solver run."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class LayoutDiagnostics:
    """Diagnostics and telemetry from a layout solve execution."""

    status: SolverStatus
    duration_seconds: float
    defect_count: int = 0
    details: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SolvedLayout:
    """Solved 3D spatial layout associating room vnums to Coord3D positions."""

    coordinates: Mapping[int, Coord3D]
    diagnostics: LayoutDiagnostics = field(
        default_factory=lambda: LayoutDiagnostics(
            status=SolverStatus.OPTIMAL, duration_seconds=0.0
        )
    )


@dataclass(frozen=True)
class LayoutOptions:
    """Configuration options for layout solvers."""

    timeout_seconds: int = 60
    component_padding: int = 2


@dataclass(frozen=True)
class RenderOptions:
    """Configuration options for renderers."""

    split_levels: bool = False
    scale: float = 1.0
    theme: str = "default"


@dataclass(frozen=True)
class WorldGraph:
    """Immutable topology representation decoupling parsing from layout solving."""

    rooms: Mapping[int, RoomDef]
    exits: Sequence[ExitDef]
    header: AreaHeader | None = None

    @classmethod
    def from_area_data(cls, area: AreaData) -> WorldGraph:
        """Construct a WorldGraph from an AreaData domain model."""
        rooms = {r.vnum: r for r in area.rooms}
        exits = tuple(e for r in area.rooms for e in r.exits)
        return cls(rooms=rooms, exits=exits, header=area.header)
