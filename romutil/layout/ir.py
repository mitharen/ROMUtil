"""Intermediate representations for the hierarchical layout solver pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Mapping, Sequence

from romutil.models import RoomDef
from romutil.types import Coord3D, WorldGraph


@dataclass(frozen=True)
class LayoutEdge:
    """Directed edge in a LayoutGraph connecting room vnums."""

    src: int
    dst: int
    direction: int
    weight: int = 1
    cut: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayoutGraph:
    """Internal graph representation of room topologies and layout edges."""

    rooms: Mapping[int, RoomDef]
    edges: Sequence[LayoutEdge]

    def __post_init__(self) -> None:
        if not isinstance(self.edges, tuple):
            object.__setattr__(self, "edges", tuple(self.edges))

    @classmethod
    def from_world_graph(cls, graph: WorldGraph) -> LayoutGraph:
        """Convert a WorldGraph into a LayoutGraph."""
        edges: list[LayoutEdge] = []
        for room in graph.rooms.values():
            for exit_def in room.exits:
                edges.append(
                    LayoutEdge(
                        src=room.vnum,
                        dst=exit_def.dst_vnum,
                        direction=exit_def.direction,
                    )
                )
        return cls(rooms=dict(graph.rooms), edges=tuple(edges))


@dataclass(frozen=True)
class CorridorSegment:
    """Collapsed corridor segment between two endpoint room vnums."""

    start_vnum: int
    end_vnum: int
    direction: int
    intermediate_vnums: Sequence[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.intermediate_vnums, tuple):
            object.__setattr__(self, "intermediate_vnums", tuple(self.intermediate_vnums))


@dataclass(frozen=True)
class SimplifiedGraph:
    """Graph simplified by collapsing linear corridors and severing chords."""

    graph: LayoutGraph
    corridors: Sequence[CorridorSegment] = field(default_factory=tuple)
    severed_chords: Sequence[LayoutEdge] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.corridors, tuple):
            object.__setattr__(self, "corridors", tuple(self.corridors))
        if not isinstance(self.severed_chords, tuple):
            object.__setattr__(self, "severed_chords", tuple(self.severed_chords))


@dataclass(frozen=True)
class SolveTreeNode:
    """Node in a hierarchical solve tree representing a cluster or component."""

    node_id: str
    node_type: str
    rooms: frozenset[int] | Collection[int]
    children: Sequence[SolveTreeNode] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rooms, frozenset):
            object.__setattr__(self, "rooms", frozenset(self.rooms))
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))


@dataclass(frozen=True)
class SolveTree:
    """Hierarchical decomposition solve tree."""

    root: SolveTreeNode
    components: Sequence[SolveTreeNode] = field(default_factory=tuple)
    graph: LayoutGraph | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple):
            object.__setattr__(self, "components", tuple(self.components))


@dataclass(frozen=True)
class NodeSolution:
    """Solved 3D spatial layout coordinates for a single solve tree node."""

    node_id: str
    local_coordinates: Mapping[int, Coord3D]
    bounding_box: tuple[Coord3D, Coord3D] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TreeSolution:
    """Combined solutions for all solve tree nodes."""

    node_solutions: Mapping[str, NodeSolution]
