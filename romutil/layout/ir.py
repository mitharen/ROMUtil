"""Intermediate representations for the hierarchical layout solver pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Collection, Mapping, Sequence

from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.models import Room
from romutil.types import Coord3D, WorldGraph


class TreeNodeType(str, Enum):
    """Canonical classification for hierarchical solve tree nodes."""

    WORLD_ROOT = "world_root"
    RIGID_CLUSTER = "rigid_cluster"
    SERIES_LOOP = "series_loop"
    SPLIT_PAIR_CONTAINER = "split_pair_container"
    BLOCK = "block"
    BICONNECTED_BLOCK = "biconnected_block"
    EMPTY = "empty"


class CutType(str, Enum):
    """Topological cut classification defining boundary interfaces between solve tree nodes."""

    ARTICULATION_POINT = "articulation_point"
    BRIDGE = "bridge"
    SPLIT_PAIR = "split_pair"
    WARP_CYCLE = "warp_cycle"


@dataclass(frozen=True)
class SplitPairCut:
    """Cut contract for a 2-cut split-pair separating a component at two poles."""

    poles: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.poles, tuple):
            object.__setattr__(self, "poles", tuple(self.poles))


@dataclass(frozen=True)
class ArticulationPointCut:
    """Cut contract for a 1-cut articulation point vertex."""

    anchor_vnum: int


@dataclass(frozen=True)
class BridgeCut:
    """Cut contract for a 1-cut bridge edge."""

    bridge_edge: tuple[int, int]
    direction: int
    weight: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.bridge_edge, tuple):
            object.__setattr__(self, "bridge_edge", tuple(self.bridge_edge))


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

    rooms: Mapping[int, Room]
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
    node_type: TreeNodeType | str
    rooms: frozenset[int] | Collection[int]
    children: Sequence[SolveTreeNode] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.node_type, str) and not isinstance(self.node_type, TreeNodeType):
            try:
                object.__setattr__(self, "node_type", TreeNodeType(self.node_type))
            except ValueError:
                pass
        if not isinstance(self.rooms, frozenset):
            object.__setattr__(self, "rooms", frozenset(self.rooms))
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))

    @property
    def cut_type(self) -> CutType | None:
        """Inspect and return the CutType for this node if defined in metadata or contract."""
        contract = self.cut_contract
        if contract is not None:
            if isinstance(contract, SplitPairCut):
                return CutType.SPLIT_PAIR
            if isinstance(contract, ArticulationPointCut):
                return CutType.ARTICULATION_POINT
            if isinstance(contract, BridgeCut):
                return CutType.BRIDGE
        raw_cut = self.metadata.get("cut_type")
        if isinstance(raw_cut, CutType):
            return raw_cut
        if isinstance(raw_cut, str):
            try:
                return CutType(raw_cut)
            except ValueError:
                pass
        if "poles" in self.metadata:
            return CutType.SPLIT_PAIR
        if "anchor_vnum" in self.metadata:
            return CutType.ARTICULATION_POINT
        if "bridge_edge" in self.metadata:
            return CutType.BRIDGE
        return None

    @property
    def cut_contract(self) -> SplitPairCut | ArticulationPointCut | BridgeCut | None:
        """Inspect and return the structured cut contract for this node."""
        contract = self.metadata.get("cut_contract")
        if isinstance(contract, (SplitPairCut, ArticulationPointCut, BridgeCut)):
            return contract
        raw_cut = self.metadata.get("cut_type")
        if raw_cut == CutType.SPLIT_PAIR or raw_cut == "split_pair" or "poles" in self.metadata:
            poles = self.metadata.get("poles")
            if poles is not None and len(poles) == 2:
                return SplitPairCut(poles=(int(poles[0]), int(poles[1])))
        if raw_cut == CutType.ARTICULATION_POINT or raw_cut == "articulation_point" or "anchor_vnum" in self.metadata:
            anchor = self.metadata.get("anchor_vnum")
            if anchor is not None:
                return ArticulationPointCut(anchor_vnum=int(anchor))
        if raw_cut == CutType.BRIDGE or raw_cut == "bridge" or "bridge_edge" in self.metadata:
            edge = self.metadata.get("bridge_edge")
            if edge is not None and len(edge) == 2:
                return BridgeCut(
                    bridge_edge=(int(edge[0]), int(edge[1])),
                    direction=int(self.metadata.get("direction", 0)),
                    weight=int(self.metadata.get("weight", 1)),
                )
        return None


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
    sub_boxes: tuple[BoundingBox3D, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    port_coordinates: Mapping[int, Coord3D] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sub_boxes, tuple):
            object.__setattr__(self, "sub_boxes", tuple(self.sub_boxes))
        if not isinstance(self.port_coordinates, dict):
            object.__setattr__(self, "port_coordinates", dict(self.port_coordinates))

    @property
    def coordinates(self) -> Mapping[int, Coord3D]:
        """Return canonical coordinates strictly for owned rooms."""
        return self.local_coordinates

    @property
    def bounding_box(self) -> BoundingBox3D | None:
        """Compute the minimal enclosing 3D bounding box over sub-boxes, local coordinates, or metadata."""
        if self.sub_boxes:
            return BoundingBox3D(
                min_x=min(b.min_x for b in self.sub_boxes),
                max_x=max(b.max_x for b in self.sub_boxes),
                min_y=min(b.min_y for b in self.sub_boxes),
                max_y=max(b.max_y for b in self.sub_boxes),
                min_z=min(b.min_z for b in self.sub_boxes),
                max_z=max(b.max_z for b in self.sub_boxes),
            )
        if "bounding_box" in self.metadata and isinstance(self.metadata["bounding_box"], BoundingBox3D):
            return self.metadata["bounding_box"]
        if self.local_coordinates:
            return BoundingBox3D.from_coordinates(self.local_coordinates.values())
        return None


@dataclass(frozen=True)
class TreeSolution:
    """Combined solutions for all solve tree nodes."""

    node_solutions: Mapping[str, NodeSolution]
