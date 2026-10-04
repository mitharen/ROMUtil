"""Tests for romutil.interfaces protocols and romutil.types value objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TextIO
import pytest

from romutil.interfaces import BaseLayoutSolver, BaseParser, BaseRenderer
from romutil.models import AreaData, AreaHeader, ExitDef, RoomDef
from romutil.types import (
    Coord3D,
    LayoutDiagnostics,
    LayoutOptions,
    RenderOptions,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


class TestCoord3D:
    """Tests for Coord3D value object."""

    def test_coord3d_creation_and_attributes(self):
        c = Coord3D(x=1, y=-2, z=3)
        assert c.x == 1
        assert c.y == -2
        assert c.z == 3

    def test_coord3d_immutability(self):
        c = Coord3D(x=0, y=0, z=0)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            c.x = 5  # type: ignore[misc]
        with pytest.raises((FrozenInstanceError, AttributeError)):
            c.y = 5  # type: ignore[misc]
        with pytest.raises((FrozenInstanceError, AttributeError)):
            c.z = 5  # type: ignore[misc]

    def test_coord3d_equality_and_hash(self):
        c1 = Coord3D(x=10, y=20, z=30)
        c2 = Coord3D(x=10, y=20, z=30)
        c3 = Coord3D(x=10, y=20, z=31)

        assert c1 == c2
        assert c1 != c3
        assert hash(c1) == hash(c2)

        mapping = {c1: "origin"}
        assert mapping[c2] == "origin"


class TestSolverStatus:
    """Tests for SolverStatus enum."""

    def test_solver_status_values(self):
        assert SolverStatus.OPTIMAL == "optimal"
        assert SolverStatus.FEASIBLE == "feasible"
        assert SolverStatus.TIMEOUT == "timeout"
        assert SolverStatus.INFEASIBLE == "infeasible"
        assert SolverStatus.OPTIMAL.value == "optimal"

    def test_solver_status_members(self):
        statuses = {s.value for s in SolverStatus}
        assert statuses == {"optimal", "feasible", "timeout", "infeasible"}


class TestLayoutDiagnostics:
    """Tests for LayoutDiagnostics value object."""

    def test_layout_diagnostics_defaults(self):
        diag = LayoutDiagnostics(
            status=SolverStatus.OPTIMAL,
            duration_seconds=1.25,
        )
        assert diag.status == SolverStatus.OPTIMAL
        assert diag.duration_seconds == 1.25
        assert diag.defect_count == 0
        assert diag.details == {}

    def test_layout_diagnostics_custom(self):
        diag = LayoutDiagnostics(
            status=SolverStatus.TIMEOUT,
            duration_seconds=60.0,
            defect_count=3,
            details={"room_collisions": 2, "penetrations": 1},
        )
        assert diag.defect_count == 3
        assert diag.details["room_collisions"] == 2

    def test_layout_diagnostics_immutability(self):
        diag = LayoutDiagnostics(
            status=SolverStatus.OPTIMAL,
            duration_seconds=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            diag.defect_count = 1  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            diag.status = SolverStatus.FEASIBLE  # type: ignore[misc]


class TestSolvedLayout:
    """Tests for SolvedLayout value object."""

    def test_solved_layout_creation(self):
        coords = {100: Coord3D(x=0, y=0, z=0), 101: Coord3D(x=1, y=0, z=0)}
        diag = LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.1)
        layout = SolvedLayout(coordinates=coords, diagnostics=diag)

        assert layout.coordinates[100] == Coord3D(x=0, y=0, z=0)
        assert layout.diagnostics.status == SolverStatus.OPTIMAL

    def test_solved_layout_immutability(self):
        layout = SolvedLayout(
            coordinates={},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.1),
        )
        with pytest.raises(FrozenInstanceError):
            layout.coordinates = {}  # type: ignore[misc]


class TestLayoutAndRenderOptions:
    """Tests for LayoutOptions and RenderOptions configuration objects."""

    def test_layout_options_defaults_and_custom(self):
        opts = LayoutOptions()
        assert opts.timeout_seconds == 60
        assert opts.component_padding == 2

        custom = LayoutOptions(timeout_seconds=120, component_padding=5)
        assert custom.timeout_seconds == 120
        assert custom.component_padding == 5

        with pytest.raises(FrozenInstanceError):
            custom.timeout_seconds = 30  # type: ignore[misc]

    def test_render_options_defaults_and_custom(self):
        opts = RenderOptions()
        assert opts.split_levels is False
        assert opts.scale == 1.0
        assert opts.theme == "default"

        custom = RenderOptions(split_levels=True, scale=2.5, theme="dark")
        assert custom.split_levels is True
        assert custom.scale == 2.5
        assert custom.theme == "dark"

        with pytest.raises(FrozenInstanceError):
            custom.theme = "light"  # type: ignore[misc]


class TestWorldGraph:
    """Tests for WorldGraph value object and factory methods."""

    def test_world_graph_direct_instantiation(self):
        r1 = RoomDef(vnum=1, name="Room 1", description="First")
        ex1 = ExitDef(direction=0, dst_vnum=2)
        hdr = AreaHeader(filename="test.are", name="Test", builder="Auth", vnum_min=1, vnum_max=2)

        graph = WorldGraph(rooms={1: r1}, exits=(ex1,), header=hdr)
        assert graph.rooms[1] == r1
        assert len(graph.exits) == 1
        assert graph.header == hdr

    def test_world_graph_immutability(self):
        graph = WorldGraph(rooms={}, exits=(), header=None)
        with pytest.raises(FrozenInstanceError):
            graph.rooms = {}  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            graph.header = None  # type: ignore[misc]

    def test_world_graph_from_empty_area_data(self):
        area = AreaData()
        graph = WorldGraph.from_area_data(area)

        assert graph.rooms == {}
        assert graph.exits == ()
        assert graph.header is None

    def test_world_graph_from_populated_area_data(self):
        ex1 = ExitDef(direction=0, dst_vnum=2)
        ex2 = ExitDef(direction=1, dst_vnum=3)
        ex3 = ExitDef(direction=2, dst_vnum=1)

        r1 = RoomDef(vnum=1, name="Hall", description="Entrance", exits=(ex1, ex2))
        r2 = RoomDef(vnum=2, name="Chamber", description="Inner", exits=(ex3,))
        hdr = AreaHeader(filename="test.are", name="Test", builder="Auth", vnum_min=1, vnum_max=2)

        area = AreaData(header=hdr, rooms=(r1, r2))
        graph = WorldGraph.from_area_data(area)

        assert graph.header == hdr
        assert set(graph.rooms.keys()) == {1, 2}
        assert graph.rooms[1] == r1
        assert graph.rooms[2] == r2
        assert graph.exits == (ex1, ex2, ex3)


class TestProtocols:
    """Runtime checkability and protocol conformance tests."""

    def test_base_parser_protocol(self):
        class ValidParser:
            def parse(self, source: Path | str | TextIO) -> WorldGraph:
                return WorldGraph(rooms={}, exits=())

        class InvalidParser:
            def ingest(self, source: Path | str | TextIO) -> WorldGraph:
                return WorldGraph(rooms={}, exits=())

        valid = ValidParser()
        assert isinstance(valid, BaseParser)
        assert valid.parse("dummy").rooms == {}

        assert not isinstance(InvalidParser(), BaseParser)
        assert not isinstance(object(), BaseParser)

    def test_base_layout_solver_protocol(self):
        class ValidSolver:
            def solve(self, graph: WorldGraph, options: LayoutOptions | None = None) -> SolvedLayout:
                return SolvedLayout(
                    coordinates={},
                    diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
                )

        class InvalidSolver:
            def calculate_layout(self, graph: WorldGraph) -> SolvedLayout:
                return SolvedLayout(
                    coordinates={},
                    diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
                )

        valid = ValidSolver()
        assert isinstance(valid, BaseLayoutSolver)
        layout = valid.solve(WorldGraph(rooms={}, exits=()))
        assert layout.diagnostics.status == SolverStatus.OPTIMAL

        assert not isinstance(InvalidSolver(), BaseLayoutSolver)
        assert not isinstance(object(), BaseLayoutSolver)

    def test_base_renderer_protocol(self):
        class ValidRenderer:
            def render(
                self,
                graph: WorldGraph,
                layout: SolvedLayout,
                output_path: Path | str,
                options: RenderOptions | None = None,
            ) -> Path:
                return Path(output_path)

        class InvalidRenderer:
            def draw(self, graph: WorldGraph, layout: SolvedLayout) -> Path:
                return Path("out.svg")

        valid = ValidRenderer()
        assert isinstance(valid, BaseRenderer)
        dummy_layout = SolvedLayout(
            coordinates={},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
        )
        assert valid.render(WorldGraph(rooms={}, exits=()), dummy_layout, "out.svg") == Path("out.svg")

        assert not isinstance(InvalidRenderer(), BaseRenderer)
        assert not isinstance(object(), BaseRenderer)
