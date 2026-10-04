"""Unit and integration tests for decoupled renderers consuming (WorldGraph, SolvedLayout)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from romutil.interfaces import BaseRenderer as InterfaceBaseRenderer
from romutil.models import AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.renderers import (
    BaseRenderer,
    HTMLRenderer,
    JSONRenderer,
    SVGRenderer,
    adapt_legacy_rdb,
    build_area_json,
    get_renderer,
    render_map,
)
from romutil.types import (
    Coord3D,
    LayoutDiagnostics,
    RenderOptions,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


@pytest.fixture
def sample_world_graph_and_layout() -> tuple[WorldGraph, SolvedLayout]:
    """Create a decoupled WorldGraph and SolvedLayout test fixture with 2 floors."""
    ex100_north = ExitDef(direction=0, dst_vnum=101)  # North to 101
    ex100_up = ExitDef(direction=4, dst_vnum=200)     # Up to 200
    ex100_east = ExitDef(direction=1, dst_vnum=999)   # East stub to external room 999

    r100 = RoomDef(
        vnum=100,
        name="Ground Hall",
        description="A grand entrance hall on the ground level.",
        exits=(ex100_north, ex100_up, ex100_east),
        area_name="Castle",
        area_file="castle.are",
    )

    ex101_south = ExitDef(direction=2, dst_vnum=100)  # Reciprocal south to 100
    r101 = RoomDef(
        vnum=101,
        name="Throne Room",
        description="The royal throne room.",
        exits=(ex101_south,),
        area_name="Castle",
        area_file="castle.are",
    )

    ex200_down = ExitDef(direction=5, dst_vnum=100)   # Reciprocal down to 100
    r200 = RoomDef(
        vnum=200,
        name="Tower Observatory",
        description="A high observation deck with telescope.",
        exits=(ex200_down,),
        area_name="Castle",
        area_file="castle.are",
    )

    header = AreaHeader(
        filename="castle.are",
        name="Royal Castle",
        builder="King",
        vnum_min=100,
        vnum_max=200,
    )

    graph = WorldGraph(
        rooms={100: r100, 101: r101, 200: r200},
        exits=(ex100_north, ex100_up, ex100_east, ex101_south, ex200_down),
        header=header,
    )

    layout = SolvedLayout(
        coordinates={
            100: Coord3D(x=0, y=0, z=0),
            101: Coord3D(x=0, y=2, z=0),
            200: Coord3D(x=0, y=0, z=1),
        },
        diagnostics=LayoutDiagnostics(
            status=SolverStatus.OPTIMAL,
            duration_seconds=0.05,
        ),
    )
    return graph, layout


class TestBaseRendererProtocolCheck:
    """Verify BaseRenderer protocol conformance and checkability."""

    def test_protocol_conformance(self):
        svg = SVGRenderer()
        json_r = JSONRenderer()
        html_r = HTMLRenderer()

        assert isinstance(svg, BaseRenderer)
        assert isinstance(json_r, BaseRenderer)
        assert isinstance(html_r, BaseRenderer)
        assert isinstance(svg, InterfaceBaseRenderer)
        assert isinstance(json_r, InterfaceBaseRenderer)
        assert isinstance(html_r, InterfaceBaseRenderer)

    def test_non_conforming_class(self):
        class IncompleteRenderer:
            def plot(self) -> None:
                pass

        assert not isinstance(IncompleteRenderer(), BaseRenderer)
        assert not isinstance(object(), BaseRenderer)


class TestJSONRendererDecoupled:
    """Tests for decoupled JSONRenderer and build_area_json."""

    def test_json_render_primary(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_path = tmp_path / "map.json"
        renderer = JSONRenderer()

        result = renderer.render(graph, layout, out_path)
        assert result == out_path
        assert out_path.exists()

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["area"]["name"] == "Royal Castle"
        assert data["area"]["file"] == "castle.are"
        assert data["bounds"] == {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 2, "min_z": 0, "max_z": 1}
        assert len(data["rooms"]) == 3

        r100 = next(r for r in data["rooms"] if r["vnum"] == 100)
        assert r100["coords"] == {"x": 0, "y": 0, "z": 0}
        assert r100["name"] == "Ground Hall"
        assert r100["area_name"] == "Castle"
        assert r100["area_file"] == "castle.are"

        # Check exits on room 100
        exits_by_dir = {e["direction"]: e for e in r100["exits"]}
        # North exit to 101 has reciprocal return in 101 (south)
        assert exits_by_dir["north"]["dst"] == 101
        assert exits_by_dir["north"]["distance"] == 2
        assert exits_by_dir["north"]["one_way"] is False

        # Up exit to 200 has reciprocal return in 200 (down)
        assert exits_by_dir["up"]["dst"] == 200
        assert exits_by_dir["up"]["distance"] == 1
        assert exits_by_dir["up"]["one_way"] is False

        # East exit to 999 is external boundary stub
        assert exits_by_dir["east"]["dst"] == 999
        assert exits_by_dir["east"]["distance"] == 1
        assert exits_by_dir["east"]["one_way"] is True

    def test_json_render_options_and_indent(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_path = tmp_path / "custom.json"
        renderer = JSONRenderer()
        custom_bounds = {"min_x": -10, "max_x": 10, "min_y": -10, "max_y": 10, "min_z": 0, "max_z": 5}
        options = RenderOptions(split_levels=False, scale=2.0)

        result = renderer.render(graph, layout, out_path, options=options, indent=4, bounds=custom_bounds)
        assert result == out_path
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["bounds"]["min_x"] == -10
        assert data["bounds"]["max_x"] == 10

    def test_json_render_type_errors(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        renderer = JSONRenderer()

        with pytest.raises(TypeError, match="Expected SolvedLayout"):
            renderer.render(graph, "invalid_layout", tmp_path / "out.json")  # type: ignore

        with pytest.raises(TypeError, match="output_path must be specified"):
            renderer.render(graph, layout, None)  # type: ignore

    def test_build_area_json_pure_without_legacy_fields(self):
        # RoomDef does not possess x, y, z, or dummy attributes
        r_pure = RoomDef(vnum=42, name="Pure Room", description="No legacy fields")
        assert not hasattr(r_pure, "x")
        assert not hasattr(r_pure, "y")
        assert not hasattr(r_pure, "z")
        assert not hasattr(r_pure, "dummy")

        graph = WorldGraph(rooms={42: r_pure}, exits=(), header=None)
        layout = SolvedLayout(
            coordinates={42: Coord3D(x=5, y=10, z=2)},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.01),
        )

        data = build_area_json(graph, layout)
        assert len(data["rooms"]) == 1
        assert data["rooms"][0]["vnum"] == 42
        assert data["rooms"][0]["coords"] == {"x": 5, "y": 10, "z": 2}


class TestHTMLRendererDecoupled:
    """Tests for decoupled HTMLRenderer."""

    def test_html_render_primary(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_path = tmp_path / "viewer.html"
        renderer = HTMLRenderer()

        result = renderer.render(graph, layout, out_path)
        assert result == out_path
        assert out_path.exists()

        content = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Royal Castle" in content
        assert "castle.are" in content
        assert '<svg id="map-svg"' in content
        assert '<canvas id="minimap"' in content

    def test_html_render_with_title_and_bounds(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_path = tmp_path / "custom_viewer.html"
        renderer = HTMLRenderer()
        custom_bounds = {"min_x": -5, "max_x": 15, "min_y": -5, "max_y": 15, "min_z": 0, "max_z": 2}

        renderer.render(
            graph,
            layout,
            out_path,
            options=RenderOptions(theme="dark"),
            title="Custom Royal Viewer",
            bounds=custom_bounds,
        )
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "Custom Royal Viewer" in content

    def test_html_render_type_errors(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        renderer = HTMLRenderer()

        with pytest.raises(TypeError, match="Expected SolvedLayout"):
            renderer.render(graph, "invalid", tmp_path / "out.html")  # type: ignore

        with pytest.raises(TypeError, match="output_path must be specified"):
            renderer.render(graph, layout, None)  # type: ignore


class TestSVGRendererDecoupled:
    """Tests for decoupled SVGRenderer."""

    def test_svg_render_primary(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_path = tmp_path / "map.svg"
        renderer = SVGRenderer()

        result = renderer.render(graph, layout, out_path)
        assert result == out_path
        assert out_path.exists()

        content = out_path.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "elevation-0" in content
        assert "elevation-1" in content
        assert "Ground Hall" in content
        assert "Throne Room" in content
        assert "Tower Observatory" in content

    def test_svg_render_split_levels(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_path = tmp_path / "castle.svg"
        renderer = SVGRenderer()

        result = renderer.render(
            graph,
            layout,
            out_path,
            options=RenderOptions(split_levels=True),
        )
        assert result == out_path
        z0 = tmp_path / "castle_z0.svg"
        z1 = tmp_path / "castle_z1.svg"
        assert z0.exists()
        assert z1.exists()

        c0 = z0.read_text(encoding="utf-8")
        c1 = z1.read_text(encoding="utf-8")
        assert "elevation-0" in c0
        assert "elevation-1" not in c0
        assert "elevation-1" in c1
        assert "elevation-0" not in c1

    def test_svg_constructor_decoupled_and_plot(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out_file = tmp_path / "direct_plot.svg"
        renderer = SVGRenderer(
            name=out_file,
            graph=graph,
            layout=layout,
            options=RenderOptions(),
            target_z=0,
        )
        renderer.plot()
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "elevation-0" in content
        assert "elevation-1" not in content

    def test_svg_directional_stubs_all_compass_bearings(self, tmp_path):
        """Test directional projection stubs for all 10 Diku directions."""
        directions = [
            (Direction.north, 0),
            (Direction.east, 1),
            (Direction.south, 2),
            (Direction.west, 3),
            (Direction.up, 4),
            (Direction.down, 5),
            (Direction.northeast, 6),
            (Direction.northwest, 7),
            (Direction.southeast, 8),
            (Direction.southwest, 9),
        ]
        exits = tuple(ExitDef(direction=dir_enum, dst_vnum=9000 + i) for i, (dir_enum, _) in enumerate(directions))
        r = RoomDef(vnum=10, name="Hub Room", description="Exits in all directions", exits=exits)

        graph = WorldGraph(rooms={10: r}, exits=exits)
        layout = SolvedLayout(
            coordinates={10: Coord3D(x=5, y=5, z=0)},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.01),
        )

        out = tmp_path / "compass.svg"
        renderer = SVGRenderer()
        renderer.render(graph, layout, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content

        # Directly test proj_exit stub branches
        svg_inst = SVGRenderer(graph=graph, layout=layout)
        for ex in exits:
            proj = svg_inst.proj_exit(ex, src_vnum=10)
            assert proj is not None
            start, end = proj
            assert start != end

    def test_svg_proj_room_input_types(self):
        coord = Coord3D(x=3, y=4, z=1)
        r = RoomDef(vnum=10, name="Test", description="")
        layout = SolvedLayout(
            coordinates={10: coord},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
        )
        renderer = SVGRenderer(layout=layout)
        renderer.z_max = 2.0
        renderer.y_max = 10.0

        p1 = renderer.proj_room(coord)
        p2 = renderer.proj_room(10)
        p3 = renderer.proj_room(r)
        assert p1 is not None and p2 is not None and p3 is not None
        assert p1 == p2 == p3

        # Missing vnum returns None
        assert renderer.proj_room(999) is None
        assert renderer.proj_room("invalid") is None

    def test_svg_render_type_errors(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        renderer = SVGRenderer()

        with pytest.raises(TypeError, match="Expected SolvedLayout"):
            renderer.render(graph, "invalid", tmp_path / "out.svg")  # type: ignore

        with pytest.raises(TypeError, match="output_path must be specified"):
            renderer.render(graph, layout, None)  # type: ignore


class TestRenderMapDecoupledDispatcher:
    """Tests for render_map dispatcher consuming (WorldGraph, SolvedLayout)."""

    def test_render_map_svg(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out = tmp_path / "map.svg"
        result = render_map(graph, layout, out)
        assert result == out
        assert out.exists()
        assert "<svg" in out.read_text(encoding="utf-8")

    def test_render_map_json(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out = tmp_path / "map.json"
        result = render_map(graph, layout, out)
        assert result == out
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["rooms"]) == 3

    def test_render_map_html(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out = tmp_path / "viewer.html"
        result = render_map(graph, layout, out)
        assert result == out
        assert out.exists()
        assert "<!DOCTYPE html>" in out.read_text(encoding="utf-8")

    def test_render_map_explicit_fmt(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out = tmp_path / "raw_file"
        result = render_map(graph, layout, out, fmt="json")
        assert result == out
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["rooms"]) == 3

    def test_render_map_default_svg_when_no_ext(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        out = tmp_path / "raw_file_svg"
        result = render_map(graph, layout, out)
        assert result == out
        assert out.exists()
        assert "<svg" in out.read_text(encoding="utf-8")

    def test_render_map_type_errors(self, sample_world_graph_and_layout, tmp_path):
        graph, layout = sample_world_graph_and_layout
        with pytest.raises(TypeError, match="Expected SolvedLayout"):
            render_map(graph, "invalid_layout", tmp_path / "out.svg")  # type: ignore

        with pytest.raises(TypeError, match="output_path must be specified"):
            render_map(graph, layout, None)  # type: ignore


class TestEdgeCasesAndAdaptLegacyRdb:
    """Test boundary cases and legacy conversion utilities."""

    def test_empty_world_graph(self, tmp_path):
        empty_graph = WorldGraph(rooms={}, exits=(), header=None)
        empty_layout = SolvedLayout(
            coordinates={},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
        )

        json_out = tmp_path / "empty.json"
        JSONRenderer().render(empty_graph, empty_layout, json_out)
        assert json_out.exists()
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data["rooms"] == []
        assert data["bounds"] == {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}

        svg_out = tmp_path / "empty.svg"
        SVGRenderer().render(empty_graph, empty_layout, svg_out)
        assert svg_out.exists()

        html_out = tmp_path / "empty.html"
        HTMLRenderer().render(empty_graph, empty_layout, html_out)
        assert html_out.exists()

    def test_single_room_world_graph(self, tmp_path):
        r = RoomDef(vnum=1, name="Lone Room", description="Only room in the world.")
        graph = WorldGraph(rooms={1: r}, exits=(), header=None)
        layout = SolvedLayout(
            coordinates={1: Coord3D(x=0, y=0, z=0)},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
        )

        out = tmp_path / "single.svg"
        SVGRenderer().render(graph, layout, out)
        assert out.exists()
        assert "Lone Room" in out.read_text(encoding="utf-8")

    def test_adapt_legacy_rdb_filters_dummies_and_handles_headers(self):
        r1 = Room(RoomDef(vnum=10, name="Real Room", description="Valid"))
        r1.x, r1.y, r1.z = 1, 2, 3
        dummy = Room(RoomDef(vnum=99, name="Dummy Room", description="Warp"))
        dummy.dummy = True

        # Header as tuple ("legacy.are", "Legacy Area")
        header_tuple = ("legacy.are", "Legacy Area")
        graph, layout = adapt_legacy_rdb([r1, dummy], header=header_tuple)

        assert len(graph.rooms) == 1
        assert 10 in graph.rooms
        assert 99 not in graph.rooms
        assert layout.coordinates[10] == Coord3D(x=1, y=2, z=3)
        assert graph.header is not None
        assert graph.header.filename == "legacy.are"
        assert graph.header.name == "Legacy Area"

    def test_adapt_legacy_rdb_none_and_invalid_inputs(self):
        graph, layout = adapt_legacy_rdb(None, header=None)  # type: ignore
        assert graph.rooms == {}
        assert layout.coordinates == {}
        assert graph.header is None

    def test_adapt_legacy_rdb_duck_typed_header_and_unplaced_rooms(self):
        class MockHeader:
            filename = "duck.are"
            name = "Duck World"
            builder = "Donald"
            vnum_min = 1
            vnum_max = 5

        r_placed = Room(RoomDef(vnum=1, name="Placed", description=""))
        r_placed.x, r_placed.y, r_placed.z = 0, 0, 0

        r_unplaced = Room(RoomDef(vnum=2, name="Unplaced", description=""))
        # x, y, z are None

        graph, layout = adapt_legacy_rdb({1: r_placed, 2: r_unplaced}, header=MockHeader())
        assert len(graph.rooms) == 2
        assert len(layout.coordinates) == 1
        assert 1 in layout.coordinates
        assert 2 not in layout.coordinates
        assert graph.header is not None
        assert graph.header.name == "Duck World"

    def test_svg_legacy_keyword_and_positional_constructors(self, tmp_path):
        r = Room(RoomDef(vnum=1, name="Room", description=""))
        r.x, r.y, r.z = 0, 0, 0
        rdb = {1: r}

        # Keyword rdb
        renderer_kw = SVGRenderer(rdb=rdb)
        assert 1 in renderer_kw.graph.rooms
        assert 1 in renderer_kw.layout.coordinates

        # Positional with integer target_z in options slot
        renderer_pos = SVGRenderer(str(tmp_path / "pos.svg"), rdb, None, 0)
        assert renderer_pos.target_z == 0
        renderer_pos.plot()
        assert (tmp_path / "pos.svg").exists()

    def test_json_and_svg_exits_with_target_vnum_and_custom_attributes(self, tmp_path):
        class MockExit:
            def __init__(self):
                self.direction = 99  # Unknown direction
                self.dst = 2
                self.target_vnum = 20
                self.distance = 5
                self.one_way = True

        r1 = RoomDef(vnum=1, name="R1", description="", exits=(MockExit(),))  # type: ignore
        graph = WorldGraph(rooms={1: r1}, exits=())
        layout = SolvedLayout(
            coordinates={1: Coord3D(x=0, y=0, z=0)},
            diagnostics=LayoutDiagnostics(status=SolverStatus.OPTIMAL, duration_seconds=0.0),
        )

        json_out = tmp_path / "custom_exits.json"
        JSONRenderer().render(graph, layout, json_out)
        data = json.loads(json_out.read_text(encoding="utf-8"))
        ex0 = data["rooms"][0]["exits"][0]
        assert ex0["direction"] == "99"
        assert ex0["dst"] == 20
        assert ex0["distance"] == 5
        assert ex0["one_way"] is True

        # Test render_map with 3rd positional format string
        r = Room(RoomDef(vnum=1, name="R", description=""))
        r.x, r.y, r.z = 0, 0, 0
        map_out = tmp_path / "map_pos_fmt"
        render_map({1: r}, map_out, "json")
        assert map_out.exists()
