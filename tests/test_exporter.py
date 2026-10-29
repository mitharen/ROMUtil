"""Unit tests for exporter serialization and HTML/JSON viewer generation."""

import json
import re
from pathlib import Path
import pytest

from romutil.models import AreaHeader, Exit, Room
from romutil.renderers import (
    build_area_json,
    export_html,
    export_json,
    generate_html_viewer,
)
from romutil.types import Coord3D, SolvedLayout, WorldGraph


class TestExporterUnit:
    """Unit tests for exporter serialization and viewer generation."""

    def test_build_area_json_schema(self):
        r1 = Room(
            vnum=100,
            name="Hallway",
            description="Long hallway.",
            exits=(
                Exit(direction=0, dst_vnum=101),
                Exit(direction=1, dst_vnum=102),
            ),
        )
        r2 = Room(
            vnum=101,
            name="North Room",
            description="North room.",
            exits=(Exit(direction=3, dst_vnum=100),),
        )
        r3 = Room(
            vnum=102,
            name="East Room",
            description="East room.",
            exits=(Exit(direction=4, dst_vnum=100),),
        )

        area_meta = AreaHeader(
            filename="test.are",
            name="Test Area",
            builder="Builder",
            vnum_min=100,
            vnum_max=102,
        )
        graph = WorldGraph(
            rooms={100: r1, 101: r2, 102: r3},
            exits=r1.exits + r2.exits + r3.exits,
            header=area_meta,
        )
        layout = SolvedLayout(
            coordinates={
                100: Coord3D(x=0, y=1, z=0),
                101: Coord3D(x=0, y=2, z=0),
                102: Coord3D(x=1, y=1, z=0),
            }
        )

        data = build_area_json(graph, layout)

        # 1. Verify Top-Level Structure
        assert isinstance(data, dict)
        assert set(data.keys()) == {"area", "bounds", "rooms", "mobiles"}

        # 2. Verify Area Meta
        assert data["area"]["name"] == "Test Area"
        assert data["area"]["file"] == "test.are"

        # 3. Verify Bounds
        bounds = data["bounds"]
        assert bounds["min_x"] == 0
        assert bounds["max_x"] == 1
        assert bounds["min_y"] == 1
        assert bounds["max_y"] == 2
        assert bounds["min_z"] == 0
        assert bounds["max_z"] == 0

        # 4. Verify Rooms List
        assert len(data["rooms"]) == 3
        vnums = [r["vnum"] for r in data["rooms"]]
        assert vnums == [100, 101, 102]

        room_100 = data["rooms"][0]
        assert room_100["vnum"] == 100
        assert room_100["name"] == "Hallway"
        assert room_100["desc"] == "Long hallway."
        assert room_100["coords"] == {"x": 0, "y": 1, "z": 0}
        assert len(room_100["exits"]) == 2

        # Verify Exit Structure
        ex0 = room_100["exits"][0]
        assert "direction" in ex0 and isinstance(ex0["direction"], str)
        assert "dst" in ex0 and isinstance(ex0["dst"], int)
        assert "distance" in ex0 and isinstance(ex0["distance"], int)
        assert "one_way" in ex0 and isinstance(ex0["one_way"], bool)

    def test_build_area_json_empty(self):
        graph = WorldGraph(rooms={}, exits=(), header=None)
        layout = SolvedLayout(coordinates={})
        data = build_area_json(graph, layout)
        assert data["rooms"] == []
        assert data["bounds"] == {
            "min_x": 0,
            "max_x": 0,
            "min_y": 0,
            "max_y": 0,
            "min_z": 0,
            "max_z": 0,
        }
        assert data["area"] == {"name": "", "file": ""}

    def test_build_area_json_custom_bounds(self):
        r1 = Room(vnum=10, name="Room 10", description="Desc")
        graph = WorldGraph(rooms={10: r1}, exits=())
        layout = SolvedLayout(coordinates={10: Coord3D(x=2, y=3, z=1)})

        data = build_area_json(
            graph,
            layout,
            bounds={
                "min_x": 0,
                "max_x": 5,
                "min_y": 0,
                "max_y": 5,
                "min_z": 0,
                "max_z": 2,
            },
        )
        assert len(data["rooms"]) == 1
        assert data["rooms"][0]["vnum"] == 10
        assert data["bounds"]["max_x"] == 5

    def test_build_area_json_header_metadata(self):
        header = AreaHeader(filename="obj.are", name="Object Area", builder="B", vnum_min=1, vnum_max=1)
        r = Room(vnum=1, name="Start", description="Desc")
        graph = WorldGraph(rooms={1: r}, exits=(), header=header)
        layout = SolvedLayout(coordinates={1: Coord3D(x=0, y=0, z=0)})
        data = build_area_json(graph, layout)
        assert data["area"]["name"] == "Object Area"
        assert data["area"]["file"] == "obj.are"

        graph_none = WorldGraph(rooms={1: r}, exits=(), header=None)
        data_none = build_area_json(graph_none, layout)
        assert data_none["area"]["name"] == ""
        assert data_none["area"]["file"] == ""

    def test_export_json_file_writing(self, tmp_path):
        target = tmp_path / "sub" / "map.json"
        data = {"area": {"name": "A", "file": "f"}, "bounds": {}, "rooms": []}
        out = export_json(data, target)
        assert out == target
        assert target.exists()
        with open(target, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_generate_html_viewer_structure(self):
        data = {
            "area": {"name": "Test Viewer Area", "file": "test.are"},
            "bounds": {"min_x": 0, "max_x": 4, "min_y": 0, "max_y": 4, "min_z": 0, "max_z": 1},
            "rooms": [
                {
                    "vnum": 1,
                    "name": "First Room",
                    "desc": "First description</script><script>",
                    "coords": {"x": 0, "y": 0, "z": 0},
                    "exits": [{"direction": "north", "dst": 2, "distance": 1, "one_way": False}],
                }
            ],
        }
        html = generate_html_viewer(data)
        assert "<!DOCTYPE html>" in html
        assert '<svg id="map-svg"' in html
        assert '<canvas id="minimap"' in html
        assert '<script id="romutil-data" type="application/json">' in html
        # Script tags inside JSON must be escaped
        assert "</script><script>" not in html
        assert r"<\/script><script>" in html or r"<\/script>" in html

    def test_html_zero_broken_external_assets(self):
        data = {
            "area": {"name": "Safe Area", "file": "safe.are"},
            "bounds": {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 0},
            "rooms": [],
        }
        html = generate_html_viewer(data)

        # 1. Zero external scripts
        script_tags = re.findall(r"<script\b[^>]*>", html, re.IGNORECASE)
        for tag in script_tags:
            assert "src=" not in tag.lower(), f"Forbidden external script tag: {tag}"

        # 2. Zero external stylesheets
        link_tags = re.findall(r"<link\b[^>]*>", html, re.IGNORECASE)
        for tag in link_tags:
            assert "stylesheet" not in tag.lower(), f"Forbidden external link tag: {tag}"

        # 3. Zero external URLs (http:// or https://)
        external_srcs = re.findall(r"src=['\"]https?://", html, re.IGNORECASE)
        assert len(external_srcs) == 0, f"External src URLs found in HTML: {external_srcs}"
        external_hrefs = re.findall(r"href=['\"]https?://", html, re.IGNORECASE)
        assert len(external_hrefs) == 0, f"External href URLs found in HTML: {external_hrefs}"
        css_urls = re.findall(r"url\(['\"]?https?://", html, re.IGNORECASE)
        assert len(css_urls) == 0, f"External CSS URLs found in HTML: {css_urls}"

    def test_export_html_file_writing(self, tmp_path):
        target = tmp_path / "viewer.html"
        data = {"area": {"name": "A", "file": "f"}, "bounds": {}, "rooms": []}
        out = export_html(data, target, title="Custom Title")
        assert out == target
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "<title>Custom Title</title>" in content


class TestHtmlElevationViewerUnit:
    """Unit tests for HTML viewer elevation layers and multi-plane z-ordering."""

    def test_html_viewer_contains_elevation_layers_and_sorting_logic(self):
        data = {
            "area": {"name": "Multi Floor Area", "file": "multi.are"},
            "bounds": {"min_x": 0, "max_x": 5, "min_y": 0, "max_y": 5, "min_z": 0, "max_z": 3},
            "rooms": [
                {"vnum": 300, "name": "R3", "desc": "", "coords": {"x": 2, "y": 2, "z": 3}, "exits": []},
                {"vnum": 100, "name": "R0", "desc": "", "coords": {"x": 2, "y": 2, "z": 0}, "exits": []},
                {"vnum": 200, "name": "R1", "desc": "", "coords": {"x": 2, "y": 2, "z": 1}, "exits": []},
            ],
        }
        html = generate_html_viewer(data)

        # 1. Verify elevation layer style is embedded
        assert ".elevation-layer" in html

        # 2. Verify sortedRooms logic exists in client script
        assert "const sortedRooms = [...rooms].sort(" in html
        assert "a.coords.z - b.coords.z" in html
        assert "b.coords.y - a.coords.y" in html
        assert "a.coords.x - b.coords.x" in html

        # 3. Verify dynamic elevation layer creation
        assert "layer.setAttribute('id', 'elevation-' + z);" in html
        assert "layer.setAttribute('class', 'elevation-layer');" in html

        # 4. Verify minimap canvas renders using sortedRooms
        assert "sortedRooms.forEach(r => {" in html

    def test_simulated_canvas_draw_list_strict_ascending_z_order(self):
        rooms = [
            {"vnum": 500, "name": "Floor 0 Room", "coords": {"x": 1, "y": 2, "z": 0}},
            {"vnum": 400, "name": "Floor 1 Room Low Y", "coords": {"x": 1, "y": 1, "z": 1}},
            {"vnum": 300, "name": "Floor 1 Room High Y", "coords": {"x": 1, "y": 3, "z": 1}},
            {"vnum": 200, "name": "Floor 2 Room West", "coords": {"x": 0, "y": 2, "z": 2}},
            {"vnum": 100, "name": "Floor 2 Room East", "coords": {"x": 2, "y": 2, "z": 2}},
        ]

        def painters_comparator(r):
            return (r["coords"]["z"], -r["coords"]["y"], r["coords"]["x"], r["vnum"])

        draw_queue = sorted(rooms, key=painters_comparator)

        # Assert draw_queue follows strict ascending Z order
        zs = [r["coords"]["z"] for r in draw_queue]
        assert zs == [0, 1, 1, 2, 2]
        assert draw_queue[0]["vnum"] == 500  # Floor 0
        assert draw_queue[1]["vnum"] == 300  # Floor 1 High Y
        assert draw_queue[2]["vnum"] == 400  # Floor 1 Low Y
        assert draw_queue[3]["vnum"] == 200  # Floor 2 West
        assert draw_queue[4]["vnum"] == 100  # Floor 2 East

    def test_inter_floor_exit_max_elevation_assignment(self):
        data = {
            "area": {"name": "Tower", "file": "tower.are"},
            "bounds": {"min_x": 0, "max_x": 2, "min_y": 0, "max_y": 2, "min_z": 0, "max_z": 1},
            "rooms": [
                {
                    "vnum": 1,
                    "name": "Ground",
                    "coords": {"x": 0, "y": 0, "z": 0},
                    "exits": [{"direction": "up", "dst": 2, "distance": 1, "one_way": False}],
                },
                {
                    "vnum": 2,
                    "name": "Upper",
                    "coords": {"x": 0, "y": 0, "z": 1},
                    "exits": [{"direction": "down", "dst": 1, "distance": 1, "one_way": False}],
                },
            ],
        }
        html = generate_html_viewer(data)
        assert "Math.max(r.coords.z, target.coords.z)" in html
        assert "elevationLayers.get(edgeZ)" in html
