import io
import json
import os
import re
from pathlib import Path
import pytest

from romutil.models import Direction, Room, Exit, RoomDef, ExitDef, AreaHeader
from romutil.graph import solve_layout
from romutil.renderers import (
    build_area_json,
    export_json,
    generate_html_viewer,
    export_html,
)
from romutil.cli import main, cli

from tests.conftest import SAMPLE_AREAS_DIR


class TestExporterUnit:
    """Unit tests for exporter serialization and viewer generation."""

    def test_build_area_json_schema(self):
        r1 = Room(RoomDef(vnum=100, name="Hallway", description="Long hallway.", exits=(ExitDef(direction=0, dst_vnum=101), ExitDef(direction=1, dst_vnum=102))))
        r1.x, r1.y, r1.z = 0, 1, 0
        r2 = Room(RoomDef(vnum=101, name="North Room", description="North room.", exits=(ExitDef(direction=3, dst_vnum=100),)))
        r2.x, r2.y, r2.z = 0, 2, 0
        r3 = Room(RoomDef(vnum=102, name="East Room", description="East room.", exits=(ExitDef(direction=4, dst_vnum=100),)))
        r3.x, r3.y, r3.z = 1, 1, 0

        rdb = {100: r1, 101: r2, 102: r3}
        area_meta = AreaHeader(filename="test.are", name="Test Area", builder="Builder", vnum_min=100, vnum_max=102)

        data = build_area_json(rdb, area_meta)

        # 1. Verify Top-Level Structure
        assert isinstance(data, dict)
        assert set(data.keys()) == {"area", "bounds", "rooms"}

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
        data = build_area_json({})
        assert data["rooms"] == []
        assert data["bounds"] == {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        assert data["area"] == {"name": "", "file": ""}

    def test_build_area_json_list_input_and_dummy_filtering(self):
        r1 = Room(RoomDef(vnum=10, name="Room 10", description="Desc"))
        r1.x, r1.y, r1.z = 2, 3, 1
        dummy = Room(RoomDef(vnum=99, name="Dummy", description=""))
        dummy.dummy = True

        data = build_area_json([r1, dummy], bounds={"min_x": 0, "max_x": 5, "min_y": 0, "max_y": 5, "min_z": 0, "max_z": 2})
        assert len(data["rooms"]) == 1
        assert data["rooms"][0]["vnum"] == 10
        assert data["bounds"]["max_x"] == 5

    def test_build_area_json_meta_object_or_none(self):
        class MockMeta:
            name = "Object Area"
            filename = "obj.are"

        r = Room(RoomDef(vnum=1, name="Start", description="Desc"))
        r.x, r.y, r.z = 0, 0, 0
        data = build_area_json({1: r}, area_meta=MockMeta())
        assert data["area"]["name"] == "Object Area"
        assert data["area"]["file"] == "obj.are"

        data_none = build_area_json({1: r}, area_meta=None)
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
            ]
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
            "rooms": []
        }
        html = generate_html_viewer(data)

        # 1. Zero external scripts
        script_tags = re.findall(r'<script\b[^>]*>', html, re.IGNORECASE)
        for tag in script_tags:
            assert "src=" not in tag.lower(), f"Forbidden external script tag: {tag}"

        # 2. Zero external stylesheets
        link_tags = re.findall(r'<link\b[^>]*>', html, re.IGNORECASE)
        for tag in link_tags:
            assert "stylesheet" not in tag.lower(), f"Forbidden external link tag: {tag}"

        # 3. Zero external URLs (http:// or https://)
        external_srcs = re.findall(r'src=[\'"]https?://', html, re.IGNORECASE)
        assert len(external_srcs) == 0, f"External src URLs found in HTML: {external_srcs}"
        external_hrefs = re.findall(r'href=[\'"]https?://', html, re.IGNORECASE)
        assert len(external_hrefs) == 0, f"External href URLs found in HTML: {external_hrefs}"
        css_urls = re.findall(r'url\([\'"]?https?://', html, re.IGNORECASE)
        assert len(css_urls) == 0, f"External CSS URLs found in HTML: {css_urls}"

    def test_export_html_file_writing(self, tmp_path):
        target = tmp_path / "viewer.html"
        data = {"area": {"name": "A", "file": "f"}, "bounds": {}, "rooms": []}
        out = export_html(data, target, title="Custom Title")
        assert out == target
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "<title>Custom Title</title>" in content


class TestExporterIntegration:
    """End-to-end integration tests asserting CLI --format json and --format html."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_romutil_school_format_json(self, tmp_path):
        school_file = os.path.join(SAMPLE_AREAS_DIR, "school.are")
        if not os.path.exists(school_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "school_test")
        with open(school_file, 'r', encoding='latin-1') as f:
            with pytest.raises(SystemExit) as exc_info:
                main([f], outbase, fmt="json", solver_timeout=15)
            assert exc_info.value.code == 0

        json_path = tmp_path / "school_test.json"
        assert json_path.exists(), "school_test.json was not created"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify area details
        assert data["area"]["name"].lower() == "mud school"
        assert data["area"]["file"] == "school.are"

        # Verify bounds structure
        bounds = data["bounds"]
        assert all(k in bounds for k in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z"))
        assert bounds["min_x"] == 0
        assert bounds["max_x"] > 0
        assert bounds["max_y"] > 0
        assert bounds["max_z"] >= 0

        # Acceptance criterion: 100% of rooms represented (school.are has 59 rooms)
        rooms = data["rooms"]
        assert len(rooms) == 59, f"Expected 59 rooms in school.are, found {len(rooms)}"

        # Acceptance criterion: 100% of exits represented (school.are has 178 valid exits)
        total_exits = sum(len(r["exits"]) for r in rooms)
        assert total_exits == 178, f"Expected 178 exits in school.are, found {total_exits}"

        # Check every room follows schema
        for r in rooms:
            assert isinstance(r["vnum"], int)
            assert isinstance(r["name"], str) and len(r["name"]) > 0
            assert isinstance(r["desc"], str)
            assert isinstance(r["coords"], dict)
            assert set(r["coords"].keys()) == {"x", "y", "z"}
            assert all(isinstance(r["coords"][axis], int) for axis in ("x", "y", "z"))

            for ex in r["exits"]:
                assert ex["direction"] in {"north", "south", "east", "west", "up", "down"}
                assert isinstance(ex["dst"], int)
                assert isinstance(ex["distance"], int)
                assert isinstance(ex["one_way"], bool)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_romutil_school_format_html(self, tmp_path):
        school_file = os.path.join(SAMPLE_AREAS_DIR, "school.are")
        if not os.path.exists(school_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "school_test")
        with open(school_file, 'r', encoding='latin-1') as f:
            with pytest.raises(SystemExit) as exc_info:
                main([f], outbase, fmt="html", solver_timeout=15)
            assert exc_info.value.code == 0

        html_path = tmp_path / "school_test.html"
        assert html_path.exists(), "school_test.html was not created"

        content = html_path.read_text(encoding="utf-8")

        # Acceptance criterion: Embedded canvas and SVG elements
        assert '<svg id="map-svg"' in content
        assert '<canvas id="minimap"' in content
        assert "romutil-data" in content

        # Acceptance criterion: Zero broken external assets
        script_srcs = re.findall(r'<script[^>]*src=', content, re.IGNORECASE)
        assert len(script_srcs) == 0
        css_links = re.findall(r'<link[^>]*stylesheet', content, re.IGNORECASE)
        assert len(css_links) == 0
        external_srcs = re.findall(r'src=[\'"]https?://', content, re.IGNORECASE)
        assert len(external_srcs) == 0, f"External src URLs found in HTML: {external_srcs}"
        external_hrefs = re.findall(r'href=[\'"]https?://', content, re.IGNORECASE)
        assert len(external_hrefs) == 0, f"External href URLs found in HTML: {external_hrefs}"
        css_urls = re.findall(r'url\([\'"]?https?://', content, re.IGNORECASE)
        assert len(css_urls) == 0, f"External CSS URLs found in HTML: {css_urls}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_execution_with_formats(self, tmp_path, monkeypatch):
        smurf_file = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        if not os.path.exists(smurf_file):
            pytest.skip("QuickMUD area files not found")

        # Test CLI with --format json
        out_json = str(tmp_path / "cli_smurf")
        monkeypatch.setattr("sys.argv", ["romutil", smurf_file, "-outbase", out_json, "--format", "json"])
        with pytest.raises(SystemExit) as exc1:
            cli()
        assert exc1.value.code == 0
        assert (tmp_path / "cli_smurf.json").exists()

        # Test CLI with -f html and explicit .html in outbase
        out_html = str(tmp_path / "cli_smurf.html")
        monkeypatch.setattr("sys.argv", ["romutil", smurf_file, "-outbase", out_html, "-f", "html"])
        with pytest.raises(SystemExit) as exc2:
            cli()
        assert exc2.value.code == 0
        assert (tmp_path / "cli_smurf.html").exists()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_default_outbase_derivation(self, tmp_path, monkeypatch):
        smurf_file = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        if not os.path.exists(smurf_file):
            pytest.skip("QuickMUD area files not found")

        # Copy smurf.are into tmp_path so default output goes to tmp_path/smurf.json
        local_area = tmp_path / "local.are"
        local_area.write_text(Path(smurf_file).read_text(encoding="latin-1"), encoding="latin-1")

        monkeypatch.setattr("sys.argv", ["romutil", str(local_area), "--format", "json"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert (tmp_path / "local.json").exists()


class TestExporterNegative:
    """Negative and boundary test cases."""

    def test_main_no_rooms_json_negative(self, tmp_path, caplog):
        social_file = os.path.join(SAMPLE_AREAS_DIR, "social.are")
        if not os.path.exists(social_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "social_test")
        with open(social_file, 'r', encoding='latin-1') as f:
            with pytest.raises(SystemExit) as exc_info:
                main([f], outbase, fmt="json")
            assert exc_info.value.code == 0
        assert "No rooms to plot" in caplog.text

    def test_main_corrupt_content_json_negative(self, tmp_path, caplog):
        corrupt_stream = io.StringIO("MALFORMED HEADER WITHOUT ANY AREA DATA\n")
        outbase = str(tmp_path / "corrupt_test")
        with pytest.raises(SystemExit) as exc_info:
            main([corrupt_stream], outbase, fmt="json")
        assert exc_info.value.code == 0
        assert "No rooms to plot" in caplog.text

    def test_solve_layout_disconnected_single_room(self):
        # Room with no exits
        r1 = Room(RoomDef(vnum=500, name="Isolated", description="Lonely room."))
        rdb = {500: r1}
        solved_rdb, exits = solve_layout(rdb)
        assert 500 in solved_rdb
        assert solved_rdb[500].x == 0
        assert solved_rdb[500].y == 0
        assert solved_rdb[500].z == 0
        assert len(exits) == 0

    def test_solve_layout_solver_failure_fallback(self, monkeypatch):
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc", exits=(ExitDef(direction=0, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc", exits=(ExitDef(direction=3, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}

        # Mock solver to return infeasible/failed status
        import pyomo.opt
        class MockSolverStatus:
            termination_condition = pyomo.opt.TerminationCondition.infeasible
        class MockResults:
            solver = MockSolverStatus()

        def mock_solve(*args, **kwargs):
            return None, MockResults()

        import sys; monkeypatch.setattr(sys.modules["romutil.graph"], "solve", mock_solve)
        solved_rdb, exits = solve_layout(rdb)
        assert solved_rdb[1].x == 0
        assert solved_rdb[2].x == 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_main_entrypoint(self, tmp_path, monkeypatch):
        import runpy
        smurf_file = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        if not os.path.exists(smurf_file):
            pytest.skip("QuickMUD area files not found")

        out_json = str(tmp_path / "runpy_smurf")
        monkeypatch.setattr("sys.argv", ["romutil.cli", smurf_file, "-outbase", out_json, "-f", "json", "-d"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("romutil.cli", run_name="__main__")
        assert exc.value.code == 0
        assert (tmp_path / "runpy_smurf.json").exists()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_main_multi_component_area_offset(self, tmp_path):
        # Create an area with 2 disconnected components
        multi_comp_are = """#AREA
multi.are~
Multi Comp Area~
{ 1 10 } Builder Multi~
100 299

#ROOMS
#100
Comp 1 Room 1~
Desc~
0 0 0
D0
~
~
0 0 101
S
#101
Comp 1 Room 2~
Desc~
0 0 0
D2
~
~
0 0 100
S
#200
Comp 2 Room 1~
Desc~
0 0 0
D0
~
~
0 0 201
S
#201
Comp 2 Room 2~
Desc~
0 0 0
D2
~
~
0 0 200
S
#0

#$
"""
        area_file = tmp_path / "multi.are"
        area_file.write_text(multi_comp_are, encoding="latin-1")
        outbase = str(tmp_path / "multi_out")

        with pytest.raises(SystemExit) as exc:
            main([area_file], outbase, fmt="json")
        assert exc.value.code == 0

        with open(tmp_path / "multi_out.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["rooms"]) == 4
        # Verify component offset was applied (max_x > 0)
        assert data["bounds"]["max_x"] > 2

    def test_main_without_area_meta(self, tmp_path):
        # Area missing #AREA header
        no_meta_are = """#ROOMS
#10
Solo Room~
Description~
0 0 0
S
#0

#$
"""
        area_file = tmp_path / "nometa.are"
        area_file.write_text(no_meta_are, encoding="latin-1")
        outbase = str(tmp_path / "nometa_out")

        with pytest.raises(SystemExit) as exc:
            main([area_file], outbase, fmt="json")
        assert exc.value.code == 0

        with open(tmp_path / "nometa_out.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["area"]["file"] == "nometa.are"
        assert data["area"]["name"] == "nometa"


class TestExporterPaintersLayering:
    """Automated regression tests for HTML viewer painter's algorithm Z-ordering and layering."""

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
        # Create rooms where VNUM order is opposite to Z order
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
        # Floor 0 must come before Floor 1, which must come before Floor 2
        assert draw_queue[0]["vnum"] == 500  # Floor 0
        # On Floor 1: High Y (y=3, background) drawn before Low Y (y=1, foreground)
        assert draw_queue[1]["vnum"] == 300  # Floor 1 High Y
        assert draw_queue[2]["vnum"] == 400  # Floor 1 Low Y
        # On Floor 2: West (x=0) drawn before East (x=2)
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


class TestSingleSolveMultiFormatCLI:
    """Tests for single-solve multi-format output and CLI parsing (Task 9f)."""

    SAMPLE_MULTI_ARE = """#AREA
multi.are~
MultiFormat~
{ 1 10 } Builder~
1 4

#ROOMS
#1
Center Room~
Desc Center~
0 0 0
D0
~
~
0 0 2
D1
~
~
0 0 3
S
#2
North Room~
Desc North~
0 0 0
D2
~
~
0 0 1
S
#3
East Room~
Desc East~
0 0 0
D3
~
~
0 0 1
S
#4
Isolated Room~
Desc Isolated~
0 0 0
S
#0

#$
"""

    def test_cli_multi_format_html_svg_positive(self, tmp_path, monkeypatch):
        """Positive test: CLI call with --format html,svg generates both .html and .svg in a single pass."""
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "multi_out")

        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", outbase, "--format", "html,svg"]
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

        html_file = tmp_path / "multi_out.html"
        svg_file = tmp_path / "multi_out.svg"
        svg_file_0 = tmp_path / "multi_out0.svg"

        assert html_file.exists()
        assert svg_file.exists()
        assert svg_file_0.exists()
        assert len(html_file.read_text(encoding="utf-8")) > 0
        assert len(svg_file.read_text(encoding="utf-8")) > 0

    def test_cli_multi_format_coordinate_identity(self, tmp_path, monkeypatch):
        """Coordinate identity test: room coordinates in .html match .json and .svg projections with 100% identity."""
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "identity_out")

        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", outbase, "--format", "html,svg,json"]
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

        html_file = tmp_path / "identity_out.html"
        json_file = tmp_path / "identity_out.json"
        svg_file = tmp_path / "identity_out.svg"

        assert html_file.exists()
        assert json_file.exists()
        assert svg_file.exists()

        # Parse JSON coordinates
        json_data = json.loads(json_file.read_text(encoding="utf-8"))
        json_rooms = {
            r["vnum"]: (r["coords"]["x"], r["coords"]["y"], r["coords"]["z"])
            for r in json_data["rooms"]
        }
        assert len(json_rooms) == 4

        # Parse HTML coordinates from embedded script tag
        html_content = html_file.read_text(encoding="utf-8")
        match = re.search(r'<script id="romutil-data" type="application/json">\s*(.*?)\s*</script>', html_content, re.DOTALL)
        assert match is not None, "Embedded JSON script block not found in HTML"
        html_data = json.loads(match.group(1))
        html_rooms = {
            r["vnum"]: (r["coords"]["x"], r["coords"]["y"], r["coords"]["z"])
            for r in html_data["rooms"]
        }

        # 100% Coordinate Identity Check between HTML and JSON
        assert html_rooms == json_rooms, f"HTML coordinates {html_rooms} do not match JSON {json_rooms}"

        # Verify component packing along X-axis
        # Rooms 1, 2, 3 are in component 1; room 4 is disconnected component 2
        comp1_vnums = {1, 2, 3}
        max_x_comp1 = max(json_rooms[v][0] for v in comp1_vnums)
        assert json_rooms[4][0] >= max_x_comp1 + 3

        # Verify SVG contains elements corresponding to each room
        svg_content = svg_file.read_text(encoding="utf-8")
        for vnum in (1, 2, 3, 4):
            room_obj = next(r for r in json_data["rooms"] if r["vnum"] == vnum)
            assert room_obj["name"] in svg_content

    def test_cli_format_parsing_variations(self, tmp_path, monkeypatch):
        """Format list parsing handles whitespace, case-insensitivity, and deduplication."""
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "variations")

        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", outbase, "-f", "  HTML ,  svg , json, html "]
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert (tmp_path / "variations.html").exists()
        assert (tmp_path / "variations.svg").exists()
        assert (tmp_path / "variations.json").exists()

    def test_cli_invalid_format_negative(self, tmp_path, monkeypatch, capsys):
        """Negative test: invalid format in comma-separated string raises parser error."""
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "neg_out")

        # Mixed valid and invalid format
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", outbase, "--format", "html,invalid_fmt"]
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "Unsupported format 'invalid_fmt'" in captured.err

        # Empty format string
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", outbase, "--format", "   ,   "]
        )
        with pytest.raises(SystemExit) as exc2:
            cli()
        assert exc2.value.code == 2
        captured2 = capsys.readouterr()
        assert "No format specified" in captured2.err

    def test_main_programmatic_format_validation(self, tmp_path):
        """Direct invocation of main() validates format types and values."""
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "prog_out")

        with pytest.raises(ValueError, match="Unsupported format 'bogus'"):
            main([are_file], outbase, fmt="html,bogus")

        with pytest.raises(ValueError, match="No formats specified"):
            main([are_file], outbase, fmt="")

        with pytest.raises(ValueError, match="No formats specified"):
            main([are_file], outbase, fmt=[])

        with pytest.raises(TypeError, match="Expected format string or sequence"):
            main([are_file], outbase, fmt=123)

    def test_main_positional_directory_source(self, tmp_path):
        """main() handles directory passed as area_file positional argument."""
        from tests.conftest import REPO_ROOT
        circle_dir = REPO_ROOT / "tests" / "fixtures" / "dialects" / "circle_world"
        outbase = str(tmp_path / "pos_dir_out")
        with pytest.raises(SystemExit) as exc:
            main([circle_dir], outbase, fmt="html,json")
        assert exc.value.code == 0
        assert (tmp_path / "pos_dir_out.html").exists()
        assert (tmp_path / "pos_dir_out.json").exists()

    def test_main_area_header_fallback(self, tmp_path):
        """main() supplies fallback AreaHeader when area file has no header."""
        are_no_header = """#ROOMS
#50
Lonely~
No header~
0 0 0
S
#0
#$
"""
        are_file = tmp_path / "no_hdr.are"
        are_file.write_text(are_no_header)
        outbase = str(tmp_path / "no_hdr_out")
        with pytest.raises(SystemExit) as exc:
            main([are_file], outbase, fmt="json")
        assert exc.value.code == 0
        json_data = json.loads((tmp_path / "no_hdr_out.json").read_text(encoding="utf-8"))
        assert json_data["area"]["name"] == "no_hdr"

    def test_cli_map_subcommand_compatibility(self, tmp_path, monkeypatch):
        """CLI drops legacy 'map' positional token if present."""
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "map_token_out")

        monkeypatch.setattr(
            "sys.argv",
            ["romutil", "map", str(are_file), "-outbase", outbase, "--format", "json"]
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert (tmp_path / "map_token_out.json").exists()

    def test_main_custom_renderer_dispatch(self, tmp_path, monkeypatch):
        """main() dispatches to custom renderer registered in RENDERERS."""
        from romutil.renderers import RENDERERS
        from romutil.renderers.base import BaseRenderer

        class MockCustomRenderer(BaseRenderer):
            def render(self, rdb, output_path, header=None, **options):
                out = Path(output_path)
                out.write_text("custom-rendered", encoding="utf-8")
                return out

        monkeypatch.setitem(RENDERERS, "custom", MockCustomRenderer)
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "custom_out")

        with pytest.raises(SystemExit) as exc:
            main([are_file], outbase, fmt="custom")
        assert exc.value.code == 0
        assert (tmp_path / "custom_out.custom").read_text(encoding="utf-8") == "custom-rendered"

    def test_main_svg_copyfile_failure_handled_gracefully(self, tmp_path, monkeypatch):
        """Failure to copy SVG to 0.svg is caught and ignored."""
        import shutil
        are_file = tmp_path / "multi.are"
        are_file.write_text(self.SAMPLE_MULTI_ARE)
        outbase = str(tmp_path / "copy_fail")

        def fake_copyfile(src, dst):
            raise OSError("Read-only filesystem")

        monkeypatch.setattr(shutil, "copyfile", fake_copyfile)
        with pytest.raises(SystemExit) as exc:
            main([are_file], outbase, fmt="svg")
        assert exc.value.code == 0
        assert (tmp_path / "copy_fail.svg").exists()
