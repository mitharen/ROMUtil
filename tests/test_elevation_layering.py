import io
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pyomo.opt

from romutil.models import Direction, Room, Exit, RoomDef, ExitDef, AreaHeader, AreaData
from romutil.renderers.svg import SVGRenderer as Plotter, _DynamicPalette
from romutil.parser import Parser
from romutil.graph import graph
from romutil.cli import cli, main

SAMPLE_TOWER_ARE = """#AREA
tower.are~
Tower~
{ 1 10 } Wizard Tower~
100 102

#ROOMS
#100
Ground Floor~
Desc 0~
0 0 0
D4
Upstairs~
~
0 0 101
S
#101
First Floor~
Desc 1~
0 0 0
D4
Upstairs~
~
0 0 102
D5
Downstairs~
~
0 0 100
S
#102
Tower Top~
Desc 2~
0 0 0
D5
Downstairs~
~
0 0 101
S
#0

#$
"""


class TestSvgElevationGrouping:
    """Automated tests for SVG elevation grouping and interactive controls."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_elevation_layering_tower_are(self, tmp_path):
        """Verify elevation layering and group tags on tower.are."""
        return self.test_svg_elevation_grouping_tags_for_all_unique_z(tmp_path)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_svg_elevation_grouping_tags_for_all_unique_z(self, tmp_path):
        out_svg = str(tmp_path / "tower.svg")
        parsed = Parser().parse(SAMPLE_TOWER_ARE)
        rooms = [s[1] for s in parsed if s and s[0] == "#ROOMS"][0]
        rdb = {r.vnum: Room(r) for r in rooms}

        graph(rdb, out_svg, AreaHeader(filename="tower.are", name="Tower", builder="", vnum_min=100, vnum_max=102), split_levels=False)

        assert os.path.exists(out_svg)
        tree = ET.parse(out_svg)
        root = tree.getroot()

        # Assert <g id="elevation-{z}"> exists for every unique Z-coordinate (0, 1, 2)
        unique_zs = sorted(list(set(int(r.z) for r in rdb.values() if not r.dummy)))
        assert len(unique_zs) == 3
        assert unique_zs == [0, 1, 2]

        for z in unique_zs:
            layer = root.find(f".//*[@id='elevation-{z}']")
            assert layer is not None, f"Missing layer <g id='elevation-{z}'>"
            assert layer.attrib.get("class") == "elevation-layer"
            assert layer.attrib.get("data-z") == str(z)

            # Assert room rects inside this layer correspond to rooms at elevation z
            rects = layer.findall("{http://www.w3.org/2000/svg}rect")
            assert len(rects) >= 1

    def test_interactive_elevation_controls_embedded(self, tmp_path):
        out_svg = str(tmp_path / "controls.svg")
        r0 = Room(RoomDef(vnum=1, name="R0", description="Desc 0"))
        r0.x, r0.y, r0.z = 0, 0, 0
        r1 = Room(RoomDef(vnum=2, name="R1", description="Desc 1"))
        r1.x, r1.y, r1.z = 1, 0, 1
        rdb = {1: r0, 2: r1}

        plotter = Plotter(out_svg, rdb, [])
        plotter.plot()

        tree = ET.parse(out_svg)
        root = tree.getroot()

        # Verify embedded controls container
        ctrl_g = root.find(".//*[@id='elevation-controls']")
        assert ctrl_g is not None
        assert ctrl_g.attrib.get("class") == "elevation-controls"

        # Verify toggle buttons for each level
        btn0 = root.find(".//*[@id='toggle-btn-0']")
        btn1 = root.find(".//*[@id='toggle-btn-1']")
        assert btn0 is not None
        assert btn1 is not None
        assert "toggleElevation('0')" in btn0.attrib.get("onclick", "")
        assert "toggleElevation('1')" in btn1.attrib.get("onclick", "")

        # Verify SVG text contains stylesheet and script definitions
        with open(out_svg, "r", encoding="utf-8") as f:
            svg_text = f.read()
        assert ".elevation-layer" in svg_text
        assert "function toggleElevation" in svg_text

    def test_single_elevation_controls_and_group(self, tmp_path):
        out_svg = str(tmp_path / "single.svg")
        r0 = Room(RoomDef(vnum=1, name="Single Room", description="Alone"))
        r0.x, r0.y, r0.z = 0, 0, 0
        plotter = Plotter(out_svg, {1: r0}, [])
        plotter.plot()

        tree = ET.parse(out_svg)
        root = tree.getroot()
        layer0 = root.find(".//*[@id='elevation-0']")
        assert layer0 is not None
        assert layer0.attrib.get("data-z") == "0"


class TestMultiPlaneExport:
    """Automated tests for --split-levels CLI flag and multi-plane SVG generation."""

    def test_split_levels_generates_exact_n_files_with_outbase(self, tmp_path):
        are_file = tmp_path / "tower.are"
        are_file.write_text(SAMPLE_TOWER_ARE)

        outbase = str(tmp_path / "tower_split")
        with pytest.raises(SystemExit) as exc_info:
            main([are_file], outbase, split_levels=True)
        assert exc_info.value.code == 0

        # Assert exactly N=3 individual SVG files are generated
        svg_files = sorted(list(tmp_path.glob("tower_split*.svg")))
        assert len(svg_files) == 3
        expected_names = ["tower_split_z0.svg", "tower_split_z1.svg", "tower_split_z2.svg"]
        assert [f.name for f in svg_files] == expected_names

        # Neither combined tower_split.svg nor tower_split0.svg should exist
        assert not (tmp_path / "tower_split.svg").exists()
        assert not (tmp_path / "tower_split0.svg").exists()

        # Each generated file must contain <g id="elevation-{z}"> for its level
        for z, fpath in enumerate(svg_files):
            tree = ET.parse(str(fpath))
            root = tree.getroot()
            layer = root.find(f".//*[@id='elevation-{z}']")
            assert layer is not None
            assert layer.attrib.get("data-z") == str(z)

    def test_split_levels_cli_without_outbase(self, tmp_path, monkeypatch):
        are_file = tmp_path / "tower.are"
        are_file.write_text(SAMPLE_TOWER_ARE)

        # Run cli() without -outbase: should default to tower_z{z}.svg
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "--split-levels"])

        with pytest.raises(SystemExit) as exc_info:
            cli()
        assert exc_info.value.code == 0

        svg_files = sorted(list(tmp_path.glob("tower_z*.svg")))
        assert len(svg_files) == 3
        assert [f.name for f in svg_files] == ["tower_z0.svg", "tower_z1.svg", "tower_z2.svg"]

    def test_split_levels_multi_components(self, tmp_path):
        # Two disconnected components with exits
        r1 = Room(RoomDef(vnum=10, name="C1 R1", description="", exits=(ExitDef(direction=Direction.east.value, dst_vnum=11),)))
        r2 = Room(RoomDef(vnum=11, name="C1 R2", description="", exits=(ExitDef(direction=Direction.west.value, dst_vnum=10),)))
        r3 = Room(RoomDef(vnum=20, name="C2 R1", description="", exits=(ExitDef(direction=Direction.north.value, dst_vnum=21),)))
        r4 = Room(RoomDef(vnum=21, name="C2 R2", description="", exits=(ExitDef(direction=Direction.south.value, dst_vnum=20),)))

        outbase = str(tmp_path / "multi_comp")
        are_meta = AreaHeader(filename="test.are", name="MultiComp", builder="", vnum_min=10, vnum_max=21)

        graph({10: r1, 11: r2}, f"{outbase}0.svg", are_meta, split_levels=True, outbase=f"{outbase}0")
        graph({20: r3, 21: r4}, f"{outbase}1.svg", are_meta, split_levels=True, outbase=f"{outbase}1")

        comp0_files = list(tmp_path.glob("multi_comp0_z*.svg"))
        comp1_files = list(tmp_path.glob("multi_comp1_z*.svg"))
        assert len(comp0_files) >= 1
        assert len(comp1_files) >= 1


class TestDynamicColorPalette:
    """Automated tests for dynamic HSL gradient palette and arbitrarily deep elevations."""

    def test_deep_elevation_12_floors_no_index_error_and_no_clamping(self, tmp_path):
        out_svg = str(tmp_path / "deep_floors.svg")
        rdb = {}
        exits = []
        for z in range(12):
            vnum = 100 + z
            r = Room(RoomDef(vnum=vnum, name=f"Floor {z}", description=f"Room description {z}"))
            r.x, r.y, r.z = 0, 0, z
            rdb[vnum] = r
            if z > 0:
                ex = Exit(ExitDef(direction=Direction.up.value, dst_vnum=vnum), source=vnum - 1)
                exits.append(ex)

        plotter = Plotter(out_svg, rdb, exits)
        # Should not raise IndexError
        plotter.plot()

        assert os.path.exists(out_svg)
        tree = ET.parse(out_svg)
        root = tree.getroot()

        colors = []
        for z in range(12):
            layer = root.find(f".//*[@id='elevation-{z}']")
            assert layer is not None
            rect = layer.find("{http://www.w3.org/2000/svg}rect")
            assert rect is not None
            colors.append(rect.attrib["fill"])

        assert len(colors) == 12
        # Assert NO color clamping: every one of the 12 floors must have a distinct color
        assert len(set(colors)) == 12
        # Baseline clamped floors >= 6 to the same 'violet' color
        assert "violet" not in colors

    def test_dynamic_palette_protocol_and_len(self):
        rdb = {}
        for z in range(10):
            r = Room(RoomDef(vnum=100 + z, name=f"R{z}", description=""))
            r.x, r.y, r.z = 0, 0, z
            rdb[100 + z] = r

        plotter = Plotter("dummy.svg", rdb, [])
        palette = plotter.colors
        assert len(palette) == 10
        assert palette[0] == "hsl(0.0, 75%, 50%)"
        assert palette[9] == "hsl(300.0, 75%, 50%)"

        # Fallback for arbitrary elevation not in rdb
        fallback_color = palette[99]
        assert fallback_color.startswith("hsl(")

    def test_get_color_edge_cases(self):
        # Empty rdb
        plotter_empty = Plotter("dummy.svg", {}, [])
        assert plotter_empty.get_color(0) == "hsl(0, 75%, 50%)"
        assert plotter_empty.get_unique_elevations() == [0]

        # Single elevation
        r0 = Room(RoomDef(vnum=1, name="R", description=""))
        r0.x, r0.y, r0.z = 0, 0, 0
        plotter_single = Plotter("dummy.svg", {1: r0}, [])
        assert plotter_single.get_color(0) == "hsl(0, 75%, 50%)"


class TestPlotterEdgeCases:
    """Coverage and regression tests for Plotter projection edge cases."""

    def test_proj_exit_missing_src_or_none_coords(self):
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r1.x, r1.y, r1.z = 0, 0, 0
        plotter = Plotter("dummy.svg", {1: r1}, [])

        # ex.src not in rdb
        ex_unknown = Exit(ExitDef(direction=0, dst_vnum=1), source=999)
        assert plotter.proj_exit(ex_unknown) is None

        # ex.src has None coords
        r_none = Room(RoomDef(vnum=2, name="R2", description=""))
        r_none.x = None
        plotter.rdb[2] = r_none
        ex_none_src = Exit(ExitDef(direction=0, dst_vnum=1), source=2)
        assert plotter.proj_exit(ex_none_src) is None

        # ex.dst in rdb has None coords
        ex_none_dst = Exit(ExitDef(direction=0, dst_vnum=2), source=1)
        assert plotter.proj_exit(ex_none_dst) is None

    def test_proj_exit_unhandled_direction(self):
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r1.x, r1.y, r1.z = 0, 0, 0
        plotter = Plotter("dummy.svg", {1: r1}, [])

        # Direction that is not N, E, S, W, U, D
        ex_fake_dir = Exit(ExitDef(direction=0, dst_vnum=999), source=1)
        ex_fake_dir.direction = "custom"
        proj = plotter.proj_exit(ex_fake_dir)
        assert proj is not None
        # start == end
        assert proj[0] == proj[1]

    def test_plot_with_empty_or_all_none_rooms(self, tmp_path):
        out_svg = str(tmp_path / "empty.svg")
        r_none = Room(RoomDef(vnum=1, name="Void", description=""))
        r_none.x, r_none.y, r_none.z = None, None, None
        plotter = Plotter(out_svg, {1: r_none}, [])
        plotter.plot()
        assert os.path.exists(out_svg)

    def test_plot_with_target_z_and_exits(self, tmp_path):
        out_svg = str(tmp_path / "target_z.svg")
        r1 = Room(RoomDef(vnum=1, name="Level 0", description="", exits=(ExitDef(direction=Direction.up.value, dst_vnum=2),)))
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room(RoomDef(vnum=2, name="Level 1", description="", exits=(ExitDef(direction=Direction.down.value, dst_vnum=1),)))
        r2.x, r2.y, r2.z = 0, 0, 1
        ex = r1.exits[0]

        # Plot level 0 specifically
        plotter = Plotter(out_svg, {1: r1, 2: r2}, [ex], target_z=0)
        plotter.plot()

        tree = ET.parse(out_svg)
        root = tree.getroot()
        layer0 = root.find(".//*[@id='elevation-0']")
        assert layer0 is not None
        assert root.find(".//*[@id='elevation-1']") is None


class TestGraphAndCliEdgeCases:
    """Coverage and regression tests for graph and CLI edge cases."""

    def test_graph_solver_failure_branch(self, monkeypatch, caplog):
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=Direction.east.value, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=Direction.west.value, dst_vnum=1),)))
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0]]

        fake_results = MagicMock()
        fake_results.solver.termination_condition = pyomo.opt.TerminationCondition.infeasible
        graph_module = sys.modules["romutil.graph"]
        monkeypatch.setattr(graph_module, "solve", lambda r, e: (None, fake_results))

        graph(rdb, "dummy.svg", AreaHeader(filename="test.are", name="Fail", builder="", vnum_min=1, vnum_max=2))
        assert "Solver failed!" in caplog.text

    def test_cli_debug_flag(self, tmp_path, monkeypatch):
        are_file = tmp_path / "tower.are"
        are_file.write_text(SAMPLE_TOWER_ARE)
        outbase = str(tmp_path / "dbg_out")

        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "-outbase", outbase, "-d"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert os.path.exists(f"{outbase}0.svg")


    def test_plotter_none_projections_during_plot(self, tmp_path):
        out_svg = str(tmp_path / "none_proj.svg")
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=Direction.east.value, dst_vnum=2),)))
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=Direction.west.value, dst_vnum=1),)))
        # Set r2 coordinates to None to trigger projection is None during plot
        r2.x, r2.y, r2.z = None, None, None
        ex = r1.exits[0]

        plotter = Plotter(out_svg, {1: r1, 2: r2}, [ex])
        plotter.plot()
        assert os.path.exists(out_svg)

    def test_graph_hallway_asymmetric_exit(self, tmp_path):
        # Room 2 has 2 exits, but destination doesn't have reciprocal exit
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=Direction.east.value, dst_vnum=3), ExitDef(direction=Direction.west.value, dst_vnum=1))))
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))
        rdb = {1: r1, 2: r2, 3: r3}
        out_svg = str(tmp_path / "asym.svg")
        # Should not crash on hallway collapse check
        graph(rdb, out_svg, AreaHeader(filename="test.are", name="Asym", builder="", vnum_min=1, vnum_max=3))

    def test_cli_empty_sections_handled(self, tmp_path, monkeypatch):
        fake_parser = MagicMock()
        fake_parser.parse.return_value = AreaData(rooms=())
        cli_module = sys.modules["romutil.cli"]
        monkeypatch.setattr(cli_module, "Parser", lambda: fake_parser)

        are_file = tmp_path / "empty_sec.are"
        are_file.write_text("dummy")
        outbase = str(tmp_path / "empty_out")

        with pytest.raises(SystemExit) as exc:
            main([are_file], outbase)
        assert exc.value.code == 0


class TestElevationPaintersLayering:
    """Automated regression tests asserting painter's algorithm Z-order and depth sorting."""

    def test_svg_elevation_layer_strict_ascending_z_order(self, tmp_path):
        out_svg = str(tmp_path / "scrambled_floors.svg")
        # Insert rooms in scrambled order: Z=3, Z=0, Z=2, Z=-1, Z=1
        rdb = {}
        for z in [3, 0, 2, -1, 1]:
            vnum = 100 + z
            r = Room(RoomDef(vnum=vnum, name=f"Floor {z}", description=""))
            r.x, r.y, r.z = 0, 0, z
            rdb[vnum] = r

        plotter = Plotter(out_svg, rdb, [])
        plotter.plot()

        tree = ET.parse(out_svg)
        root = tree.getroot()
        layers = root.findall(".//*[@class='elevation-layer']")
        assert len(layers) == 5

        layer_zs = [int(layer.attrib.get("data-z")) for layer in layers]
        # Must be strictly sorted ascending: -1, 0, 1, 2, 3
        assert layer_zs == [-1, 0, 1, 2, 3]
        for i in range(len(layer_zs) - 1):
            assert layer_zs[i] < layer_zs[i + 1]

    def test_svg_elevation_layer_isometric_screen_depth_sorting(self, tmp_path):
        out_svg = str(tmp_path / "depth_sort.svg")
        rA = Room(RoomDef(vnum=10, name="Room A", description=""))
        rA.x, rA.y, rA.z = 1, 3, 0
        rB = Room(RoomDef(vnum=20, name="Room B", description=""))
        rB.x, rB.y, rB.z = 1, 1, 0
        rC = Room(RoomDef(vnum=30, name="Room C", description=""))
        rC.x, rC.y, rC.z = 2, 1, 0

        # Insert out of order: B, C, A
        rdb = {20: rB, 30: rC, 10: rA}
        plotter = Plotter(out_svg, rdb, [])
        plotter.plot()

        tree = ET.parse(out_svg)
        root = tree.getroot()
        layer0 = root.find(".//*[@id='elevation-0']")
        assert layer0 is not None

        rects = layer0.findall("{http://www.w3.org/2000/svg}rect")
        assert len(rects) == 3
        rect_ys = [float(r.attrib["y"]) for r in rects]
        # A (y=3, furthest north / top of screen) has smallest screen y in projection
        # B (y=1) and C (y=1) have larger screen y
        assert rect_ys[0] < rect_ys[1]

    def test_inter_floor_exit_layering_and_occlusion(self, tmp_path):
        out_svg = str(tmp_path / "inter_floor.svg")
        r0 = Room(RoomDef(vnum=1, name="Ground", description="", exits=(ExitDef(direction=Direction.up.value, dst_vnum=2),)))
        r0.x, r0.y, r0.z = 0, 0, 0
        r1 = Room(RoomDef(vnum=2, name="Upper", description="", exits=(ExitDef(direction=Direction.down.value, dst_vnum=1),)))
        r1.x, r1.y, r1.z = 0, 0, 1
        ex = r0.exits[0]

        plotter = Plotter(out_svg, {1: r0, 2: r1}, [ex])
        plotter.plot()

        tree = ET.parse(out_svg)
        root = tree.getroot()

        # Elevation 0 must appear BEFORE Elevation 1
        layers = root.findall(".//*[@class='elevation-layer']")
        assert len(layers) == 2
        assert layers[0].attrib["data-z"] == "0"
        assert layers[1].attrib["data-z"] == "1"

        # Inter-floor exit connecting 0 and 1 must be in elevation-1 (the higher layer)
        layer0_lines = layers[0].findall("{http://www.w3.org/2000/svg}line")
        layer1_lines = layers[1].findall("{http://www.w3.org/2000/svg}line")
        assert len(layer0_lines) == 0
        assert len(layer1_lines) == 1

        # In elevation-1, exit line must appear BEFORE room rect, ensuring room geometry overlays the exit
        children = list(layers[1])
        line_idx = next(i for i, child in enumerate(children) if child.tag.endswith("line"))
        rect_idx = next(i for i, child in enumerate(children) if child.tag.endswith("rect"))
        assert line_idx < rect_idx


@pytest.mark.slow
@pytest.mark.integration
def test_elevation_layering_tower_are(tmp_path):
    """Module-level alias for tower.are elevation layering tests."""
    TestSvgElevationGrouping().test_svg_elevation_grouping_tags_for_all_unique_z(tmp_path)
