"""Comprehensive test suite for CircleMUD 3.x and tbaMUD split world loading.

Validates:
- Directory discovery across flat and hierarchical lib/world structures
- Header and reset command parsing from .zon files
- Matching of .wld rooms to .zon metadata by filename prefix and VNUM range
- Cohesive AreaData model synthesis
- End-to-end MILP layout solving and SVG/JSON/HTML export
- CLI invocation via directory paths and --circle-dir flag
- Robust error handling for edge cases (missing files, empty dirs, syntax errors)
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from romutil.models import AreaData, AreaHeader, ExitDef, ResetDef, Room
from romutil.parser import (
    Parser,
    parse_circlemud_directory,
    parse_circlemud_zone_file,
)
from romutil.graph import solve_layout, graph
from romutil.renderers import build_area_json, export_json, export_html
from romutil.cli import cli, main


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dialects" / "circle_world"


class TestCircleMudParserPositive:
    """Positive test cases verifying split world directory parsing."""

    def test_parse_authentic_split_fixture(self):
        """Verify parsing authentic CircleMUD split world fixture."""
        area = parse_circlemud_directory(FIXTURE_DIR)

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "Northern Midgaard"
        assert area.header.filename == "30.zon"
        assert area.header.builder == "Unknown"
        assert area.header.vnum_min <= 3001
        assert area.header.vnum_max >= 3005

        # 5 rooms defined in 30.wld
        assert len(area.rooms) == 5
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 3001 in vnum_map
        assert 3002 in vnum_map
        assert 3003 in vnum_map
        assert 3004 in vnum_map
        assert 3005 in vnum_map

        temple = vnum_map[3001]
        assert temple.name == "The Temple of Midgaard"
        assert temple.room_flags == 156
        assert temple.sector == 0
        assert len(temple.exits) == 3

        # Exits from temple: north to sanctum (3002), east to altar (3004), south to square (3003)
        north_exit = next(e for e in temple.exits if e.direction == 0)
        assert north_exit.dst_vnum == 3002
        assert north_exit.flags == 1
        assert north_exit.keyword == "door cedar"

        east_exit = next(e for e in temple.exits if e.direction == 1)
        assert east_exit.dst_vnum == 3004

        south_exit = next(e for e in temple.exits if e.direction == 2)
        assert south_exit.dst_vnum == 3003

        # Resets parsed from 30.zon
        assert len(area.resets) == 3
        cmd_types = [r.command for r in area.resets]
        assert cmd_types == ["M", "O", "D"]
        assert area.resets[0].args == (0, 3010, 1, 3001)
        assert area.resets[1].args == (0, 3001, 1, 3004)
        assert area.resets[2].args == (0, 3001, 0, 1)

    def test_parse_hierarchical_lib_world_structure(self, tmp_path):
        """Verify discovery within standard CircleMUD lib/world/{wld,zon} layout."""
        world_dir = tmp_path / "lib" / "world"
        wld_dir = world_dir / "wld"
        zon_dir = world_dir / "zon"
        wld_dir.mkdir(parents=True)
        zon_dir.mkdir(parents=True)

        # Copy fixture files
        (wld_dir / "30.wld").write_text((FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"), encoding="latin-1")
        (zon_dir / "30.zon").write_text((FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1")

        # Create index manifest
        (wld_dir / "index").write_text("30.wld\n$\n", encoding="latin-1")

        area = parse_circlemud_directory(world_dir)
        assert isinstance(area, AreaData)
        assert len(area.rooms) == 5
        assert area.header is not None
        assert area.header.name == "Northern Midgaard"
        assert len(area.resets) == 3

    def test_parse_from_wld_subdirectory_directly(self, tmp_path):
        """Verify loading directly from wld/ directory with sibling ../zon/."""
        world_dir = tmp_path / "world"
        wld_dir = world_dir / "wld"
        zon_dir = world_dir / "zon"
        wld_dir.mkdir(parents=True)
        zon_dir.mkdir(parents=True)

        (wld_dir / "30.wld").write_text((FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"), encoding="latin-1")
        (zon_dir / "30.zon").write_text((FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1")

        area = parse_circlemud_directory(wld_dir)
        assert isinstance(area, AreaData)
        assert len(area.rooms) == 5
        assert area.header.name == "Northern Midgaard"

    def test_vnum_range_zone_matching(self, tmp_path):
        """Verify matching .zon to .wld by VNUM range when filename stems differ."""
        # Name the wld file differently than the zon file
        (tmp_path / "temple_precinct.wld").write_text(
            (FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"), encoding="latin-1"
        )
        (tmp_path / "30.zon").write_text(
            (FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1"
        )

        area = parse_circlemud_directory(tmp_path)
        assert isinstance(area, AreaData)
        assert area.header.name == "Northern Midgaard"
        assert len(area.rooms) == 5
        assert len(area.resets) == 3

    def test_multiple_zones_merging(self, tmp_path):
        """Verify merging multiple split zones into a cohesive AreaData model."""
        (tmp_path / "30.wld").write_text(
            (FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"), encoding="latin-1"
        )
        (tmp_path / "30.zon").write_text(
            (FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1"
        )

        # Add second zone 31
        zone31_wld = """#3101
Catacombs Entrance~
A dark stairway descends into the ancient catacombs below Midgaard.
~
31 1 0 0 0 0
D4
Up leads back to the temple square.~
~
0 -1 3003
S
$~
"""
        zone31_zon = """#31
Midgaard Catacombs~
3199 20 2
M 0 3105 2 3101  * Skeleton Guardian
S
$
"""
        (tmp_path / "31.wld").write_text(zone31_wld, encoding="latin-1")
        (tmp_path / "31.zon").write_text(zone31_zon, encoding="latin-1")

        area = parse_circlemud_directory(tmp_path)
        assert isinstance(area, AreaData)
        assert len(area.rooms) == 6  # 5 from zone 30 + 1 from zone 31
        assert len(area.resets) == 4  # 3 from zone 30 + 1 from zone 31
        assert area.header.vnum_min <= 3001
        assert area.header.vnum_max >= 3101

    def test_tbamud_zone_extended_format(self, tmp_path):
        """Verify parsing tbaMUD zone format with author line and extended numeric line."""
        tbamud_zon = """#30
DikuMUD~
Northern Midgaard~
3000 3099 15 2 d 0 0 0 1 33
* Resets
R 0 3000 3006 -1 \t(the teleporter)
O 0 3006 99 3000 \t(the teleporter)
M 0 3011 1 3000 \t(travelling saleswoman)
S
$
"""
        header, resets = parse_circlemud_zone_file(tbamud_zon)
        assert header.builder == "DikuMUD"
        assert header.name == "Northern Midgaard"
        assert header.vnum_min == 3000
        assert header.vnum_max == 3099

        assert len(resets) == 3
        assert resets[0].command == "R"
        assert resets[0].args == (0, 3000, 3006, -1)
        assert resets[0].comment == "the teleporter"

    def test_parse_circlemud_zone_file_standalone(self, tmp_path):
        """Verify standalone parse_circlemud_zone_file from file and string."""
        zon_file = tmp_path / "test.zon"
        zon_file.write_text(
            "#42\nTest Zone~\n4299 10 1\nS\n$\n", encoding="latin-1"
        )
        header, resets = parse_circlemud_zone_file(zon_file)
        assert header.name == "Test Zone"
        assert header.vnum_min == 4200
        assert header.vnum_max == 4299
        assert len(resets) == 0


@pytest.mark.slow
@pytest.mark.integration
class TestCircleMudLayoutAndExport:
    """Validate spatial layout solving and SVG/JSON/HTML rendering for split world."""

    def test_solver_and_svg_generation(self, tmp_path):
        """Verify layout solver computes collision-free coordinates and writes valid SVG."""
        area = parse_circlemud_directory(FIXTURE_DIR)
        rdb = {r.vnum: Room(r) for r in area.rooms}

        svg_path = tmp_path / "circle_midgaard.svg"
        graph(rdb, str(svg_path), area.header, split_levels=False)

        assert svg_path.exists()
        content = svg_path.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "</svg>" in content

        tree = ET.fromstring(content)
        assert tree.tag.endswith("svg")

        # Check for room elements and titles
        assert "The Temple of Midgaard" in content
        assert "Market Square" in content
        assert "Grand Altar" in content

    def test_json_and_html_export(self, tmp_path):
        """Verify JSON model build and HTML interactive viewer export."""
        area = parse_circlemud_directory(FIXTURE_DIR)
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rooms, _ = solve_layout(rdb, area.header)
        data = build_area_json(solved_rooms, area.header)

        assert data["area"]["name"] == "Northern Midgaard"
        assert len(data["rooms"]) == 5

        json_file = str(tmp_path / "circle.json")
        export_json(data, json_file)
        assert Path(json_file).exists()

        html_file = str(tmp_path / "circle.html")
        export_html(data, html_file)
        assert Path(html_file).exists()
        html_content = Path(html_file).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "Northern Midgaard" in html_content


class TestCircleMudCliIntegration:
    """Verify CLI behavior when invoking romutil on CircleMUD directories."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_positional_directory_argument(self, tmp_path, monkeypatch):
        """Invoking romutil with a directory positional argument loads split world."""
        outbase = str(tmp_path / "map_pos")
        monkeypatch.setattr("sys.argv", ["romutil", str(FIXTURE_DIR), "-outbase", outbase])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert (tmp_path / "map_pos0.svg").exists()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_circle_dir_flag(self, tmp_path, monkeypatch):
        """Invoking romutil with --circle-dir flag loads split world."""
        outbase = str(tmp_path / "map_flag")
        monkeypatch.setattr("sys.argv", ["romutil", "--circle-dir", str(FIXTURE_DIR), "-outbase", outbase])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert (tmp_path / "map_flag0.svg").exists()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_circle_dir_json_and_html(self, tmp_path, monkeypatch):
        """Invoking romutil with --circle-dir exports JSON and HTML."""
        out_json = str(tmp_path / "circle_map")
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", "--circle-dir", str(FIXTURE_DIR), "-outbase", out_json, "-f", "json"],
        )
        with pytest.raises(SystemExit) as exc1:
            cli()
        assert exc1.value.code == 0
        assert (tmp_path / "circle_map.json").exists()

        out_html = str(tmp_path / "circle_map")
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", "--circle-dir", str(FIXTURE_DIR), "-outbase", out_html, "-f", "html"],
        )
        with pytest.raises(SystemExit) as exc2:
            cli()
        assert exc2.value.code == 0
        assert (tmp_path / "circle_map.html").exists()

    def test_cli_default_outbase_with_circle_dir(self, tmp_path, monkeypatch):
        """Invoking romutil with --circle-dir without -outbase derives default outbase."""
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", "--circle-dir", str(FIXTURE_DIR)],
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

    def test_cli_no_args_raises_error(self, monkeypatch):
        """Invoking romutil with no areas and no --circle-dir exits with error."""
        monkeypatch.setattr("sys.argv", ["romutil"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code != 0


class TestCircleMudNegativeAndEdgeCases:
    """Negative and boundary test cases for CircleMUD directory and zone parsing."""

    def test_missing_zon_file_synthesizes_header(self, tmp_path):
        """Directory with .wld but missing .zon synthesizes default AreaHeader gracefully."""
        (tmp_path / "30.wld").write_text(
            (FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"), encoding="latin-1"
        )
        # No .zon file present
        area = parse_circlemud_directory(tmp_path)
        assert isinstance(area, AreaData)
        assert len(area.rooms) == 5
        assert area.header is not None
        assert area.header.name == "Zone 30"
        assert area.header.vnum_min == 3001
        assert area.header.vnum_max == 3005
        assert len(area.resets) == 0

    def test_missing_wld_file_raises_filenotfound(self, tmp_path):
        """Directory with only .zon but no .wld room files raises FileNotFoundError."""
        (tmp_path / "30.zon").write_text(
            (FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1"
        )
        with pytest.raises(FileNotFoundError, match="No CircleMUD .wld room files found"):
            parse_circlemud_directory(tmp_path)

    def test_empty_directory_raises_filenotfound(self, tmp_path):
        """Completely empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No CircleMUD .wld room files found"):
            parse_circlemud_directory(tmp_path)

    def test_nonexistent_directory_raises_filenotfound(self, tmp_path):
        """Non-existent directory path raises FileNotFoundError."""
        bad_path = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            parse_circlemud_directory(bad_path)

    def test_not_a_directory_raises_error(self, tmp_path):
        """Passing a file path instead of directory raises NotADirectoryError."""
        fake_file = tmp_path / "area.are"
        fake_file.write_text("not a dir", encoding="utf-8")
        with pytest.raises(NotADirectoryError, match="Path is not a directory"):
            parse_circlemud_directory(fake_file)

    def test_invalid_zon_header_missing_zone_num(self):
        """Zone file missing '#<zone_num>' raises ValueError."""
        invalid_zon = "Invalid Header Line\nNorthern Midgaard~\n3099 15 2\nS\n$\n"
        with pytest.raises(ValueError, match="expected '#<zone_num>'"):
            parse_circlemud_zone_file(invalid_zon)

    def test_invalid_zon_missing_numeric_line(self):
        """Zone file missing numeric parameters line raises ValueError."""
        invalid_zon = "#30\nNorthern Midgaard~\nS\n$\n"
        with pytest.raises(ValueError, match="missing numeric zone parameters line"):
            parse_circlemud_zone_file(invalid_zon)

    def test_invalid_wld_syntax_raises_exception(self, tmp_path):
        """Directory containing a malformed .wld file raises parse exception."""
        bad_wld = """#3001
Malformed Room~
Description~
MALFORMED_PARAMS
S
#0
"""
        (tmp_path / "30.wld").write_text(bad_wld, encoding="latin-1")
        (tmp_path / "30.zon").write_text((FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1")
        with pytest.raises(Exception):
            parse_circlemud_directory(tmp_path)

    def test_wld_with_no_rooms_raises_valueerror(self, tmp_path):
        """Directory containing .wld with no room records raises ValueError."""
        empty_wld = "#$\n"
        (tmp_path / "30.wld").write_text(empty_wld, encoding="latin-1")
        with pytest.raises(ValueError, match="No valid room definitions found"):
            parse_circlemud_directory(tmp_path)


class TestCircleMudParserCoverageExtensions:
    """Targeted coverage tests for edge branches in CircleMUD parser and CLI."""

    def test_zone_with_no_tilde_strings(self):
        """Zone with only zone number and numeric params gets default name."""
        zon_content = "#45\n4599 10 1\nS\n$\n"
        header, resets = parse_circlemud_zone_file(zon_content)
        assert header.name == "Zone 45"
        assert header.builder == "Unknown"
        assert header.vnum_min == 4500
        assert header.vnum_max == 4599

    def test_zone_with_single_numeric_param(self):
        """Zone with only top room number in numeric line."""
        zon_content = "#45\nZone 45 Name~\n4599\nS\n$\n"
        header, resets = parse_circlemud_zone_file(zon_content)
        assert header.name == "Zone 45 Name"
        assert header.vnum_max == 4599
        assert header.vnum_min == 4500

    def test_zone_with_inline_name_and_comments(self):
        """Zone with inline name on header line, comments, and non-numeric reset arguments."""
        zon_content = """* Pre-header comment
#46 Inline Name~
* Mid-header comment
4699 15 2
* Reset with non-numeric arg and empty line

V 0 1 4601 string_flag val
S
$
"""
        header, resets = parse_circlemud_zone_file(zon_content)
        assert header.name == "Inline Name"
        assert len(resets) == 1
        assert resets[0].command == "V"
        assert "string_flag" in resets[0].args

    def test_index_with_comments_and_root_fallback(self, tmp_path):
        """Test index file containing comment lines and file residing in parent root."""
        wld_dir = tmp_path / "wld"
        wld_dir.mkdir()
        # Put 30.wld in tmp_path (root), index in wld/
        (tmp_path / "30.wld").write_text((FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"), encoding="latin-1")
        (tmp_path / "30.zon").write_text((FIXTURE_DIR / "30.zon").read_text(encoding="latin-1"), encoding="latin-1")
        (wld_dir / "index").write_text("* Comment line in index\n30.wld\n$\n", encoding="latin-1")

        area = parse_circlemud_directory(tmp_path)
        assert len(area.rooms) == 5

    def test_cli_file_object_input(self, tmp_path):
        """Test CLI main() accepting an open file-like object with .name attribute."""
        import io
        stream = io.StringIO((FIXTURE_DIR / "30.wld").read_text(encoding="latin-1"))
        stream.name = "stream_area.wld"
        outbase = str(tmp_path / "stream_out")
        with pytest.raises(SystemExit) as exc:
            main([stream], outbase)
        assert exc.value.code == 0
