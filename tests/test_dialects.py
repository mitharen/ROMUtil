"""Comprehensive unit and integration tests for MUD dialect compatibility matrix.

Validates parsing and layout generation across:
- ROM 2.4 / QuickMUD
- Merc 2.1 / 2.2
- Envy 1.0 / 2.0
- DikuMUD III / CircleMUD
- DikuMUD Alfa / Gamma
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

import json
from romutil.models import AreaData, AreaHeader, Room, Exit, RoomDef, ExitDef, Direction
from romutil.parser import Parser, normalize_dialect_buffer, eval_flags
from romutil.graph import graph, solve_layout
from romutil.cli import cli, main


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "dialects"


class TestDialectParserPositive:
    """Positive test cases validating parsing of historical and modern MUD dialects."""

    def test_parse_rom24_fixture(self):
        """Validate standard ROM 2.4 baseline format parsing."""
        filepath = FIXTURES_DIR / "rom24.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "ROM 2.4 Bastion"
        assert area.header.builder == "{ 5 20 } Alander ROM Bastion"
        assert area.header.vnum_min == 100
        assert area.header.vnum_max == 199

        assert len(area.rooms) == 5
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 100 in vnum_map
        assert 101 in vnum_map

        room_100 = vnum_map[100]
        assert room_100.name == "Bastion Courtyard"
        assert room_100.sector == 1
        assert len(room_100.exits) == 4

        north_exit = next(e for e in room_100.exits if e.direction == 0)
        assert north_exit.dst_vnum == 101
        assert north_exit.keyword == "iron gate"
        assert north_exit.flags == 1

        assert len(area.resets) == 2
        assert len(area.specials) == 0

    def test_parse_merc22_fixture(self):
        """Validate Merc 2.1/2.2 format: single-line header, 5-field doors, piped flags."""
        filepath = FIXTURES_DIR / "merc22.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "Merc Outpost"
        assert area.header.builder == "Kahn"

        assert len(area.rooms) == 4
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 200 in vnum_map
        room_200 = vnum_map[200]
        assert room_200.name == "Outpost Entrance"

        # Piped bitmask flags 4|8|1024 should evaluate to composite bitmask 1036
        assert room_200.room_flags == 1036

        # Legacy 5-field door parameters parsed without error
        assert len(room_200.exits) == 2
        exit_north = next(e for e in room_200.exits if e.direction == 0)
        assert exit_north.dst_vnum == 201
        assert exit_north.key_vnum == 3005
        assert exit_north.flags == 1
        assert exit_north.keyword == "gate oak"

        # Comments inside resets and specials parsed gracefully
        assert len(area.resets) == 2

    def test_parse_envy20_fixture(self):
        """Validate Envy 1.0/2.0 format: #AREADATA block, #ROOMDATA, extended sectors, non-standard section skipping."""
        filepath = FIXTURES_DIR / "envy20.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "Envy Citadel"
        assert area.header.builder == "Kahn"
        assert area.header.vnum_min == 300
        assert area.header.vnum_max == 399

        assert len(area.rooms) == 4
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 300 in vnum_map
        room_300 = vnum_map[300]
        assert room_300.name == "Citadel Water Chamber"
        # Extended Envy sector (9 = underwater, 10 = air)
        assert room_300.sector == 9
        assert vnum_map[301].sector == 10

        assert len(room_300.exits) == 2
        assert room_300.exits[0].dst_vnum == 301
        assert room_300.exits[1].dst_vnum == 302

    def test_parse_circlemud_fixture(self):
        """Validate CircleMUD 3.x / DikuMUD III format: missing headers, 6-param room line, $~ EOF."""
        filepath = FIXTURES_DIR / "circlemud.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        # CircleMUD .wld standalone room files do not contain an #AREA header block
        assert area.header is None

        assert len(area.rooms) == 4
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 400 in vnum_map
        room_400 = vnum_map[400]
        assert room_400.name == "The Temple of Midgaard"
        assert room_400.room_flags == 156
        assert room_400.sector == 0

        assert len(room_400.exits) == 3
        exit_north = next(e for e in room_400.exits if e.direction == 0)
        assert exit_north.dst_vnum == 401
        assert exit_north.key_vnum == -1
        assert exit_north.flags == 1

    def test_parse_dikumud_alfa_fixture(self):
        """Validate DikuMUD Alfa / Gamma monolithic format: room 0 ('The Void'), Limbo, Temple, cross-zone exits."""
        filepath = FIXTURES_DIR / "dikumud_alfa.wld"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert area.header is None

        assert len(area.rooms) == 5
        vnum_map = {r.vnum: r for r in area.rooms}

        # Room 0 (The Void) in Zone 0
        assert 0 in vnum_map
        room_0 = vnum_map[0]
        assert room_0.name == "The Void"
        assert room_0.sector == 1
        assert room_0.room_flags == 0
        assert len(room_0.exits) == 1
        exit_up = room_0.exits[0]
        assert exit_up.direction == 4  # Up
        assert exit_up.dst_vnum == 3001

        # Room 1 (Limbo) in Zone 0
        assert 1 in vnum_map
        room_1 = vnum_map[1]
        assert room_1.name == "Limbo"
        assert len(room_1.exits) == 1
        assert room_1.exits[0].direction == 4  # Up to 0
        assert room_1.exits[0].dst_vnum == 0

        # Room 3001 (Temple of Midgaard) in Zone 30
        assert 3001 in vnum_map
        room_3001 = vnum_map[3001]
        assert room_3001.name == "The Temple of Midgaard"
        assert room_3001.sector == 0
        assert len(room_3001.exits) == 3

        exit_down = next(e for e in room_3001.exits if e.direction == 5)
        assert exit_down.dst_vnum == 0

        exit_north = next(e for e in room_3001.exits if e.direction == 0)
        assert exit_north.dst_vnum == 3002

        exit_south = next(e for e in room_3001.exits if e.direction == 2)
        assert exit_south.dst_vnum == 3005

        # Room 3002 and 3005 in Zone 30
        assert 3002 in vnum_map
        assert 3005 in vnum_map


class TestDialectLayoutSolver:
    """Validate that area graphs from all 4 dialects solve cleanly and generate valid maps."""

    @pytest.mark.parametrize("fixture_name,expected_room_count", [
        ("rom24.are", 5),
        ("merc22.are", 4),
        ("envy20.are", 4),
        ("circlemud.are", 4),
        ("dikumud_alfa.wld", 5),
    ])
    def test_solve_and_render_all_dialects(self, fixture_name, expected_room_count, tmp_path):
        """Ensure layout solver assigns coordinates and plots SVG for each dialect without errors."""
        filepath = FIXTURES_DIR / fixture_name
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        rdb = {r.vnum: Room(r) for r in area.rooms}
        assert len(rdb) == expected_room_count

        out_svg = tmp_path / f"{fixture_name}.svg"
        graph(rdb, str(out_svg), area.header)

        assert out_svg.exists()
        assert out_svg.stat().st_size > 0

        # Verify output SVG is valid XML
        tree = ET.parse(out_svg)
        root = tree.getroot()
        assert "svg" in root.tag.lower()

        # Verify all rooms have non-None coordinates
        for vnum, room in rdb.items():
            assert room.x is not None, f"Room {vnum} missing x coordinate"
            assert room.y is not None, f"Room {vnum} missing y coordinate"
            assert room.z is not None, f"Room {vnum} missing z coordinate"

    def test_dikumud_alfa_elevation_hierarchy(self):
        """Verify 3D elevation positioning: Limbo (z) < The Void (z) < Temple (z)."""
        filepath = FIXTURES_DIR / "dikumud_alfa.wld"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area.header)

        # Limbo (vnum 1) exits UP to The Void (vnum 0)
        # The Void (vnum 0) exits UP to Temple (vnum 3001)
        assert solved_rdb[1].z < solved_rdb[0].z
        assert solved_rdb[0].z < solved_rdb[3001].z


class TestDialectHelperFunctions:
    """Unit tests for dialect normalization and flag evaluation helpers."""

    def test_eval_flags_int(self):
        assert eval_flags(0) == 0
        assert eval_flags(1024) == 1024

    def test_eval_flags_empty_or_none(self):
        assert eval_flags("") == 0
        assert eval_flags("   ") == 0
        assert eval_flags(None) is None

    def test_eval_flags_piped_numeric(self):
        assert eval_flags("4|8|1024") == 1036
        assert eval_flags("1|2") == 3
        assert eval_flags("0|1") == 1

    def test_eval_flags_alpha_and_mixed(self):
        assert eval_flags("ABCD") == "ABCD"
        assert eval_flags("A|B") == "A|B"

    def test_normalize_empty_input(self):
        assert normalize_dialect_buffer("") == ""
        assert normalize_dialect_buffer("   ") == "   "

    def test_normalize_areadata_with_filename(self):
        raw = """#AREADATA
FileName custom.are~
Name Custom Envy~
Author Tester~
VNUMs 100 200
End
#ROOMDATA
#100
Room~
Desc~
0 0 0
S
#0
#$
"""
        normalized = normalize_dialect_buffer(raw)
        assert "custom.are~" in normalized
        assert "Custom Envy~" in normalized
        assert "Tester~" in normalized
        assert "100 200" in normalized


class TestDialectParserNegative:
    """Negative test cases validating error handling and resiliency."""

    def test_corrupted_missing_end_marker(self):
        """Truncated dialect file without #$, $~, or $ must raise syntax error."""
        truncated = """#AREA
test.are~
Test~
Builder~
100 199

#ROOMS
#100
Room 100~
Desc 100~
0 0 0
S
#0
"""
        parser = Parser()
        with pytest.raises(Exception):
            parser.parse(truncated)

    def test_invalid_syntax_in_room_params(self):
        """Room definition with malformed parameters raises exception."""
        invalid_room = """#AREA
test.are~
Test~
Builder~
100 199

#ROOMS
#100
Room 100~
Desc 100~
INVALID_PARAMS
S
#0
#$
"""
        parser = Parser()
        with pytest.raises(Exception):
            parser.parse(invalid_room)

    def test_door_with_invalid_direction(self):
        """Door definition with invalid non-numeric direction raises exception."""
        invalid_door = """#AREA
test.are~
Test~
Builder~
100 199

#ROOMS
#100
Room 100~
Desc 100~
0 0 0
DINVALID
North door~
door~
1 0 101
S
#0
#$
"""
        parser = Parser()
        with pytest.raises(Exception):
            parser.parse(invalid_door)


class TestDikuMudAlfaFeatures:
    """Comprehensive tests for DikuMUD Alfa monolithic format, sentinels, VNUM 0 node, and CLI integration."""

    @pytest.mark.parametrize("sentinel", [
        "#99999\n$~",
        "#0\n$~",
        "#99999\n$",
        "#99999",
        "$~",
        "$",
    ])
    def test_dikumud_sentinel_variations(self, sentinel):
        """Validate parsing across all documented DikuMUD termination sentinels."""
        raw_wld = f"""#0
The Void~
Swirling mist.~
0 0 1
D4
Up to temple.~
~
0 -1 3001
S
#3001
The Temple~
Marble sanctuary.~
30 0 0
D5
Down to void.~
~
0 -1 0
S
{sentinel}
"""
        area = Parser().parse(raw_wld)
        assert len(area.rooms) == 2
        vnums = [r.vnum for r in area.rooms]
        assert vnums == [0, 3001]

    def test_room_zero_domain_models_and_graph(self):
        """Validate RoomDef, Room, Exit models and bidirectional exit resolution with room 0."""
        r0_def = RoomDef(vnum=0, name="The Void", description="Mist", sector=1, exits=(ExitDef(direction=4, dst_vnum=3001),))
        r3001_def = RoomDef(vnum=3001, name="Temple", description="Pillars", sector=0, exits=(ExitDef(direction=5, dst_vnum=0),))

        r0 = Room(r0_def)
        r3001 = Room(r3001_def)

        assert r0.vnum == 0
        assert r0.exits[0].src == 0
        assert r0.exits[0].dst == 3001
        assert r0.exits[0].direction == Direction.up

        assert r3001.vnum == 3001
        assert r3001.exits[0].src == 3001
        assert r3001.exits[0].dst == 0
        assert r3001.exits[0].direction == Direction.down

        # Exit containment and hashing
        ex0 = r0.exits[0]
        assert 0 in ex0
        assert 3001 in ex0
        assert hash(ex0) == hash(r3001.exits[0])

        # Layout solve
        rdb = {0: r0, 3001: r3001}
        solved, exits = solve_layout(rdb)

        # Bidirectional exits properly resolved (one_way is False)
        assert not solved[0].exits[0].one_way
        assert not solved[3001].exits[0].one_way
        assert solved[0].z < solved[3001].z

    def test_cli_invocation_with_wld_file(self, tmp_path, monkeypatch):
        """Validate CLI execution passing monolithic .wld file."""
        wld_content = (FIXTURES_DIR / "dikumud_alfa.wld").read_text(encoding="utf-8")
        test_file = tmp_path / "tinyworld.wld"
        test_file.write_text(wld_content, encoding="utf-8")

        outbase = str(tmp_path / "alfa_out")
        monkeypatch.setattr("sys.argv", ["romutil", str(test_file), "-outbase", outbase, "-f", "json"])

        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

        json_file = tmp_path / "alfa_out.json"
        assert json_file.exists()

        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["area"]["name"] == "tinyworld"
        assert data["area"]["file"] == "tinyworld.wld"
        assert len(data["rooms"]) == 5
        assert data["rooms"][0]["vnum"] == 0
        assert data["rooms"][0]["name"] == "The Void"

    def test_cli_invocation_with_map_subcommand(self, tmp_path, monkeypatch):
        """Validate CLI execution with 'romutil map <file.wld>' syntax."""
        wld_content = (FIXTURES_DIR / "dikumud_alfa.wld").read_text(encoding="utf-8")
        test_file = tmp_path / "alfa.wld"
        test_file.write_text(wld_content, encoding="utf-8")

        outbase = str(tmp_path / "map_out")
        monkeypatch.setattr("sys.argv", ["romutil", "map", str(test_file), "-outbase", outbase])

        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

        svg_files = list(tmp_path.glob("map_out*.svg"))
        assert len(svg_files) >= 1
        assert svg_files[0].stat().st_size > 0

    def test_diku_room_zero_with_leading_comments(self):
        """Validate parsing when comments precede room 0 and between rooms."""
        commented_wld = """* Tinyworld zone 0
* The Void and Limbo
#0
The Void~
You float in nothingness.~
0 0 1
D4
Up exit.~
~
0 -1 3001
S
* Temple zone 30
#3001
The Temple~
Temple.~
30 0 0
D5
Down exit.~
~
0 -1 0
S
#99999
$~
"""
        area = Parser().parse(commented_wld)
        assert len(area.rooms) == 2
        assert area.rooms[0].vnum == 0
        assert area.rooms[0].name == "The Void"
        assert area.rooms[1].vnum == 3001
        assert area.rooms[1].name == "The Temple"

    def test_negative_malformed_room_zero_params(self):
        """Malformed room flags/sector in room 0 raises syntax error."""
        corrupt_params = """#0
The Void~
Desc~
NON_NUMERIC_PARAMS
S
#99999
$~
"""
        with pytest.raises(Exception):
            Parser().parse(corrupt_params)

    def test_negative_truncated_room_zero_missing_s(self):
        """Room 0 without terminating S delimiter raises syntax error."""
        truncated = """#0
The Void~
Desc~
0 0 1
D4
Up~
~
0 -1 3001
#3001
Temple~
Desc~
30 0 0
S
#99999
$~
"""
        with pytest.raises(Exception):
            Parser().parse(truncated)
