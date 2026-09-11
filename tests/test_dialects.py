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
from romutil.parser import Parser, normalize_dialect_buffer, eval_flags, sanitize_ackmud_colour
from romutil.renderers import SVGRenderer, HTMLRenderer, JSONRenderer
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

    def test_parse_smaug_fixture(self):
        """Validate SMAUG format: 10-direction exits (D0-D9), #ROOMDATA, #AUTHOR, #RANGES, #RESETMSG, #ECONOMY, #REPAIRS."""
        filepath = FIXTURES_DIR / "smaug.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "Astral Sanctum"
        assert area.header.builder == "Altrag"
        assert area.header.filename == "astral_sanctum.are"

        assert len(area.rooms) == 11
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 500 in vnum_map
        nexus = vnum_map[500]
        assert nexus.name == "Central Nexus"
        assert len(nexus.exits) == 10

        # Check cardinal and vertical exits
        assert next(e for e in nexus.exits if e.direction == 0).dst_vnum == 501  # North
        assert next(e for e in nexus.exits if e.direction == 1).dst_vnum == 502  # East
        assert next(e for e in nexus.exits if e.direction == 2).dst_vnum == 503  # South
        assert next(e for e in nexus.exits if e.direction == 3).dst_vnum == 504  # West
        assert next(e for e in nexus.exits if e.direction == 4).dst_vnum == 505  # Up
        assert next(e for e in nexus.exits if e.direction == 5).dst_vnum == 506  # Down

        # Check diagonal exits
        assert next(e for e in nexus.exits if e.direction == 6).dst_vnum == 507  # Northeast
        assert next(e for e in nexus.exits if e.direction == 7).dst_vnum == 508  # Northwest
        assert next(e for e in nexus.exits if e.direction == 8).dst_vnum == 509  # Southeast
        assert next(e for e in nexus.exits if e.direction == 9).dst_vnum == 510  # Southwest

        # Check reciprocal diagonal exits
        assert next(e for e in vnum_map[507].exits if e.direction == 9).dst_vnum == 500  # SW -> 500
        assert next(e for e in vnum_map[508].exits if e.direction == 8).dst_vnum == 500  # SE -> 500
        assert next(e for e in vnum_map[509].exits if e.direction == 7).dst_vnum == 500  # NW -> 500
        assert next(e for e in vnum_map[510].exits if e.direction == 6).dst_vnum == 500  # NE -> 500

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


    def test_parse_ackmud_fixture(self):
        """Validate ACK!MUD 4.3 format: tagged #AREA header lines, @@ colour markup sanitization."""
        filepath = FIXTURES_DIR / "ackmud.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "The Citadel of Ack"
        assert area.header.builder == "Kline"
        assert area.header.vnum_min == 500
        assert area.header.vnum_max == 599

        assert len(area.rooms) == 5
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 500 in vnum_map
        assert 501 in vnum_map
        assert 502 in vnum_map
        assert 503 in vnum_map
        assert 504 in vnum_map

        room_500 = vnum_map[500]
        # Colour tokens @@R and @@y sanitized from title
        assert room_500.name == "Grand Entrance"
        # Colour tokens sanitized from description while preserving text
        assert "towering gates of the Citadel of Ack" in room_500.description
        assert "shimmering portal flickers faintly" in room_500.description
        assert "@@" not in room_500.description

        # Exits parsed cleanly
        assert len(room_500.exits) == 4
        north_exit = next(e for e in room_500.exits if e.direction == 0)
        assert north_exit.dst_vnum == 501
        assert north_exit.keyword == "iron gate"
        assert north_exit.flags == 1

        east_exit = next(e for e in room_500.exits if e.direction == 1)
        assert east_exit.dst_vnum == 502

        south_exit = next(e for e in room_500.exits if e.direction == 2)
        assert south_exit.dst_vnum == 503

        west_exit = next(e for e in room_500.exits if e.direction == 3)
        assert west_exit.dst_vnum == 504

        # Reciprocal exits verified
        room_501 = vnum_map[501]
        assert room_501.name == "Central Courtyard"
        assert "Flags flutter in the mountain wind" in room_501.description
        assert "@@" not in room_501.description
        assert next(e for e in room_501.exits if e.direction == 2).dst_vnum == 500

        # Resets parsed
        assert len(area.resets) == 1

    def test_parse_anatolia_fixture(self):
        """Validate ANATOLIA 3.0 format: #RESETMESSAGE, #FLAG, custom sectors, and exits."""
        filepath = FIXTURES_DIR / "anatolia.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))

        assert isinstance(area, AreaData)
        assert isinstance(area.header, AreaHeader)
        assert area.header.filename == "anatolia.are"
        assert area.header.name == "Anatolia Citadel"
        assert area.header.builder == "{ 1 90 } Ilya Anatolia Citadel"
        assert area.header.vnum_min == 100
        assert area.header.vnum_max == 199

        # Custom ANATOLIA 3.0 top-level attributes
        assert area.reset_message == "The cold wind howls across the Anatolian plains."
        assert area.flag == "reset_before arena"
        assert area["#RESETMESSAGE"] == "The cold wind howls across the Anatolian plains."
        assert area["#FLAG"] == "reset_before arena"

        # Rooms and exits
        assert len(area.rooms) == 5
        vnum_map = {r.vnum: r for r in area.rooms}
        assert 100 in vnum_map
        assert 101 in vnum_map
        assert 102 in vnum_map
        assert 103 in vnum_map
        assert 104 in vnum_map

        room_100 = vnum_map[100]
        assert room_100.name == "Citadel Gates"
        assert len(room_100.exits) == 4

        north_exit = next(e for e in room_100.exits if e.direction == 0)
        assert north_exit.dst_vnum == 101
        assert north_exit.keyword == "ironwood gate"
        assert north_exit.flags == 1

        # Resets
        assert len(area.resets) == 2


@pytest.mark.slow
@pytest.mark.integration
class TestDialectLayoutSolver:
    """Validate that area graphs from all 4 dialects solve cleanly and generate valid maps."""

    @pytest.mark.parametrize("fixture_name,expected_room_count", [
        ("rom24.are", 5),
        ("merc22.are", 4),
        ("envy20.are", 4),
        ("circlemud.are", 4),
        ("dikumud_alfa.wld", 5),
        ("ackmud.are", 5),
        ("anatolia.are", 5),
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


class TestAckMudCompatibility:
    """Comprehensive unit and regression tests for ACK!MUD / AckFUSS compatibility (Task 8f)."""

    def test_colour_sanitization_tokens(self):
        """Verify all standard ACK!MUD @@<char> colour tokens are cleanly stripped."""
        raw = "@@RRed @@yYellow @@bBlue @@gGreen @@cCyan @@mMagenta @@wWhite @@kBlack @@pPurple @@NReset"
        assert sanitize_ackmud_colour(raw) == "Red Yellow Blue Green Cyan Magenta White Black Purple Reset"

    def test_colour_sanitization_escaped_at(self):
        """Verify @@@ expands into a single literal @ and does not strip following text."""
        raw = "Contact @@Wadmin@@@ackmud.org@@N for support."
        assert sanitize_ackmud_colour(raw) == "Contact admin@ackmud.org for support."

    def test_colour_sanitization_ascii_layout_preservation(self):
        """Verify ASCII map grids, indentation, and alignment are preserved after colour stripping."""
        raw = (
            "  @@y+-----+@@N      @@b+-----+@@N\n"
            "  @@y|  N  |@@N ---- @@b|  S  |@@N\n"
            "  @@y+-----+@@N      @@b+-----+@@N\n"
        )
        expected = (
            "  +-----+      +-----+\n"
            "  |  N  | ---- |  S  |\n"
            "  +-----+      +-----+\n"
        )
        assert sanitize_ackmud_colour(raw) == expected

    def test_colour_sanitization_edge_cases(self):
        """Verify edge cases: empty strings, strings without @@, trailing @@, and punctuation."""
        assert sanitize_ackmud_colour("") == ""
        assert sanitize_ackmud_colour("No colours here") == "No colours here"
        assert sanitize_ackmud_colour("@@") == ""
        assert sanitize_ackmud_colour("Trailing @@\nNext line") == "Trailing \nNext line"
        assert sanitize_ackmud_colour("@@!Blinking @@2Dim @@iInverse@@N") == "Blinking Dim Inverse"
        assert sanitize_ackmud_colour("@@@") == "@"
        assert sanitize_ackmud_colour("@@@@") == "@@"
        assert sanitize_ackmud_colour(None) is None  # type: ignore

    def test_tagged_area_header_minimal(self):
        """Verify minimal ACK!MUD header with K and V tags parses into typed AreaHeader."""
        text = """#AREA
K Minimal Ack Zone~
V 1000 1050

#ROOMS
#1000
Start~
Desc~
0 0 1
S
#0
#$
"""
        area = Parser().parse(text)
        assert isinstance(area.header, AreaHeader)
        assert area.header.name == "Minimal Ack Zone"
        assert area.header.vnum_min == 1000
        assert area.header.vnum_max == 1050
        assert area.header.builder == "Unknown"
        assert area.header.filename == "minimal_ack_zone.are"

    def test_tagged_area_header_with_owner(self):
        """Verify tagged header with O tag extracts builder name."""
        text = """#AREA
Q 2
K Dragon Keep~
O Stephen~
V 6000 6099

#ROOMS
#6000
Keep Entry~
Desc~
0 0 1
S
#0
#$
"""
        area = Parser().parse(text)
        assert area.header.name == "Dragon Keep"
        assert area.header.builder == "Stephen"
        assert area.header.vnum_min == 6000
        assert area.header.vnum_max == 6099

    def test_tagged_area_header_level_fallback(self):
        """Verify tagged header without O tag falls back to L (levels) tag for builder."""
        text = """#AREA
Q 1
K Goblin Cave~
L { 5 15 } Caves~
V 7000 7050

#ROOMS
#7000
Cave Mouth~
Desc~
0 0 1
S
#0
#$
"""
        area = Parser().parse(text)
        assert area.header.name == "Goblin Cave"
        assert area.header.builder == "{ 5 15 } Caves"
        assert area.header.vnum_min == 7000
        assert area.header.vnum_max == 7050

    def test_tagged_area_header_inline_and_standalone_filename(self):
        """Verify inline filename and standalone filename under #AREA."""
        text_inline = """#AREA custom_ack.are~
Q 1
K Custom Realm~
V 8000 8050
O Kline~

#ROOMS
#8000
Start~
Desc~
0 0 1
S
#0
#$
"""
        area_inline = Parser().parse(text_inline)
        assert area_inline.header.filename == "custom_ack.are"
        assert area_inline.header.name == "Custom Realm"

        text_standalone = """#AREA
standalone_ack.are~
Q 1
K Standalone Realm~
V 8100 8150
O Kline~

#ROOMS
#8100
Start~
Desc~
0 0 1
S
#0
#$
"""
        area_standalone = Parser().parse(text_standalone)
        assert area_standalone.header.filename == "standalone_ack.are"
        assert area_standalone.header.name == "Standalone Realm"

    def test_tagged_area_header_with_end_delimiter_and_comments(self):
        """Verify tagged header containing End sentinel and comments parses cleanly."""
        text = """#AREA
* Authentic ACK!MUD area header comment
Q 5
K Dark Forest~
* Another comment
O Alander~
V 9000 9099
End

#ROOMS
#9000
Forest Edge~
Desc~
0 0 1
S
#0
#$
"""
        area = Parser().parse(text)
        assert area.header.name == "Dark Forest"
        assert area.header.builder == "Alander"
        assert area.header.vnum_min == 9000
        assert area.header.vnum_max == 9099

    def test_ackmud_render_all_formats(self, tmp_path):
        """Verify solved ACK!MUD area renders seamlessly to SVG, HTML, and JSON."""
        filepath = FIXTURES_DIR / "ackmud.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=10)

        # SVG Rendering
        svg_file = tmp_path / "ackmud.svg"
        out_svg = SVGRenderer().render(solved_rdb, svg_file, header=area.header, exits=exits)
        assert out_svg.exists()
        svg_content = out_svg.read_text(encoding="utf-8")
        assert "<svg" in svg_content
        assert "Grand Entrance" in svg_content
        assert "Central Courtyard" in svg_content
        # Confirm no raw @@ tokens leaked into SVG
        assert "@@" not in svg_content

        # HTML Rendering
        html_file = tmp_path / "ackmud.html"
        out_html = HTMLRenderer().render(solved_rdb, html_file, header=area.header, exits=exits)
        assert out_html.exists()
        html_content = out_html.read_text(encoding="utf-8")
        assert "The Citadel of Ack" in html_content
        assert "Grand Entrance" in html_content
        assert "@@" not in html_content

        # JSON Rendering
        json_file = tmp_path / "ackmud.json"
        out_json = JSONRenderer().render(solved_rdb, json_file, header=area.header, exits=exits)
        assert out_json.exists()
        json_dict = json.loads(out_json.read_text(encoding="utf-8"))
        assert json_dict["area"]["name"] == "The Citadel of Ack"
        # builder not stored under area dict
        assert len(json_dict["rooms"]) == 5
        room_names = [r["name"] for r in json_dict["rooms"]]
        assert "Grand Entrance" in room_names
        assert all("@@" not in name for name in room_names)

    def test_tagged_area_header_edge_cases(self):
        """Verify edge cases: A tag, single VNUM, missing name derived from filename, and non-ACK single tag."""
        # 1. 'A' tag for filename, single VNUM, and missing name (derived from filename stem)
        text_a = """#AREA
A lost_valley.are~
Q 1
V 500
O Mystic~

#ROOMS
#500
Start~
Desc~
0 0 1
S
#0
#$
"""
        area_a = Parser().parse(text_a)
        assert area_a.header.filename == "lost_valley.are"
        assert area_a.header.name == "Lost Valley"
        assert area_a.header.vnum_min == 500
        assert area_a.header.vnum_max == 0
        assert area_a.header.builder == "Mystic"

        # 2. Header with only a single non-KVQ ACK tag should not match ACK header
        text_non_ack = """#AREA
M none~
#ROOMS
"""
        norm = normalize_dialect_buffer(text_non_ack)
        assert "M none~" in norm

class TestAnatoliaDialectFeatures:
    """Comprehensive test suite for ANATOLIA 3.0 section tolerance, layout, and rendering."""

    def test_anatolia_render_all_formats(self, tmp_path):
        """Verify solved ANATOLIA area renders seamlessly to SVG, HTML, and JSON."""
        filepath = FIXTURES_DIR / "anatolia.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=10)

        # SVG Rendering
        svg_file = tmp_path / "anatolia.svg"
        out_svg = SVGRenderer().render(solved_rdb, svg_file, header=area.header, exits=exits)
        assert out_svg.exists()
        svg_content = out_svg.read_text(encoding="utf-8")
        assert "<svg" in svg_content
        assert "Citadel Gates" in svg_content
        assert "Citadel Courtyard" in svg_content
        assert "Battle Arena" in svg_content

        # HTML Rendering
        html_file = tmp_path / "anatolia.html"
        out_html = HTMLRenderer().render(solved_rdb, html_file, header=area.header, exits=exits)
        assert out_html.exists()
        html_content = out_html.read_text(encoding="utf-8")
        assert "Anatolia Citadel" in html_content
        assert "Citadel Gates" in html_content
        assert "Battle Arena" in html_content

        # JSON Rendering
        json_file = tmp_path / "anatolia.json"
        out_json = JSONRenderer().render(solved_rdb, json_file, header=area.header, exits=exits)
        assert out_json.exists()
        json_dict = json.loads(out_json.read_text(encoding="utf-8"))
        assert json_dict["area"]["name"] == "Anatolia Citadel"
        assert len(json_dict["rooms"]) == 5
        room_names = [r["name"] for r in json_dict["rooms"]]
        assert "Citadel Gates" in room_names
        assert "Eastern Steppe" in room_names

    def test_standalone_resetmessage_variations(self):
        """Test inline, multiline, and whitespace-padded #RESETMESSAGE variations."""
        # 1. Inline single-line
        text_inline = """#AREA
inline.are~
Inline~
Author~
10 20

#RESETMESSAGE You hear a hawk screeching in the distance.~

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        area1 = Parser().parse(text_inline)
        assert area1.reset_message == "You hear a hawk screeching in the distance."
        assert area1.flag is None

        # 2. Multiline message with internal formatting
        text_multiline = """#AREA
multi.are~
Multi~
Author~
10 20

#RESETMESSAGE
The wind sweeps across the steppes,
bringing the sharp chill of winter.~

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        area2 = Parser().parse(text_multiline)
        assert area2.reset_message == "The wind sweeps across the steppes,\nbringing the sharp chill of winter."

    def test_standalone_flag_variations(self):
        """Test alphanumeric, numeric bitvector, and piped flag variations."""
        # 1. Alphanumeric flag words
        text_alpha = """#AREA
flags.are~
Flags~
Author~
10 20

#FLAG reset_before arena battle_arena

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        area_alpha = Parser().parse(text_alpha)
        assert area_alpha.flag == "reset_before arena battle_arena"

        # 2. Numeric bitvector
        text_num = """#AREA
num.are~
Num~
Author~
10 20

#FLAG 2048

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        area_num = Parser().parse(text_num)
        assert area_num.flag == "2048"

        # 3. Piped bitmasks normalized by preprocessor
        text_piped = """#AREA
piped.are~
Piped~
Author~
10 20

#FLAG 4|8|1024

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        area_piped = Parser().parse(text_piped)
        assert area_piped.flag == "1036"

        # 4. Empty flag section
        text_empty = """#AREA
empty.are~
Empty~
Author~
10 20

#FLAG

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        area_empty = Parser().parse(text_empty)
        assert area_empty.flag is None

    def test_section_ordering_and_areadata_mapping(self):
        """Verify arbitrary section ordering resilience and AreaData dict/iteration behavior."""
        text_reordered = """#AREA
order.are~
Reordered~
Author~
10 20

#FLAG arena

#RESETMESSAGE An ominous thunder rumbles.~

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0

#$
"""
        area = Parser().parse(text_reordered)
        assert area.flag == "arena"
        assert area.reset_message == "An ominous thunder rumbles."
        assert area["#FLAG"] == "arena"
        assert area["#RESETMESSAGE"] == "An ominous thunder rumbles."

        # Iteration yields custom sections
        section_dict = dict(list(area))
        assert section_dict["#RESETMESSAGE"] == "An ominous thunder rumbles."
        assert section_dict["#FLAG"] == "arena"
        assert len(section_dict) == 11

    def test_anatolia_negative_malformed_sections(self):
        """Verify that malformed sections or missing terminations raise exceptions."""
        # Truncated reset message without tilde delimiter
        bad_text = """#AREA
bad.are~
Bad~
Author~
10 20

#RESETMESSAGE Missing tilde terminator here

#ROOMS
#10
Room 10~
Desc~
0 0 1
S
#0
#$
"""
        with pytest.raises(Exception):
            Parser().parse(bad_text)


class TestSmaugCompatibility:
    """Comprehensive test suite for SMAUG and SmaugFUSS 10-direction compatibility."""

    def test_smaug_fixture_parsing(self):
        """Verify parsing of authentic SMAUG fixture with all 10 exit directions."""
        filepath = FIXTURES_DIR / "smaug.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))
        assert len(area.rooms) == 11
        nexus = next(r for r in area.rooms if r.vnum == 500)
        assert len(nexus.exits) == 10

        dirs = {e.direction for e in nexus.exits}
        assert dirs == set(range(10))

    def test_smaug_direction_model_and_inversions(self):
        """Verify 10-direction Direction enum values, inversions, and matrix mapping."""
        # Value assertions
        assert Direction.north == 0
        assert Direction.east == 1
        assert Direction.up == 2
        assert Direction.south == 3
        assert Direction.west == 4
        assert Direction.down == 5
        assert Direction.northeast == 6
        assert Direction.northwest == 7
        assert Direction.southeast == 8
        assert Direction.southwest == 9

        # Inversion assertions
        assert Direction.north.invert() == Direction.south
        assert Direction.south.invert() == Direction.north
        assert Direction.east.invert() == Direction.west
        assert Direction.west.invert() == Direction.east
        assert Direction.up.invert() == Direction.down
        assert Direction.down.invert() == Direction.up
        assert Direction.northeast.invert() == Direction.southwest
        assert Direction.southwest.invert() == Direction.northeast
        assert Direction.northwest.invert() == Direction.southeast
        assert Direction.southeast.invert() == Direction.northwest

        # Double inversion invariance for all 10 directions
        for d in Direction:
            assert d.invert().invert() == d

        # direction_matrix mapping
        from romutil.models import direction_matrix
        assert len(direction_matrix) == 10
        assert direction_matrix[0] == Direction.north
        assert direction_matrix[1] == Direction.east
        assert direction_matrix[2] == Direction.south
        assert direction_matrix[3] == Direction.west
        assert direction_matrix[4] == Direction.up
        assert direction_matrix[5] == Direction.down
        assert direction_matrix[6] == Direction.northeast
        assert direction_matrix[7] == Direction.northwest
        assert direction_matrix[8] == Direction.southeast
        assert direction_matrix[9] == Direction.southwest

        # Exit model equality and symmetric inversion
        e_ne = Exit(src=500, dst=507, direction=Direction.northeast)
        e_sw = Exit(src=507, dst=500, direction=Direction.southwest)
        assert e_ne == e_sw
        assert hash(e_ne) == hash(e_sw)

        e_nw = Exit(src=500, dst=508, direction=Direction.northwest)
        e_se = Exit(src=508, dst=500, direction=Direction.southeast)
        assert e_nw == e_se
        assert hash(e_nw) == hash(e_se)

        # ExitDef mapping with raw integer 6-9
        exit_def_6 = ExitDef(direction=6, dst_vnum=507)
        mapped_exit = Exit(exit_def_6, source=500)
        assert mapped_exit.direction == Direction.northeast

    def test_smaug_layout_solving_geometric_properties(self):
        """Verify solver assigns correct relative coordinates for cardinal, vertical, and diagonal exits."""
        filepath = FIXTURES_DIR / "smaug.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)

        nexus = solved_rdb[500]
        assert nexus.x is not None and nexus.y is not None and nexus.z is not None

        # Verify cardinal directions
        assert solved_rdb[501].x == nexus.x and solved_rdb[501].y > nexus.y and solved_rdb[501].z == nexus.z  # North
        assert solved_rdb[502].x > nexus.x and solved_rdb[502].y == nexus.y and solved_rdb[502].z == nexus.z  # East
        assert solved_rdb[503].x == nexus.x and solved_rdb[503].y < nexus.y and solved_rdb[503].z == nexus.z  # South
        assert solved_rdb[504].x < nexus.x and solved_rdb[504].y == nexus.y and solved_rdb[504].z == nexus.z  # West
        assert solved_rdb[505].x == nexus.x and solved_rdb[505].y == nexus.y and solved_rdb[505].z > nexus.z  # Up
        assert solved_rdb[506].x == nexus.x and solved_rdb[506].y == nexus.y and solved_rdb[506].z < nexus.z  # Down

        # Verify diagonal directions
        ne = solved_rdb[507]
        assert ne.x > nexus.x and ne.y > nexus.y and ne.z == nexus.z, f"Northeast failed: ({ne.x}, {ne.y})"

        nw = solved_rdb[508]
        assert nw.x < nexus.x and nw.y > nexus.y and nw.z == nexus.z, f"Northwest failed: ({nw.x}, {nw.y})"

        se = solved_rdb[509]
        assert se.x > nexus.x and se.y < nexus.y and se.z == nexus.z, f"Southeast failed: ({se.x}, {se.y})"

        sw = solved_rdb[510]
        assert sw.x < nexus.x and sw.y < nexus.y and sw.z == nexus.z, f"Southwest failed: ({sw.x}, {sw.y})"

    def test_smaug_rendering_diagonals(self, tmp_path):
        """Verify SVG, HTML, and JSON renderers handle areas with diagonal exits."""
        filepath = FIXTURES_DIR / "smaug.are"
        area = Parser().parse(filepath.read_text(encoding="utf-8"))
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)

        # SVG Rendering
        svg_file = tmp_path / "smaug.svg"
        out_svg = SVGRenderer().render(solved_rdb, svg_file, header=area.header, exits=exits)
        assert out_svg.exists()
        svg_content = out_svg.read_text(encoding="utf-8")
        assert "<svg" in svg_content
        assert "Central Nexus" in svg_content
        assert "Dawn Spire" in svg_content
        assert "Dusk Spire" in svg_content
        assert "Solstice Pavilion" in svg_content
        assert "Equinox Dome" in svg_content

        # HTML Rendering
        html_file = tmp_path / "smaug.html"
        out_html = HTMLRenderer().render(solved_rdb, html_file, header=area.header, exits=exits)
        assert out_html.exists()
        html_content = out_html.read_text(encoding="utf-8")
        assert "Astral Sanctum" in html_content
        assert "Dawn Spire" in html_content

        # JSON Rendering
        json_file = tmp_path / "smaug.json"
        out_json = JSONRenderer().render(solved_rdb, json_file, header=area.header, exits=exits)
        assert out_json.exists()
        json_dict = json.loads(out_json.read_text(encoding="utf-8"))
        assert json_dict["area"]["name"] == "Astral Sanctum"
        assert len(json_dict["rooms"]) == 11
        for room_data in json_dict["rooms"]:
            assert room_data["coords"]["x"] is not None
            assert room_data["coords"]["y"] is not None
            assert room_data["coords"]["z"] is not None

    def test_smaug_fussarea_normalization(self):
        """Verify SmaugFUSS #FUSSAREA header normalization and parsing."""
        fuss_text = """#FUSSAREA
#AREADATA
Version      2
Name         FUSS Astral~
Author       SmaugDev~
Vnums        600 699
Ranges       0 65 0 65
Economy      0 12500000
ResetMsg     The fabric of space shivers.~
ResetFreq    15
Flags        0
End

#ROOMDATA
#600
Fuss Sanctum~
The crystalline walls pulse with arcane energy.~
0 0 1
D6
A diagonal trail leads northeast.~
~
0 0 601
S
#601
Northeast Alcove~
A quiet alcove on the northeastern perimeter.~
0 0 1
D9
The trail leads southwest back to the sanctum.~
~
0 0 600
S
#0
#$
"""
        area = Parser().parse(fuss_text)
        assert area.header is not None
        assert area.header.name == "FUSS Astral"
        assert area.header.builder == "SmaugDev"
        assert area.header.vnum_min == 600
        assert area.header.vnum_max == 699
        assert len(area.rooms) == 2

    def test_smaug_standalone_fussarea_block(self):
        """Verify #FUSSAREA as a standalone key-value block without preceding #AREADATA."""
        fuss_standalone = """#FUSSAREA
Name Standalone FUSS~
Author Kline~
Vnums 700 799
End

#ROOMS
#700
Standalone Chamber~
A simple test room.~
0 0 1
S
#0
#$
"""
        area = Parser().parse(fuss_standalone)
        assert area.header is not None
        assert area.header.name == "Standalone FUSS"
        assert area.header.builder == "Kline"
        assert area.header.vnum_min == 700
        assert len(area.rooms) == 1

    def test_smaug_non_spatial_section_stripping(self):
        """Verify SMAUG non-spatial sections (#AUTHOR, #RANGES, #RESETMSG, #FLAGS, #ECONOMY, #REPAIRS) are stripped."""
        text = """#AREA SMAUG Strip Test~
#AUTHOR Derek~
#RANGES 1 50 1 50
#RESETMSG A sudden tremor shakes the ground.~
#FLAGS 12
#ECONOMY 100 500000
#REPAIRS
0
#ROOMDATA
#800
Sanctuary~
A peaceful sanctuary.~
0 0 1
S
#0
#$
"""
        area = Parser().parse(text)
        assert area.header is not None
        assert area.header.name == "SMAUG Strip Test"
        assert area.header.builder == "Derek"
        assert len(area.rooms) == 1
        assert area.rooms[0].vnum == 800

    def test_diagonal_one_way_exit_solving(self, tmp_path):
        """Verify one-way diagonal exits position destination in the forward half-space and render correctly."""
        text = """#AREA OneWay Diagonal~
#ROOMS
#800
Hub~
Central room.~
0 0 1
D6
~
~
0 0 801
D7
~
~
0 0 802
D8
~
~
0 0 803
D9
~
~
0 0 804
S
#801
NE OneWay~
Destination northeast.~
0 0 1
S
#802
NW OneWay~
Destination northwest.~
0 0 1
S
#803
SE OneWay~
Destination southeast.~
0 0 1
S
#804
SW OneWay~
Destination southwest.~
0 0 1
S
#0
#$
"""
        area = Parser().parse(text)
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)

        hub = solved_rdb[800]
        ne = solved_rdb[801]
        nw = solved_rdb[802]
        se = solved_rdb[803]
        sw = solved_rdb[804]

        # One-way forward half-space constraints
        assert ne.x >= hub.x + 1 and ne.y >= hub.y + 1
        assert nw.x <= hub.x - 1 and nw.y >= hub.y + 1
        assert se.x >= hub.x + 1 and se.y <= hub.y - 1
        assert sw.x <= hub.x - 1 and sw.y <= hub.y - 1

        # Verify SVG rendering with one-way diagonal boundary exits
        svg_file = tmp_path / "oneway_diagonals.svg"
        out_svg = SVGRenderer().render(solved_rdb, svg_file, header=area.header, exits=exits)
        assert out_svg.exists()

    def test_diagonal_contradictory_cycle_relaxation(self):
        """Verify that contradictory diagonal cycles engage cut relaxation and solve feasibly."""
        text = """#AREA Diagonal Cycle~
#ROOMS
#900
Room 900~
Origin room.~
0 0 1
D6
~
~
0 0 901
S
#901
Room 901~
Second room.~
0 0 1
D1
~
~
0 0 902
S
#902
Room 902~
Third room creating impossible loop by heading northeast back to origin.~
0 0 1
D6
~
~
0 0 900
S
#0
#$
"""
        area = Parser().parse(text)
        rdb = {r.vnum: Room(r) for r in area.rooms}
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=25)

        # Solver must resolve without crashing and assign valid coordinates
        for v in (900, 901, 902):
            assert solved_rdb[v].x is not None
            assert solved_rdb[v].y is not None
            assert solved_rdb[v].z is not None

        # At least one exit should have been cut to relax the contradictory loop
        assert any(getattr(e, "one_way", False) or getattr(e, "cut", False) for e in exits)

    def test_graph_restore_rooms_diagonals(self):
        """Verify restore_rooms correctly restores coordinates for collapsed diagonal corridors."""
        from romutil.graph import restore_rooms

        parent = Room(vnum=1000, name="Center")
        parent.x = 10
        parent.y = 10
        parent.z = 5

        r_ne = Room(vnum=1001, name="NE Corridor")
        r_nw = Room(vnum=1002, name="NW Corridor")
        r_se = Room(vnum=1003, name="SE Corridor")
        r_sw = Room(vnum=1004, name="SW Corridor")

        parent.fixups = [
            (r_ne, Direction.northeast, 2),
            (r_nw, Direction.northwest, 3),
            (r_se, Direction.southeast, 4),
            (r_sw, Direction.southwest, 5),
        ]

        restored = restore_rooms(parent)
        assert len(restored) == 4

        assert r_ne.x == 10 + 2 and r_ne.y == 10 + 2 and r_ne.z == 5
        assert r_nw.x == 10 - 3 and r_nw.y == 10 + 3 and r_nw.z == 5
        assert r_se.x == 10 + 4 and r_se.y == 10 - 4 and r_se.z == 5
        assert r_sw.x == 10 - 5 and r_sw.y == 10 - 5 and r_sw.z == 5

    def test_smaug_roomdata_normalization_and_termination(self):
        """Verify #ROOMDATA normalization to #ROOMS and automatic #0 termination if missing."""
        # Missing #0 terminator
        raw_text = """#AREA AutoTerm~
#ROOMDATA
#1100
Lone Room~
A solitary chamber.~
0 0 1
S
#$
"""
        area = Parser().parse(raw_text)
        assert len(area.rooms) == 1
        assert area.rooms[0].vnum == 1100

    def test_diagonal_dummy_rooms_positioning_and_svg_stubs(self, tmp_path):
        """Verify position_dummy_rooms and SVGRenderer._line_coords handle all 4 diagonals."""
        from romutil.solver import position_dummy_rooms
        from romutil.renderers.svg import SVGRenderer

        # Test Pass 1: exits from non-dummy to dummy for all 4 diagonals
        r_core = Room(vnum=1, name="Core")
        r_core.x, r_core.y, r_core.z = 10, 10, 0

        dummies = {
            Direction.northeast: Room(vnum=2, name="NE Dummy"),
            Direction.northwest: Room(vnum=3, name="NW Dummy"),
            Direction.southeast: Room(vnum=4, name="SE Dummy"),
            Direction.southwest: Room(vnum=5, name="SW Dummy"),
        }
        for d in dummies.values():
            d.dummy = True

        rdb = {1: r_core, **{r.vnum: r for r in dummies.values()}}
        exits = [
            Exit(src=1, dst=dummies[d].vnum, direction=d)
            for d in (Direction.northeast, Direction.northwest, Direction.southeast, Direction.southwest)
        ]
        position_dummy_rooms(rdb, exits)

        assert dummies[Direction.northeast].x == 11 and dummies[Direction.northeast].y == 11
        assert dummies[Direction.northwest].x == 9 and dummies[Direction.northwest].y == 11
        assert dummies[Direction.southeast].x == 11 and dummies[Direction.southeast].y == 9
        assert dummies[Direction.southwest].x == 9 and dummies[Direction.southwest].y == 9

        # Test Pass 2: exits from dummy to non-dummy for all 4 diagonals
        dummies_rev = {
            Direction.northeast: Room(vnum=12, name="NE Rev"),
            Direction.northwest: Room(vnum=13, name="NW Rev"),
            Direction.southeast: Room(vnum=14, name="SE Rev"),
            Direction.southwest: Room(vnum=15, name="SW Rev"),
        }
        for d in dummies_rev.values():
            d.dummy = True

        rdb_rev = {1: r_core, **{r.vnum: r for r in dummies_rev.values()}}
        exits_rev = [
            Exit(src=dummies_rev[d].vnum, dst=1, direction=d)
            for d in (Direction.northeast, Direction.northwest, Direction.southeast, Direction.southwest)
        ]
        position_dummy_rooms(rdb_rev, exits_rev)

        assert dummies_rev[Direction.northeast].x == 9 and dummies_rev[Direction.northeast].y == 9
        assert dummies_rev[Direction.northwest].x == 11 and dummies_rev[Direction.northwest].y == 9
        assert dummies_rev[Direction.southeast].x == 9 and dummies_rev[Direction.southeast].y == 11
        assert dummies_rev[Direction.southwest].x == 11 and dummies_rev[Direction.southwest].y == 11

        # Test SVGRenderer renders all dummy diagonal stubs without exception
        svg_file = tmp_path / "dummy_stubs.svg"
        header = AreaHeader(filename="stubs.are", name="Stubs", builder="Tester", vnum_min=1, vnum_max=15)
        out_svg = SVGRenderer().render(rdb, svg_file, header=header, exits=exits)
        assert out_svg.exists()
