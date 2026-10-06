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
import pytest

from romutil.models import AreaData, AreaHeader, RoomDef, ExitDef, Direction
from romutil.parser import Parser, eval_flags, normalize_dialect_buffer, sanitize_ackmud_colour
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
        """Validate RoomDef and ExitDef models with room 0."""
        r0_def = RoomDef(vnum=0, name="The Void", description="Mist", sector=1, exits=(ExitDef(direction=4, dst_vnum=3001),))
        r3001_def = RoomDef(vnum=3001, name="Temple", description="Pillars", sector=0, exits=(ExitDef(direction=5, dst_vnum=0),))

        assert r0_def.vnum == 0
        assert r0_def.exits[0].dst_vnum == 3001
        assert r0_def.exits[0].direction == 4
        assert Direction(r0_def.exits[0].direction) == Direction.up

        assert r3001_def.vnum == 3001
        assert r3001_def.exits[0].dst_vnum == 0
        assert r3001_def.exits[0].direction == 5
        assert Direction(r3001_def.exits[0].direction) == Direction.down

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
        assert Direction.south == 2
        assert Direction.west == 3
        assert Direction.up == 4
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

        # Direction enum values and integer mapping for all 10 directions
        assert len(Direction) == 10
        assert Direction(0) == Direction.north
        assert Direction(1) == Direction.east
        assert Direction(2) == Direction.south
        assert Direction(3) == Direction.west
        assert Direction(4) == Direction.up
        assert Direction(5) == Direction.down
        assert Direction(6) == Direction.northeast
        assert Direction(7) == Direction.northwest
        assert Direction(8) == Direction.southeast
        assert Direction(9) == Direction.southwest

        # ExitDef mapping with raw integer 6-9
        exit_def_6 = ExitDef(direction=6, dst_vnum=507)
        assert exit_def_6.direction == 6
        assert Direction(exit_def_6.direction) == Direction.northeast
        exit_def_7 = ExitDef(direction=7, dst_vnum=508)
        assert Direction(exit_def_7.direction) == Direction.northwest


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
