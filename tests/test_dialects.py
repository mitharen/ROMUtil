"""Comprehensive unit and integration tests for MUD dialect compatibility matrix.

Validates parsing and layout generation across:
- ROM 2.4 / QuickMUD
- Merc 2.1 / 2.2
- Envy 1.0 / 2.0
- DikuMUD III / CircleMUD
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from romutil.models import AreaData, AreaHeader, Room, ExitDef
from romutil.parser import Parser, normalize_dialect_buffer, eval_flags
from romutil.graph import graph


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


class TestDialectLayoutSolver:
    """Validate that area graphs from all 4 dialects solve cleanly and generate valid maps."""

    @pytest.mark.parametrize("fixture_name,expected_room_count", [
        ("rom24.are", 5),
        ("merc22.are", 4),
        ("envy20.are", 4),
        ("circlemud.are", 4),
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
