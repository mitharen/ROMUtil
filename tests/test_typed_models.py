import dataclasses
import os
import pytest

from romutil.models import (
    AreaData,
    AreaHeader,
    Direction,
    Exit,
    ExtraDescr,
    Help,
    Mobile,
    Object,
    Reset,
    Room,
    Shop,
    Social,
    Special,
)
from romutil.parser import Parser
import importlib
cli_module = importlib.import_module("romutil.cli")

from tests.conftest import SAMPLE_AREAS_DIR


class TestTypedDataClasses:
    """Positive and negative tests for strongly-typed AST dataclasses."""

    def test_area_header(self):
        header = AreaHeader(
            filename="test.are",
            name="Test Area",
            builder="TestBuilder",
            vnum_min=100,
            vnum_max=200,
        )
        assert header.filename == "test.are"
        assert header.name == "Test Area"
        assert header.builder == "TestBuilder"
        assert header.vnum_min == 100
        assert header.vnum_max == 200

        # Strict typing: subscripting is prohibited
        with pytest.raises(TypeError):
            _ = header[0]  # type: ignore

        # Frozen immutability
        with pytest.raises(dataclasses.FrozenInstanceError):
            header.name = "New Name"  # type: ignore

    def test_exit(self):
        e1 = Exit(
            direction=0,
            dst_vnum=101,
            description="North door",
            keyword="door",
            key_vnum=1000,
            flags=1,
        )
        assert e1.direction == 0
        assert e1.dst_vnum == 101
        assert e1.description == "North door"
        assert e1.keyword == "door"
        assert e1.key_vnum == 1000
        assert e1.flags == 1

        # Subscripting is prohibited
        with pytest.raises(TypeError):
            _ = e1[0]  # type: ignore

        # Equality with tuple is strictly False
        assert e1 != (0, 101)
        assert not (e1 == (0, 101))
        assert not (e1 == "not an exit")

        # Equality with another Exit
        e2 = Exit(direction=0, dst_vnum=101, description="North door", keyword="door", key_vnum=1000, flags=1)
        assert e1 == e2
        e3 = Exit(direction=1, dst_vnum=101)
        assert e1 != e3

        # Hashing
        s = {e1, e2, e3}
        assert len(s) == 2

        # Frozen
        with pytest.raises(dataclasses.FrozenInstanceError):
            e1.dst_vnum = 200  # type: ignore

    def test_extra_descr(self):
        ext = ExtraDescr(keyword="sign", description="A wooden sign.")
        assert ext.keyword == "sign"
        assert ext.description == "A wooden sign."
        with pytest.raises(TypeError):
            _ = ext[0]  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            ext.keyword = "other"  # type: ignore

    def test_room(self):
        e1 = Exit(direction=0, dst_vnum=101)
        ext = ExtraDescr(keyword="plaque", description="A gold plaque.")
        # Test passing lists to verify __post_init__ converts to tuples
        room = Room(
            vnum=100,
            name="Hall",
            description="A grand hall.",
            room_flags=4,
            sector=1,
            exits=[e1],  # type: ignore
            extras=[ext],  # type: ignore
        )
        assert room.vnum == 100
        assert room.name == "Hall"
        assert room.description == "A grand hall."
        assert room.room_flags == 4
        assert room.sector == 1
        assert isinstance(room.exits, tuple)
        assert isinstance(room.extras, tuple)
        assert room.exits[0] == e1
        assert room.extras[0] == ext

        with pytest.raises(TypeError):
            _ = room[0]  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            room.name = "Other"  # type: ignore

    def test_mobile(self):
        mob = Mobile(
            vnum=100,
            player_name="goblin scout",
            short_desc="a goblin scout",
            long_desc="A goblin scout sneaks around.",
            desc="He looks nasty.",
            race="goblin",
            level=5,
            wealth=50,
            optionals=["opt1"],  # type: ignore
        )
        assert mob.vnum == 100
        assert mob.player_name == "goblin scout"
        assert mob.level == 5
        assert isinstance(mob.optionals, tuple)
        with pytest.raises(TypeError):
            _ = mob[0]  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            mob.vnum = 200  # type: ignore

    def test_object(self):
        obj = Object(
            vnum=200,
            name="sword rusty",
            short_desc="a rusty sword",
            desc="A rusty sword lies here.",
            item_type="weapon",
            values=[1, 2, 3, 4, 5],  # type: ignore
            level=3,
            weight=10,
            cost=25,
            optionals=["opt"],  # type: ignore
        )
        assert obj.vnum == 200
        assert obj.name == "sword rusty"
        assert obj.item_type == "weapon"
        assert isinstance(obj.values, tuple)
        assert isinstance(obj.optionals, tuple)
        with pytest.raises(TypeError):
            _ = obj[0]  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.cost = 50  # type: ignore

    def test_auxiliary_models(self):
        # Reset
        rst = Reset(command="M", args=(0, 100, 1, 100, 1), comment="* load mob")
        assert rst.command == "M"
        assert rst.args == (0, 100, 1, 100, 1)
        with pytest.raises(TypeError):
            _ = rst[0]  # type: ignore

        # Shop
        shop = Shop(keeper=100, buy_types=(1, 2, 3, 4, 5), profit_buy=120, profit_sell=80, open_hour=8, close_hour=20)
        assert shop.keeper == 100
        with pytest.raises(TypeError):
            _ = shop[0]  # type: ignore

        # Special
        spec = Special(command="M", vnum=100, spec_fun="spec_cast_mage", comment="* spec")
        assert spec.command == "M"
        assert spec.vnum == 100
        with pytest.raises(TypeError):
            _ = spec[0]  # type: ignore

        # Help
        help_entry = Help(level=1, keywords=["TEST", "HELP"], text="Help text")  # type: ignore
        assert isinstance(help_entry.keywords, tuple)
        assert help_entry.level == 1
        with pytest.raises(TypeError):
            _ = help_entry[0]  # type: ignore

        # Social
        soc = Social(name="dance", stages=["dance1", "dance2"])  # type: ignore
        assert isinstance(soc.stages, tuple)
        assert soc.name == "dance"
        with pytest.raises(TypeError):
            _ = soc[0]  # type: ignore

    def test_area_data(self):
        header = AreaHeader(filename="a.are", name="A", builder="B", vnum_min=1, vnum_max=10)
        r1 = Room(vnum=1, name="R1", description="D1")
        r2 = Room(vnum=2, name="R2", description="D2")
        area = AreaData(
            header=header,
            rooms=[r1, r2],  # type: ignore
        )
        assert area.header == header
        assert isinstance(area.rooms, tuple)
        assert len(area.rooms) == 2
        assert area.mobiles == ()
        assert area.objects == ()
        assert area.resets == ()
        assert area.shops == ()
        assert area.specials == ()
        assert area.helps == ()
        assert area.socials == ()
        assert area.reset_message is None
        assert area.flag is None

        # Verify legacy sequence/dict emulation methods are removed
        with pytest.raises(TypeError):
            _ = len(area)  # type: ignore
        with pytest.raises(TypeError):
            _ = area[0]  # type: ignore
        with pytest.raises(TypeError):
            _ = iter(area)  # type: ignore

    def test_direction_model_invariants(self):
        expected_dirs = [
            (0, Direction.north),
            (1, Direction.east),
            (2, Direction.south),
            (3, Direction.west),
            (4, Direction.up),
            (5, Direction.down),
            (6, Direction.northeast),
            (7, Direction.northwest),
            (8, Direction.southeast),
            (9, Direction.southwest),
        ]
        assert len(Direction) == 10
        for val, enum_member in expected_dirs:
            assert Direction(val) == enum_member
            assert enum_member.invert().invert() == enum_member

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

    def test_parser_returns_strongly_typed_ast(self):
        """Parser returns an AreaData instance with Room instances for all rooms."""
        parser = Parser()
        minimal_area = """#AREA
test_typed.are~
Typed Area~
{ 1 10 } Builder~
100 199

#ROOMS
#100
First Room~
Description one.~
0 0 0
D0
Door to 101~
~
0 0 101
S
#101
Second Room~
Description two.~
0 0 0
D2
Door to 100~
~
0 0 100
S
#0

#$
"""
        result = parser.parse(minimal_area)
        assert isinstance(result, AreaData)
        assert isinstance(result.header, AreaHeader)
        assert result.header.name == "Typed Area"
        assert len(result.rooms) == 2

        for r in result.rooms:
            assert isinstance(r, Room)
            assert not isinstance(r, tuple)
            for e in r.exits:
                assert isinstance(e, Exit)
                assert not isinstance(e, tuple)

    @pytest.mark.skipif(not os.path.isdir(SAMPLE_AREAS_DIR), reason="QuickMUD area files not available")
    def test_real_area_all_strongly_typed(self):
        filepath = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        parser = Parser()
        with open(filepath, "r", encoding="latin-1") as f:
            result = parser.parse(f.read())

        assert isinstance(result, AreaData)
        assert isinstance(result.header, AreaHeader)
        assert len(result.rooms) > 0
        for r in result.rooms:
            assert isinstance(r, Room)
            assert not isinstance(r, tuple)
            for e in r.exits:
                assert isinstance(e, Exit)
                assert not isinstance(e, tuple)

        for m in result.mobiles:
            assert isinstance(m, Mobile)
            assert not isinstance(m, tuple)

        for o in result.objects:
            assert isinstance(o, Object)
            assert not isinstance(o, tuple)

        for rst in result.resets:
            assert isinstance(rst, Reset)
            assert not isinstance(rst, tuple)

    def test_cli_legacy_fallback(self, monkeypatch, tmp_path):
        """Covers cli.main fallback when parser returns raw section tuples instead of AreaData."""
        class LegacyParser:
            def parse(self, content):
                return [
                    ('#AREA', ('legacy.are', 'Legacy Area', 'Builder', (1, 10))),
                    ('#ROOMS', [(100, 'Room 100', 'Desc', [(1, 101)]), (101, 'Room 101', 'Desc', [(4, 100)])]),
                ]

        monkeypatch.setattr(cli_module, "Parser", LegacyParser)
        test_file = tmp_path / "legacy.are"
        test_file.write_text("dummy")

        with pytest.raises(TypeError, match="Expected AreaData"):
            cli_module.main([test_file], str(tmp_path / "out"))
