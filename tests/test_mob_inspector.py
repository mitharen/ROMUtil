"""Unit and integration tests for mob inspector metadata preservation and viewer overlays."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from romutil.models import (
    AreaData,
    AreaHeader,
    Exit,
    Mobile,
    Object,
    Reset,
    Room,
    Shop,
    Special,
)
from romutil.parser import parse_file
from romutil.renderers import HTMLRenderer, build_area_json
from romutil.renderers.json import _format_dice, _serialize_flags
from romutil.types import Coord3D, SolvedLayout, WorldGraph


def test_world_graph_preserves_mobiles_objects_and_resets() -> None:
    """Verify that WorldGraph.from_area_data preserves mobiles, objects, resets, shops, and specials."""
    header = AreaHeader(
        filename="test.are",
        name="Test Domain",
        builder="TestBuilder",
        vnum_min=100,
        vnum_max=199,
    )
    room = Room(vnum=101, name="Entrance", description="The entrance chamber.", exits=())
    mobile = Mobile(
        vnum=150,
        player_name="guardian",
        short_desc="the guardian",
        long_desc="A towering guardian stands watch.",
        level=25,
        alignment=500,
    )
    obj = Object(
        vnum=160,
        name="halberd",
        short_desc="a gleaming halberd",
        desc="A heavy halberd rests here.",
    )
    reset_m = Reset(command="M", args=(0, 150, 1, 101, 1))
    reset_e = Reset(command="E", args=(1, 160, 1, 16))
    shop = Shop(keeper=150, buy_types=(1, 2, 0, 0, 0), profit_buy=110, profit_sell=50)
    special = Special(command="M", vnum=150, spec_fun="spec_guard")

    area = AreaData(
        header=header,
        rooms=(room,),
        mobiles=(mobile,),
        objects=(obj,),
        resets=(reset_m, reset_e),
        shops=(shop,),
        specials=(special,),
    )

    graph = WorldGraph.from_area_data(area)

    assert 150 in graph.mobiles
    assert graph.mobiles[150].player_name == "guardian"
    assert 160 in graph.objects
    assert graph.objects[160].name == "halberd"
    assert len(graph.resets) == 2
    assert graph.resets[0].command == "M"
    assert 150 in graph.shops
    assert graph.shops[150].profit_buy == 110
    assert 150 in graph.specials
    assert graph.specials[150].spec_fun == "spec_guard"


def test_world_graph_defaults() -> None:
    """Verify default empty mappings for mobiles, objects, resets, shops, and specials in WorldGraph."""
    graph = WorldGraph(rooms={}, exits=())
    assert graph.mobiles == {}
    assert graph.objects == {}
    assert graph.resets == ()
    assert graph.shops == {}
    assert graph.specials == {}


def test_build_area_json_mob_spawns_and_equipment() -> None:
    """Verify that build_area_json correctly records room mob spawns and mob equipment loadouts."""
    room = Room(vnum=201, name="Armory", description="Storage for weapons.", exits=())
    mob = Mobile(
        vnum=250,
        player_name="blacksmith",
        short_desc="the stout blacksmith",
        long_desc="A burly blacksmith works at the anvil.",
        level=18,
        race="dwarf",
        sex="male",
        alignment=350,
        act_flags="ABC",
        affected_by=0,
        hit=(4, "d8", 120),
        damage=(2, "d6", 10),
        dam_type="crush",
        ac_pierce=10,
        ac_bash=12,
        ac_slash=10,
        ac_exotic=15,
        wealth=500,
    )
    hammer = Object(
        vnum=301,
        name="hammer smithing",
        short_desc="a heavy smithing hammer",
        desc="A smithing hammer sits on the anvil.",
    )
    apron = Object(
        vnum=302,
        name="apron leather",
        short_desc="a leather apron",
        desc="A leather apron hangs from a peg.",
    )
    resets = (
        Reset(command="M", args=(0, 250, 1, 201, 1)),
        Reset(command="E", args=(1, 301, 1, 16)),  # Wielded hammer
        Reset(command="G", args=(1, 302, 1)),      # Carried apron
    )

    graph = WorldGraph(
        rooms={201: room},
        exits=(),
        mobiles={250: mob},
        objects={301: hammer, 302: apron},
        resets=resets,
    )
    layout = SolvedLayout(coordinates={201: Coord3D(0, 0, 0)})

    data = build_area_json(graph, layout)

    assert "mobiles" in data
    assert "250" in data["mobiles"]
    mob_data = data["mobiles"]["250"]

    assert mob_data["vnum"] == 250
    assert mob_data["name"] == "blacksmith"
    assert mob_data["short_desc"] == "the stout blacksmith"
    assert mob_data["level"] == 18
    assert mob_data["alignment"] == 350
    assert mob_data["hit"] == "4d8+120"
    assert mob_data["damage"] == "2d6+10"
    assert mob_data["dam_type"] == "crush"
    assert mob_data["ac_bash"] == 12
    assert mob_data["wealth"] == 500

    # Equipment verification
    assert len(mob_data["equipment"]) == 2
    eq0 = mob_data["equipment"][0]
    assert eq0["vnum"] == 301
    assert eq0["short_desc"] == "a heavy smithing hammer"
    assert eq0["wear_loc"] == 16

    eq1 = mob_data["equipment"][1]
    assert eq1["vnum"] == 302
    assert eq1["short_desc"] == "a leather apron"
    assert eq1["wear_loc"] == "inventory"

    # Room spawn verification
    assert len(data["rooms"]) == 1
    r_data = data["rooms"][0]
    assert "mobs" in r_data
    assert len(r_data["mobs"]) == 1
    assert r_data["mobs"][0]["vnum"] == 250
    assert r_data["mobs"][0]["count"] == 1
    assert r_data["mobs"][0]["global_limit"] == 1
    assert r_data["mobs"][0]["room_limit"] == 1


def test_build_area_json_multiple_spawns_accumulate_count() -> None:
    """Verify that multiple M resets for the same mob in the same room increment count."""
    room = Room(vnum=10, name="Guard Post", description="Guards stand watch.", exits=())
    mob = Mobile(vnum=50, player_name="guard", short_desc="a town guard")
    resets = (
        Reset(command="M", args=(0, 50, 4, 10, 2)),
        Reset(command="M", args=(0, 50, 4, 10, 2)),
    )
    graph = WorldGraph(
        rooms={10: room},
        exits=(),
        mobiles={50: mob},
        resets=resets,
    )
    layout = SolvedLayout(coordinates={10: Coord3D(0, 0, 0)})

    data = build_area_json(graph, layout)
    room_entry = data["rooms"][0]
    assert len(room_entry["mobs"]) == 1
    assert room_entry["mobs"][0]["vnum"] == 50
    assert room_entry["mobs"][0]["count"] == 2


def test_build_area_json_shop_and_special() -> None:
    """Verify that shopkeeper data and specials are properly attached to mobile definitions."""
    room = Room(vnum=1, name="Magic Shop", description="Potions and scrolls.", exits=())
    mob = Mobile(vnum=77, player_name="alchemist", short_desc="the alchemist")
    shop = Shop(keeper=77, buy_types=(10, 11), profit_buy=125, profit_sell=45, open_hour=8, close_hour=20)
    special = Special(command="M", vnum=77, spec_fun="spec_cast_mage")

    graph = WorldGraph(
        rooms={1: room},
        exits=(),
        mobiles={77: mob},
        shops={77: shop},
        specials={77: special},
    )
    layout = SolvedLayout(coordinates={1: Coord3D(0, 0, 0)})

    data = build_area_json(graph, layout)
    mob_data = data["mobiles"]["77"]

    assert mob_data["shop"] is not None
    assert mob_data["shop"]["profit_buy"] == 125
    assert mob_data["shop"]["profit_sell"] == 45
    assert mob_data["shop"]["open_hour"] == 8
    assert mob_data["shop"]["close_hour"] == 20
    assert mob_data["shop"]["buy_types"] == [10, 11]

    assert mob_data["special"] == "spec_cast_mage"


def test_build_area_json_resets_edge_cases() -> None:
    """Verify graceful handling of comments, unknown commands, missing objects, and unmapped mobs."""
    room = Room(vnum=5, name="Crypt", description="Dusty tomb.", exits=())
    mob = Mobile(vnum=88, player_name="wraith", short_desc="a shadowy wraith")
    resets = (
        Reset(command="*", args=()),  # Comment reset
        Reset(command="M", args=(0, 88, 1, 5, 1)),
        Reset(command="*", args=("comment",)),  # Comment between M and E
        Reset(command="E", args=(1, 9999, 1, 16)),  # Unmapped object VNUM 9999
        Reset(command="X", args=(1, 2, 3)),  # Unknown command should clear last_mob_vnum
        Reset(command="G", args=(1, 123, 1)),  # G with no preceding M (should be ignored)
        Reset(command="M", args=(0, 99, 1, 5)),  # M with 4 args (no room_limit)
        Reset(command="M", args=(0,)),  # Invalid M with insufficient args
    )
    graph = WorldGraph(
        rooms={5: room},
        exits=(),
        mobiles={88: mob},
        resets=resets,
    )
    layout = SolvedLayout(coordinates={5: Coord3D(0, 0, 0)})

    data = build_area_json(graph, layout)
    mob88 = data["mobiles"]["88"]

    assert len(mob88["equipment"]) == 1
    assert mob88["equipment"][0]["vnum"] == 9999
    assert mob88["equipment"][0]["short_desc"] == "Object #9999"

    # Mob 99 was spawned but not in graph.mobiles; verify fallback definition created
    assert "99" in data["mobiles"]
    mob99 = data["mobiles"]["99"]
    assert mob99["vnum"] == 99
    assert mob99["name"] == "Mobile #99"


def test_format_dice_and_serialize_flags() -> None:
    """Test helper formatting functions for dice notation and flag serializations."""
    assert _format_dice(None) == ""
    assert _format_dice("10d10+100") == "10d10+100"
    assert _format_dice((1, "d6", 4)) == "1d6+4"
    assert _format_dice((2, 8, 10)) == "2d8+10"
    assert _format_dice((3, "d4", -2)) == "3d4-2"
    assert _format_dice([1, 2, 3, 4]) == "1 2 3 4"

    assert _serialize_flags(None) == ""
    assert _serialize_flags(1024) == 1024
    assert _serialize_flags("SENTINEL") == "SENTINEL"
    assert _serialize_flags(["AGGRESSIVE", "SCAVENGER"]) == ["AGGRESSIVE", "SCAVENGER"]
    assert _serialize_flags(("A", "B")) == ["A", "B"]
    assert _serialize_flags({"custom"}) == ["custom"]
    assert _serialize_flags(3.14) == 3.14


def test_build_area_json_resets_invalid_types_and_type_errors() -> None:
    """Verify branch handling for non-numeric reset arguments and type validation."""
    room = Room(vnum=1, name="Chamber", description="Stone walls.", exits=())
    resets = (
        Reset(command="M", args=(0, "invalid_mob", 1, 1, 1)),
        Reset(command="M", args=(0, 10, 1, 1, 1)),
        Reset(command="E", args=(1, "invalid_obj", 1, 16)),
        Reset(command="E", args=(1, 50, 1, "custom_slot")),
    )
    graph = WorldGraph(rooms={1: room}, exits=(), resets=resets)
    layout = SolvedLayout(coordinates={1: Coord3D(0, 0, 0)})

    data = build_area_json(graph, layout)
    assert "10" in data["mobiles"]
    assert len(data["mobiles"]["10"]["equipment"]) == 1
    assert data["mobiles"]["10"]["equipment"][0]["wear_loc"] == "custom_slot"

    with pytest.raises(TypeError, match="Expected WorldGraph"):
        build_area_json("not_graph", layout)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected SolvedLayout"):
        build_area_json(graph, "not_layout")  # type: ignore[arg-type]


def test_html_renderer_embeds_mob_inspector_elements(tmp_path: Path) -> None:
    """Verify that HTMLRenderer produces an HTML file containing mob overlay and inspector DOM."""
    room = Room(vnum=300, name="Throne", description="Grand chamber.", exits=())
    mob = Mobile(vnum=400, player_name="king", short_desc="the King", level=30)
    crown = Object(vnum=500, name="crown", short_desc="a golden crown", desc="A shiny crown.")
    resets = (
        Reset(command="M", args=(0, 400, 1, 300, 1)),
        Reset(command="E", args=(1, 500, 1, 6)),  # Head slot
    )

    graph = WorldGraph(
        rooms={300: room},
        exits=(),
        mobiles={400: mob},
        objects={500: crown},
        resets=resets,
    )
    layout = SolvedLayout(coordinates={300: Coord3D(0, 0, 0)})

    out_file = tmp_path / "map_viewer.html"
    rendered_path = HTMLRenderer().render(graph, layout, out_file)
    assert rendered_path.is_file()

    content = rendered_path.read_text(encoding="utf-8")
    assert "btn-toggle-mobs" in content
    assert "card-mob-inspector" in content
    assert "room-mobs-section" in content
    assert "mob-badge-group" in content
    assert "the King" in content
    assert "a golden crown" in content


def test_midgaard_fixture_mob_inspector() -> None:
    """Integration test verifying mob inspection and resets on the canonical midgaard.are fixture."""
    fixture_path = Path("tests/fixtures/areas/midgaard.are")
    if not fixture_path.is_file():
        pytest.skip("tests/fixtures/areas/midgaard.are not found")

    area = parse_file(fixture_path)
    graph = WorldGraph.from_area_data(area)
    layout = SolvedLayout(coordinates={r.vnum: Coord3D(0, 0, 0) for r in area.rooms})

    data = build_area_json(graph, layout)

    assert "mobiles" in data
    assert len(data["mobiles"]) > 0

    # Test Hassan (VNUM 3011) at room 3001
    assert "3011" in data["mobiles"]
    hassan = data["mobiles"]["3011"]
    assert hassan["name"] == "Hassan"
    assert hassan["special"] == "spec_executioner"
    assert len(hassan["equipment"]) >= 1
    scimitar = next(e for e in hassan["equipment"] if e["vnum"] == 3005)
    assert scimitar["wear_loc"] == 16  # Wielded

    # Verify room 3001 has mob 3011 in its mobs list
    r3001 = next(r for r in data["rooms"] if r["vnum"] == 3001)
    assert any(m["vnum"] == 3011 for m in r3001["mobs"])

    # Test Weaponsmith (VNUM 3003) shopkeeper
    assert "3003" in data["mobiles"]
    weaponsmith = data["mobiles"]["3003"]
    assert weaponsmith["shop"] is not None
    assert weaponsmith["shop"]["profit_buy"] == 120
    assert len(weaponsmith["equipment"]) >= 5
