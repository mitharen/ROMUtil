from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any, Sequence


class Direction(enum.IntEnum):
    north = 0
    east = 1
    south = 2
    west = 3
    up = 4
    down = 5
    northeast = 6
    northwest = 7
    southeast = 8
    southwest = 9

    def invert(self) -> Direction:
        inverses = {
            Direction.north: Direction.south,
            Direction.south: Direction.north,
            Direction.east: Direction.west,
            Direction.west: Direction.east,
            Direction.up: Direction.down,
            Direction.down: Direction.up,
            Direction.northeast: Direction.southwest,
            Direction.southwest: Direction.northeast,
            Direction.northwest: Direction.southeast,
            Direction.southeast: Direction.northwest,
        }
        return inverses[self]




@dataclass(frozen=True)
class AreaHeader:
    filename: str
    name: str
    builder: str
    vnum_min: int
    vnum_max: int


@dataclass(frozen=True)
class Exit:
    direction: int
    dst_vnum: int
    description: str = ""
    keyword: str = ""
    key_vnum: int = 0
    flags: int = 0


@dataclass(frozen=True)
class ExtraDescr:
    keyword: str
    description: str


@dataclass(frozen=True)
class Room:
    vnum: int
    name: str
    description: str
    room_flags: Any = 0
    sector: int = 0
    exits: tuple[Exit, ...] = ()
    extras: tuple[Any, ...] = ()
    area_name: str | None = None
    area_file: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exits, tuple):
            object.__setattr__(self, "exits", tuple(self.exits) if self.exits is not None else ())
        if not isinstance(self.extras, tuple):
            object.__setattr__(self, "extras", tuple(self.extras) if self.extras is not None else ())


@dataclass(frozen=True)
class Mobile:
    vnum: int
    player_name: str
    short_desc: str
    long_desc: str = ""
    desc: str = ""
    race: str = ""
    act_flags: Any = None
    affected_by: Any = None
    alignment: int = 0
    group: int = 0
    level: int = 0
    hitroll: int = 0
    hit: Any = None
    mana: Any = None
    damage: Any = None
    dam_type: str = ""
    ac_pierce: int = 0
    ac_bash: int = 0
    ac_slash: int = 0
    ac_exotic: int = 0
    off_flags: Any = None
    imm_flags: Any = None
    res_flags: Any = None
    vuln_flags: Any = None
    start_pos: str = ""
    default_pos: str = ""
    sex: str = ""
    wealth: int = 0
    form: Any = None
    parts: Any = None
    size: str = ""
    material: Any = None
    optionals: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.optionals, tuple):
            object.__setattr__(self, "optionals", tuple(self.optionals) if self.optionals is not None else ())


@dataclass(frozen=True)
class Object:
    vnum: int
    name: str
    short_desc: str
    desc: str
    material: str = ""
    item_type: str = ""
    extra_flags: Any = None
    wear_flags: Any = None
    values: tuple[Any, ...] = ()
    level: int = 0
    weight: int = 0
    cost: int = 0
    condition: Any = None
    optionals: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values) if self.values is not None else ())
        if not isinstance(self.optionals, tuple):
            object.__setattr__(self, "optionals", tuple(self.optionals) if self.optionals is not None else ())


@dataclass(frozen=True)
class Reset:
    command: str
    args: tuple[Any, ...] = ()
    comment: str | None = None


@dataclass(frozen=True)
class Shop:
    keeper: int
    buy_types: tuple[int, ...] = ()
    profit_buy: int = 100
    profit_sell: int = 100
    open_hour: int = 0
    close_hour: int = 24
    comment: str | None = None


@dataclass(frozen=True)
class Special:
    command: str
    vnum: int
    spec_fun: str
    comment: str | None = None


@dataclass(frozen=True)
class Help:
    level: int
    keywords: tuple[str, ...] = ()
    text: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.keywords, tuple):
            object.__setattr__(self, "keywords", tuple(self.keywords) if self.keywords is not None else ())


@dataclass(frozen=True)
class Social:
    name: str
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            object.__setattr__(self, "stages", tuple(self.stages) if self.stages is not None else ())


@dataclass(frozen=True)
class AreaData:
    header: AreaHeader | None = None
    rooms: tuple[Room, ...] = ()
    mobiles: tuple[Mobile, ...] = ()
    objects: tuple[Object, ...] = ()
    resets: tuple[Reset, ...] = ()
    shops: tuple[Shop, ...] = ()
    specials: tuple[Special, ...] = ()
    helps: tuple[Help, ...] = ()
    socials: tuple[Social, ...] = ()
    reset_message: str | None = None
    flag: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("rooms", "mobiles", "objects", "resets", "shops", "specials", "helps", "socials"):
            val = getattr(self, field_name)
            if not isinstance(val, tuple):
                object.__setattr__(self, field_name, tuple(val) if val is not None else ())





def merge_areas(areas: Sequence[AreaData], title: str | None = None) -> AreaData:
    """Merge multiple AreaData containers into a unified composite AreaData.

    Combines rooms, mobiles, objects, resets, shops, specials, helps, and socials,
    while tagging each Room with its originating area provenance (area_name).
    """
    if not areas:
        header = AreaHeader(
            filename="composite.are",
            name=title or "Composite Area",
            builder="",
            vnum_min=0,
            vnum_max=0,
        )
        return AreaData(header=header)

    area_names = [a.header.name for a in areas if a.header and a.header.name]
    filenames = [a.header.filename for a in areas if a.header and a.header.filename]
    builders = sorted(set(a.header.builder for a in areas if a.header and a.header.builder))

    merged_title = title or (" + ".join(area_names) if area_names else "Composite Area")
    merged_filename = filenames[0] if filenames else "composite.are"
    merged_builder = ", ".join(builders)

    merged_rooms: list[Room] = []
    seen_room_vnums: set[int] = set()
    for area in areas:
        source_area_name = area.header.name if area.header and area.header.name else None
        source_area_file = area.header.filename if area.header and area.header.filename else None
        for room in area.rooms:
            if room.vnum in seen_room_vnums:
                continue
            seen_room_vnums.add(room.vnum)
            r_area = room.area_name or source_area_name
            r_file = getattr(room, 'area_file', None) or source_area_file
            if room.area_name != r_area or getattr(room, 'area_file', None) != r_file:
                room = Room(
                    vnum=room.vnum,
                    name=room.name,
                    description=room.description,
                    room_flags=room.room_flags,
                    sector=room.sector,
                    exits=room.exits,
                    extras=room.extras,
                    area_name=r_area,
                    area_file=r_file,
                )
            merged_rooms.append(room)

    merged_mobiles: list[Mobile] = []
    seen_mob_vnums: set[int] = set()
    for area in areas:
        for mob in area.mobiles:
            if mob.vnum not in seen_mob_vnums:
                seen_mob_vnums.add(mob.vnum)
                merged_mobiles.append(mob)

    merged_objects: list[Object] = []
    seen_obj_vnums: set[int] = set()
    for area in areas:
        for obj in area.objects:
            if obj.vnum not in seen_obj_vnums:
                seen_obj_vnums.add(obj.vnum)
                merged_objects.append(obj)

    merged_resets: list[Reset] = []
    for area in areas:
        merged_resets.extend(area.resets)

    merged_shops: list[Shop] = []
    seen_shop_keepers: set[int] = set()
    for area in areas:
        for shop in area.shops:
            if shop.keeper not in seen_shop_keepers:
                seen_shop_keepers.add(shop.keeper)
                merged_shops.append(shop)

    merged_specials: list[Special] = []
    for area in areas:
        merged_specials.extend(area.specials)

    merged_helps: list[Help] = []
    for area in areas:
        merged_helps.extend(area.helps)

    merged_socials: list[Social] = []
    seen_social_names: set[str] = set()
    for area in areas:
        for soc in area.socials:
            if soc.name not in seen_social_names:
                seen_social_names.add(soc.name)
                merged_socials.append(soc)

    reset_message = next((a.reset_message for a in areas if a.reset_message), None)
    flag = next((a.flag for a in areas if a.flag), None)

    headers_with_bounds = [a.header for a in areas if a.header and a.header.vnum_max > 0]
    if headers_with_bounds:
        v_min = min(h.vnum_min for h in headers_with_bounds)
        v_max = max(h.vnum_max for h in headers_with_bounds)
    elif merged_rooms:
        v_min = min(r.vnum for r in merged_rooms)
        v_max = max(r.vnum for r in merged_rooms)
    else:
        v_min = 0
        v_max = 0

    header = AreaHeader(
        filename=merged_filename,
        name=merged_title,
        builder=merged_builder,
        vnum_min=v_min,
        vnum_max=v_max,
    )

    return AreaData(
        header=header,
        rooms=tuple(merged_rooms),
        mobiles=tuple(merged_mobiles),
        objects=tuple(merged_objects),
        resets=tuple(merged_resets),
        shops=tuple(merged_shops),
        specials=tuple(merged_specials),
        helps=tuple(merged_helps),
        socials=tuple(merged_socials),
        reset_message=reset_message,
        flag=flag,
    )


__all__ = [
    "AreaData",
    "AreaHeader",
    "Direction",
    "Exit",
    "ExtraDescr",
    "Help",
    "Mobile",
    "Object",
    "Reset",
    "Room",
    "Shop",
    "Social",
    "Special",
    "merge_areas",
]
