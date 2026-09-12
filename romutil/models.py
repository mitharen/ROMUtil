from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any, Sequence


class Direction(enum.IntEnum):
    north = 0
    east = 1
    up = 2
    south = 3
    west = 4
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


direction_matrix = [
    Direction.north,
    Direction.east,
    Direction.south,
    Direction.west,
    Direction.up,
    Direction.down,
    Direction.northeast,
    Direction.northwest,
    Direction.southeast,
    Direction.southwest,
]


@dataclass(frozen=True)
class AreaHeader:
    filename: str
    name: str
    builder: str
    vnum_min: int
    vnum_max: int


@dataclass(frozen=True)
class ExitDef:
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
class RoomDef:
    vnum: int
    name: str
    description: str
    room_flags: Any = 0
    sector: int = 0
    exits: tuple[ExitDef, ...] = ()
    extras: tuple[Any, ...] = ()
    area_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exits, tuple):
            object.__setattr__(self, "exits", tuple(self.exits) if self.exits is not None else ())
        if not isinstance(self.extras, tuple):
            object.__setattr__(self, "extras", tuple(self.extras) if self.extras is not None else ())


@dataclass(frozen=True)
class MobileDef:
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
class ObjectDef:
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
class ResetDef:
    command: str
    args: tuple[Any, ...] = ()
    comment: str | None = None


@dataclass(frozen=True)
class ShopDef:
    keeper: int
    buy_types: tuple[int, ...] = ()
    profit_buy: int = 100
    profit_sell: int = 100
    open_hour: int = 0
    close_hour: int = 24
    comment: str | None = None


@dataclass(frozen=True)
class SpecialDef:
    command: str
    vnum: int
    spec_fun: str
    comment: str | None = None


@dataclass(frozen=True)
class HelpDef:
    level: int
    keywords: tuple[str, ...] = ()
    text: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.keywords, tuple):
            object.__setattr__(self, "keywords", tuple(self.keywords) if self.keywords is not None else ())


@dataclass(frozen=True)
class SocialDef:
    name: str
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            object.__setattr__(self, "stages", tuple(self.stages) if self.stages is not None else ())


@dataclass(frozen=True)
class AreaData:
    header: AreaHeader | None = None
    rooms: tuple[RoomDef, ...] = ()
    mobiles: tuple[MobileDef, ...] = ()
    objects: tuple[ObjectDef, ...] = ()
    resets: tuple[ResetDef, ...] = ()
    shops: tuple[ShopDef, ...] = ()
    specials: tuple[SpecialDef, ...] = ()
    helps: tuple[HelpDef, ...] = ()
    socials: tuple[SocialDef, ...] = ()
    reset_message: str | None = None
    flag: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("rooms", "mobiles", "objects", "resets", "shops", "specials", "helps", "socials"):
            val = getattr(self, field_name)
            if not isinstance(val, tuple):
                object.__setattr__(self, field_name, tuple(val) if val is not None else ())

    def __iter__(self) -> Any:
        yield ("#AREA", self.header)
        yield ("#ROOMS", self.rooms)
        yield ("#MOBILES", self.mobiles)
        yield ("#OBJECTS", self.objects)
        yield ("#RESETS", self.resets)
        yield ("#SHOPS", self.shops)
        yield ("#SPECIALS", self.specials)
        yield ("#HELPS", self.helps)
        yield ("#SOCIALS", self.socials)
        if self.reset_message is not None:
            yield ("#RESETMESSAGE", self.reset_message)
        if self.flag is not None:
            yield ("#FLAG", self.flag)

    def __len__(self) -> int:
        return len(self.rooms) if self.rooms else 9

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self)[key]
        if isinstance(key, str):
            mapping = {
                "#AREA": self.header,
                "#ROOMS": self.rooms,
                "#MOBILES": self.mobiles,
                "#OBJECTS": self.objects,
                "#RESETS": self.resets,
                "#SHOPS": self.shops,
                "#SPECIALS": self.specials,
                "#HELPS": self.helps,
                "#SOCIALS": self.socials,
                "#RESETMESSAGE": self.reset_message,
                "#FLAG": self.flag,
            }
            if key in mapping:
                return mapping[key]
        raise KeyError(key)


class Exit:
    def __init__(
        self,
        e: ExitDef | Exit | None = None,
        source: int | None = None,
        distance: int = 1,
        *,
        src: int | None = None,
        dst: int | None = None,
        direction: Direction | int | None = None,
        target_vnum: int | None = None,
    ) -> None:
        effective_source = source if source is not None else src
        if effective_source is None:
            raise ValueError("Exit requires a source vnum")
        self.src = effective_source
        self.distance = distance
        self.one_way = False
        self.cut = False
        self.target_vnum: int | None = target_vnum

        if e is not None:
            if isinstance(e, ExitDef):
                self.dst = e.dst_vnum
                self.direction = Direction(direction_matrix[e.direction])
                if self.target_vnum is None:
                    self.target_vnum = e.dst_vnum
            elif isinstance(e, Exit):
                self.dst = e.dst
                self.direction = e.direction
                self.one_way = e.one_way
                self.cut = getattr(e, 'cut', False)
                if self.target_vnum is None:
                    self.target_vnum = getattr(e, 'target_vnum', None)
                if distance == 1 and e.distance != 1:
                    self.distance = e.distance
            else:
                raise TypeError(f"Expected ExitDef or Exit, got {type(e).__name__}")
        elif dst is not None and direction is not None:
            self.dst = dst
            if self.target_vnum is None:
                self.target_vnum = dst
            if isinstance(direction, Direction):
                self.direction = direction
            else:
                self.direction = Direction(direction_matrix[direction])
        else:
            raise TypeError("Exit requires an ExitDef, Exit, or explicit keyword arguments (dst, direction, source)")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Exit):
            return False
        if self.src == other.src and self.dst == other.dst and self.direction == other.direction:
            return True
        if self.src == other.dst and self.dst == other.src and self.direction == other.direction.invert():
            return True
        return False

    def __contains__(self, room: Any) -> bool:
        return room in (self.src, self.dst)

    def __repr__(self) -> str:
        return f"{self.src} -> {self.dst} ({self.distance} {self.direction.name})"

    def __hash__(self) -> int:
        r0, r1, d = (self.src, self.dst, self.direction) if self.src < self.dst \
            else (self.dst, self.src, self.direction.invert())
        return hash(f"{r0} {r1} {d}")


class Room:
    def __init__(
        self,
        r: RoomDef | Room | None = None,
        *,
        vnum: int | None = None,
        name: str = "",
        desc: str = "",
        description: str | None = None,
        exits: list[Exit] | tuple[ExitDef, ...] | list[ExitDef] | None = None,
        target_vnum: int | None = None,
        area_name: str | None = None,
    ) -> None:
        self.fixups: list[Any] = []
        self.dummy = False
        self.target_vnum: int | None = target_vnum
        self.x: int | None = None
        self.y: int | None = None
        self.z: int | None = None
        self.area_name: str | None = area_name

        if r is not None:
            if isinstance(r, RoomDef):
                self.vnum = r.vnum
                self.name = r.name
                self.desc = r.description
                self.exits = [Exit(e, source=self.vnum) for e in r.exits if e is not None]
                if self.area_name is None:
                    self.area_name = getattr(r, 'area_name', None)
            elif isinstance(r, Room):
                self.vnum = r.vnum
                self.name = r.name
                self.desc = r.desc
                self.exits = list(r.exits)
                self.dummy = r.dummy
                if self.target_vnum is None:
                    self.target_vnum = getattr(r, 'target_vnum', None)
                if self.area_name is None:
                    self.area_name = getattr(r, 'area_name', None)
                self.fixups = list(r.fixups)
                self.x = r.x
                self.y = r.y
                self.z = r.z
            else:
                raise TypeError(f"Expected RoomDef or Room, got {type(r).__name__}")
        elif vnum is not None:
            self.vnum = vnum
            self.name = name
            self.desc = description if description is not None else desc
            self.exits = []
            if exits is not None:
                for e in exits:
                    if isinstance(e, Exit):
                        self.exits.append(e)
                    elif isinstance(e, ExitDef):
                        self.exits.append(Exit(e, source=self.vnum))
                    else:
                        raise TypeError(f"Expected Exit or ExitDef in exits, got {type(e).__name__}")
        else:
            raise TypeError("Room requires a RoomDef, Room, or keyword arguments (vnum=...)")

    def replace_exit(self, orig: int, replacement: int, distance: int) -> None:
        for e in self.exits:
            if e.dst == orig:
                e.dst = replacement
                e.distance += distance

    def __repr__(self) -> str:
        return f"[{self.vnum}: {self.name}] {{{self.exits}}}"


def merge_areas(areas: Sequence[AreaData], title: str | None = None) -> AreaData:
    """Merge multiple AreaData containers into a unified composite AreaData.

    Combines rooms, mobiles, objects, resets, shops, specials, helps, and socials,
    while tagging each RoomDef with its originating area provenance (area_name).
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

    merged_rooms: list[RoomDef] = []
    seen_room_vnums: set[int] = set()
    for area in areas:
        source_area_name = area.header.name if area.header and area.header.name else None
        for room in area.rooms:
            if room.vnum in seen_room_vnums:
                continue
            seen_room_vnums.add(room.vnum)
            r_area = room.area_name or source_area_name
            if room.area_name != r_area:
                room = RoomDef(
                    vnum=room.vnum,
                    name=room.name,
                    description=room.description,
                    room_flags=room.room_flags,
                    sector=room.sector,
                    exits=room.exits,
                    extras=room.extras,
                    area_name=r_area,
                )
            merged_rooms.append(room)

    merged_mobiles: list[MobileDef] = []
    seen_mob_vnums: set[int] = set()
    for area in areas:
        for mob in area.mobiles:
            if mob.vnum not in seen_mob_vnums:
                seen_mob_vnums.add(mob.vnum)
                merged_mobiles.append(mob)

    merged_objects: list[ObjectDef] = []
    seen_obj_vnums: set[int] = set()
    for area in areas:
        for obj in area.objects:
            if obj.vnum not in seen_obj_vnums:
                seen_obj_vnums.add(obj.vnum)
                merged_objects.append(obj)

    merged_resets: list[ResetDef] = []
    for area in areas:
        merged_resets.extend(area.resets)

    merged_shops: list[ShopDef] = []
    seen_shop_keepers: set[int] = set()
    for area in areas:
        for shop in area.shops:
            if shop.keeper not in seen_shop_keepers:
                seen_shop_keepers.add(shop.keeper)
                merged_shops.append(shop)

    merged_specials: list[SpecialDef] = []
    for area in areas:
        merged_specials.extend(area.specials)

    merged_helps: list[HelpDef] = []
    for area in areas:
        merged_helps.extend(area.helps)

    merged_socials: list[SocialDef] = []
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
