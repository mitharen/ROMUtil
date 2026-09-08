from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any


class Direction(enum.IntEnum):
    north = 0
    east = 1
    up = 2
    south = 3
    west = 4
    down = 5
    mod = 6

    def invert(self) -> Direction:
        return Direction((self + self.mod // 2) % self.mod)


direction_matrix = [
    Direction.north,
    Direction.east,
    Direction.south,
    Direction.west,
    Direction.up,
    Direction.down,
]


@dataclass(frozen=True)
class AreaHeader:
    filename: str
    name: str
    builder: str
    vnum_min: int
    vnum_max: int

    def __getitem__(self, index: int) -> Any:
        return (self.filename, self.name, self.builder, (self.vnum_min, self.vnum_max))[index]


@dataclass(frozen=True)
class ExitDef:
    direction: int
    dst_vnum: int
    description: str = ""
    keyword: str = ""
    key_vnum: int = 0
    flags: int = 0

    def __getitem__(self, index: int) -> Any:
        return (self.direction, self.dst_vnum, self.description, self.keyword, self.key_vnum, self.flags)[index]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return (self.direction, self.dst_vnum) == other
        if isinstance(other, ExitDef):
            return (
                self.direction == other.direction
                and self.dst_vnum == other.dst_vnum
                and self.description == other.description
                and self.keyword == other.keyword
                and self.key_vnum == other.key_vnum
                and self.flags == other.flags
            )
        return False

    def __hash__(self) -> int:
        return hash((self.direction, self.dst_vnum, self.description, self.keyword, self.key_vnum, self.flags))


@dataclass(frozen=True)
class ExtraDescr:
    keyword: str
    description: str

    def __getitem__(self, index: int) -> Any:
        return (self.keyword, self.description)[index]


@dataclass(frozen=True)
class RoomDef:
    vnum: int
    name: str
    description: str
    room_flags: Any = 0
    sector: int = 0
    exits: tuple[ExitDef, ...] = ()
    extras: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.exits, tuple):
            object.__setattr__(self, "exits", tuple(self.exits) if self.exits is not None else ())
        if not isinstance(self.extras, tuple):
            object.__setattr__(self, "extras", tuple(self.extras) if self.extras is not None else ())

    def __getitem__(self, index: int) -> Any:
        return (self.vnum, self.name, self.description, self.exits, self.room_flags, self.sector, self.extras)[index]


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

    def __getitem__(self, index: int) -> Any:
        return (self.vnum, self.player_name, self.short_desc, self.long_desc, self.desc, self.race)[index]


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

    def __getitem__(self, index: int) -> Any:
        return (self.vnum, self.name, self.short_desc, self.desc, self.item_type)[index]


@dataclass(frozen=True)
class ResetDef:
    command: str
    args: tuple[Any, ...] = ()
    comment: str | None = None

    def __getitem__(self, index: int) -> Any:
        return (self.command, *self.args)[index]


@dataclass(frozen=True)
class ShopDef:
    keeper: int
    buy_types: tuple[int, ...] = ()
    profit_buy: int = 100
    profit_sell: int = 100
    open_hour: int = 0
    close_hour: int = 24
    comment: str | None = None

    def __getitem__(self, index: int) -> Any:
        return (self.keeper, self.buy_types, self.profit_buy, self.profit_sell, self.open_hour, self.close_hour)[index]


@dataclass(frozen=True)
class SpecialDef:
    command: str
    vnum: int
    spec_fun: str
    comment: str | None = None

    def __getitem__(self, index: int) -> Any:
        return (self.command, self.vnum, self.spec_fun)[index]


@dataclass(frozen=True)
class HelpDef:
    level: int
    keywords: tuple[str, ...] = ()
    text: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.keywords, tuple):
            object.__setattr__(self, "keywords", tuple(self.keywords) if self.keywords is not None else ())

    def __getitem__(self, index: int) -> Any:
        return (self.keywords, self.text, self.level)[index]


@dataclass(frozen=True)
class SocialDef:
    name: str
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            object.__setattr__(self, "stages", tuple(self.stages) if self.stages is not None else ())

    def __getitem__(self, index: int) -> Any:
        return (self.name, self.stages)[index]


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
            }
            if key in mapping:
                return mapping[key]
        raise KeyError(key)


class Exit:
    def __init__(self, e: ExitDef | tuple[Any, ...] | list[Any], source: int, distance: int = 1):
        self.src = source
        if isinstance(e, ExitDef):
            self.dst = e.dst_vnum
            self.direction = Direction(direction_matrix[e.direction])
        elif isinstance(e, (tuple, list)):
            self.dst = e[1]
            self.direction = Direction(direction_matrix[e[0]])
        elif isinstance(e, Exit):
            self.dst = e.dst
            self.direction = e.direction
        else:
            self.dst = getattr(e, "dst_vnum", getattr(e, "dst", -1))
            dir_val = getattr(e, "direction", 0)
            self.direction = Direction(direction_matrix[dir_val]) if isinstance(dir_val, int) else dir_val
        self.distance = distance
        self.one_way = False

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
    def __init__(self, r: RoomDef | tuple[Any, ...] | list[Any] | Room):
        if isinstance(r, RoomDef):
            self.vnum = r.vnum
            self.name = r.name
            self.desc = r.description
            self.exits = [Exit(e, self.vnum) for e in r.exits if e is not None]
        elif isinstance(r, (tuple, list)):
            self.vnum = r[0]
            self.name = r[1]
            self.desc = r[2]
            self.exits = [] if len(r) <= 3 or not r[3] else [Exit(e, self.vnum) for e in r[3] if e is not None]
        elif isinstance(r, Room):
            self.vnum = r.vnum
            self.name = r.name
            self.desc = r.desc
            self.exits = list(r.exits)
        else:
            self.vnum = getattr(r, "vnum", 0)
            self.name = getattr(r, "name", "")
            self.desc = getattr(r, "desc", getattr(r, "description", ""))
            exits_attr = getattr(r, "exits", [])
            self.exits = [Exit(e, self.vnum) for e in exits_attr if e is not None]

        self.fixups: list[Any] = []
        self.dummy = False
        self.x: int | None = None
        self.y: int | None = None
        self.z: int | None = None

    def replace_exit(self, orig: int, replacement: int, distance: int) -> None:
        for e in self.exits:
            if e.dst == orig:
                e.dst = replacement
                e.distance += distance

    def __repr__(self) -> str:
        return f"[{self.vnum}: {self.name}] {{{self.exits}}}"
