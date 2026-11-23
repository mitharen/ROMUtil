from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any, Mapping, Sequence


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


class AreaDialect(str, enum.Enum):
    """Supported Diku/Merc/ROM area file dialects."""

    ROM = "rom"
    MERC = "merc"
    ENVY = "envy"
    DIKU = "diku"
    CIRCLE = "circle"
    SMAUG = "smaug"
    ANATOLIA = "anatolia"
    ACK = "ack"


class TerrainType(str, enum.Enum):
    """Unified canonical terrain classification normalized across MUD dialects."""

    INSIDE = "inside"
    CITY = "city"
    FIELD = "field"
    FOREST = "forest"
    HILLS = "hills"
    MOUNTAIN = "mountain"
    WATER_SWIM = "water_swim"
    WATER_NOSWIM = "water_noswim"
    UNDERWATER = "underwater"
    AIR = "air"
    DESERT = "desert"
    OCEANFLOOR = "oceanfloor"
    UNDERGROUND = "underground"
    LAVA = "lava"
    SWAMP = "swamp"
    ICE = "ice"
    UNUSED = "unused"
    UNKNOWN = "unknown"


TERRAIN_LABELS: dict[TerrainType, str] = {
    TerrainType.INSIDE: "Inside",
    TerrainType.CITY: "City",
    TerrainType.FIELD: "Field",
    TerrainType.FOREST: "Forest",
    TerrainType.HILLS: "Hills",
    TerrainType.MOUNTAIN: "Mountain",
    TerrainType.WATER_SWIM: "Water (Swim)",
    TerrainType.WATER_NOSWIM: "Water (No Swim)",
    TerrainType.UNDERWATER: "Underwater",
    TerrainType.AIR: "Air",
    TerrainType.DESERT: "Desert",
    TerrainType.OCEANFLOOR: "Oceanfloor",
    TerrainType.UNDERGROUND: "Underground",
    TerrainType.LAVA: "Lava",
    TerrainType.SWAMP: "Swamp",
    TerrainType.ICE: "Ice",
    TerrainType.UNUSED: "Unused",
    TerrainType.UNKNOWN: "Unknown",
}

_BASE_SECTORS: dict[int, TerrainType] = {
    0: TerrainType.INSIDE,
    1: TerrainType.CITY,
    2: TerrainType.FIELD,
    3: TerrainType.FOREST,
    4: TerrainType.HILLS,
    5: TerrainType.MOUNTAIN,
    6: TerrainType.WATER_SWIM,
    7: TerrainType.WATER_NOSWIM,
}

_ROM_SECTORS: dict[int, TerrainType] = {
    **_BASE_SECTORS,
    8: TerrainType.UNUSED,
    9: TerrainType.AIR,
    10: TerrainType.DESERT,
}

DIALECT_SECTOR_TABLES: dict[AreaDialect, dict[int, TerrainType]] = {
    AreaDialect.ROM: _ROM_SECTORS,
    AreaDialect.MERC: _ROM_SECTORS,
    AreaDialect.ANATOLIA: _ROM_SECTORS,
    AreaDialect.ACK: _ROM_SECTORS,
    AreaDialect.DIKU: dict(_BASE_SECTORS),
    AreaDialect.CIRCLE: {
        **_BASE_SECTORS,
        8: TerrainType.UNDERWATER,
        9: TerrainType.AIR,
    },
    AreaDialect.ENVY: {
        **_BASE_SECTORS,
        8: TerrainType.UNUSED,
        9: TerrainType.UNDERWATER,
        10: TerrainType.AIR,
    },
    AreaDialect.SMAUG: {
        **_BASE_SECTORS,
        8: TerrainType.UNDERWATER,
        9: TerrainType.AIR,
        10: TerrainType.DESERT,
        11: TerrainType.UNKNOWN,
        12: TerrainType.OCEANFLOOR,
        13: TerrainType.UNDERGROUND,
        14: TerrainType.LAVA,
        15: TerrainType.SWAMP,
        16: TerrainType.ICE,
    },
}


def resolve_dialect(dialect: AreaDialect | str) -> AreaDialect:
    """Resolve an AreaDialect enum or string identifier into an AreaDialect."""
    if isinstance(dialect, AreaDialect):
        return dialect
    if isinstance(dialect, str):
        key = dialect.strip().lower()
        try:
            return AreaDialect(key)
        except ValueError as exc:
            supported = ", ".join(d.value for d in AreaDialect)
            raise ValueError(f"Unsupported area dialect {dialect!r}. Supported: {supported}") from exc
    raise TypeError(f"Expected AreaDialect or str, got {type(dialect).__name__}")


def normalize_sector(sector: int, dialect: AreaDialect | str = AreaDialect.ROM) -> TerrainType:
    """Map a dialect-specific numeric sector ID into a unified canonical TerrainType."""
    resolved_dialect = resolve_dialect(dialect)
    table = DIALECT_SECTOR_TABLES.get(resolved_dialect, _ROM_SECTORS)
    return table.get(int(sector), TerrainType.UNKNOWN)


def get_terrain_label(terrain: TerrainType | str, sector: int | None = None) -> str:
    """Return the human-readable terrain label for a canonical TerrainType."""
    if isinstance(terrain, TerrainType):
        t_enum = terrain
    else:
        try:
            t_enum = TerrainType(str(terrain).strip().lower())
        except ValueError:
            t_enum = TerrainType.UNKNOWN
    if t_enum == TerrainType.UNKNOWN and sector is not None:
        return f"Unknown ({int(sector)})"
    return TERRAIN_LABELS.get(t_enum, "Unknown")


_INDOOR_TERRAINS: tuple[TerrainType, ...] = (
    TerrainType.INSIDE,
    TerrainType.UNDERGROUND,
    TerrainType.UNUSED,
    TerrainType.UNKNOWN,
)
_CITY_TERRAINS: tuple[TerrainType, ...] = (
    TerrainType.CITY,
    TerrainType.WATER_SWIM,
)
_WILDERNESS_TERRAINS: tuple[TerrainType, ...] = (
    TerrainType.FIELD,
    TerrainType.FOREST,
    TerrainType.HILLS,
    TerrainType.MOUNTAIN,
    TerrainType.WATER_NOSWIM,
    TerrainType.UNDERWATER,
    TerrainType.OCEANFLOOR,
    TerrainType.AIR,
    TerrainType.DESERT,
    TerrainType.LAVA,
    TerrainType.SWAMP,
    TerrainType.ICE,
)


def _validate_terrain_scale_value(value: Any, label: str) -> int:
    """Validate that a terrain scale factor is a positive integer (>= 1), rejecting booleans."""
    if isinstance(value, bool) or isinstance(value, (float, complex)):
        raise ValueError(f"{label} must be a positive integer >= 1, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer >= 1, got {value!r}.") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be a positive integer >= 1, got {parsed}.")
    return parsed


def build_terrain_scales(
    indoor: int = 1,
    city: int = 1,
    wilderness: int = 1,
) -> dict[TerrainType, int]:
    """Build a complete TerrainType-to-scale mapping from indoor, city, and wilderness scales."""
    indoor_scale = _validate_terrain_scale_value(indoor, "indoor terrain scale")
    city_scale = _validate_terrain_scale_value(city, "city terrain scale")
    wilderness_scale = _validate_terrain_scale_value(wilderness, "wilderness terrain scale")

    scales: dict[TerrainType, int] = {}
    for terrain in _INDOOR_TERRAINS:
        scales[terrain] = indoor_scale
    for terrain in _CITY_TERRAINS:
        scales[terrain] = city_scale
    for terrain in _WILDERNESS_TERRAINS:
        scales[terrain] = wilderness_scale
    return scales


def parse_terrain_scales(
    spec: Mapping[TerrainType, int] | str | Sequence[int] | None,
) -> dict[TerrainType, int] | None:
    """Parse and validate a terrain scale specification into a TerrainType scale mapping."""
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        resolved: dict[TerrainType, int] = {}
        for key, val in spec.items():
            if isinstance(key, TerrainType):
                t_enum = key
            elif isinstance(key, str):
                try:
                    t_enum = TerrainType(key.strip().lower())
                except ValueError as exc:
                    raise ValueError(f"Unknown terrain type: {key!r}") from exc
            else:
                raise ValueError(f"Invalid terrain type key: {key!r}")
            resolved[t_enum] = _validate_terrain_scale_value(val, f"terrain scale for {t_enum.value}")
        return resolved
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 3 or any(not part or not (part.isascii() and part.isdigit()) for part in parts):
            raise ValueError(
                f"Invalid terrain scales {spec!r}; expected '<indoor>:<city>:<wilderness>' of positive integers >= 1."
            )
        return build_terrain_scales(int(parts[0]), int(parts[1]), int(parts[2]))
    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        if len(spec) != 3:
            raise ValueError(
                f"Expected 3 terrain scale integers (indoor, city, wilderness), got {len(spec)}."
            )
        if any(isinstance(v, bool) or not isinstance(v, int) for v in spec):
            raise ValueError(
                f"Expected 3 non-boolean integer terrain scales, got {spec!r}."
            )
        return build_terrain_scales(int(spec[0]), int(spec[1]), int(spec[2]))
    raise ValueError(f"Unsupported terrain scales specification: {spec!r}")


UNIFORM_TERRAIN_SCALES: dict[TerrainType, int] = {t: 1 for t in TerrainType}
TERRAIN_SCALES_1_1_2: dict[TerrainType, int] = build_terrain_scales(1, 1, 2)
DEFAULT_TERRAIN_SCALES: dict[TerrainType, int] = build_terrain_scales(1, 2, 4)

ROOM_INDOORS_FLAG = 8


def has_indoors_flag(room_flags: Any) -> bool:
    """Return True if room_flags includes ROOM_INDOORS (bit 8 or ROM ASCII flag 'D')."""
    if isinstance(room_flags, int):
        return bool(room_flags & ROOM_INDOORS_FLAG)
    if isinstance(room_flags, str):
        s = room_flags.strip()
        if not s:
            return False
        for part in s.split("|"):
            part = part.strip()
            if part.lstrip("-").isdigit():
                if int(part) & ROOM_INDOORS_FLAG:
                    return True
            elif "D" in part:
                return True
    return False


def normalize_room_terrain(
    sector: int,
    room_flags: Any = 0,
    dialect: AreaDialect | str = AreaDialect.ROM,
) -> TerrainType:
    """Normalize a room's terrain; the room-level INDOORS flag forces ``INSIDE``.

    The raw ``sector`` is preserved on the room and then only conveys relative
    movement difficulty.
    """
    if has_indoors_flag(room_flags):
        return TerrainType.INSIDE
    return normalize_sector(sector, dialect)


def get_terrain_scale(
    terrain: TerrainType | str | None,
    scales: Mapping[TerrainType, int] | None = None,
) -> int:
    """Return the spatial footprint scale (>= 1) for a canonical terrain type."""
    scale_map = UNIFORM_TERRAIN_SCALES if scales is None else scales
    if terrain is None:
        t_enum = TerrainType.INSIDE
    elif isinstance(terrain, TerrainType):
        t_enum = terrain
    else:
        try:
            t_enum = TerrainType(str(terrain).strip().lower())
        except ValueError:
            t_enum = TerrainType.UNKNOWN
    raw = scale_map.get(t_enum, 1)
    return max(1, int(raw))


def scale_to_extent(scale: int, coord_unit: int) -> tuple[int, int]:
    """Convert a room's integer scale factor and grid unit into (lo, hi) footprint half-extents."""
    if coord_unit == 1:
        half = (scale - 1) // 2
        return (half, half)
    return (scale, scale - 1)


def compute_room_footprints(
    rooms: Mapping[int, Room],
    terrain_scales: Mapping[TerrainType, int] | None = None,
    *,
    coord_unit: int | None = None,
) -> tuple[int, dict[int, int], dict[int, tuple[int, int]]]:
    """Compute grid unit and per-room discrete footprint extents across a room mapping.

    When all present room scales are odd integers (including uniform 1s), ``coord_unit = 1``
    and each room spans ``[x - lo, x + hi]`` with ``lo = hi = (scale - 1) // 2``. When any
    present room scale is even (e.g. ``1 / 2 / 4``), ``coord_unit = 2`` half-cell coordinates
    are used with ``lo = scale`` and ``hi = scale - 1`` so arbitrary scale pairs tile cleanly on
    integer coordinates.
    """
    room_scales: dict[int, int] = {
        vnum: get_terrain_scale(getattr(room, "terrain", None), terrain_scales)
        for vnum, room in rooms.items()
    }
    if coord_unit is None:
        resolved_unit = 2 if any(s % 2 == 0 for s in room_scales.values()) else 1
    else:
        resolved_unit = int(coord_unit)
    room_extents: dict[int, tuple[int, int]] = {
        vnum: scale_to_extent(scale, resolved_unit)
        for vnum, scale in room_scales.items()
    }
    return resolved_unit, room_scales, room_extents


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
    terrain: TerrainType | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exits, tuple):
            object.__setattr__(self, "exits", tuple(self.exits) if self.exits is not None else ())
        if not isinstance(self.extras, tuple):
            object.__setattr__(self, "extras", tuple(self.extras) if self.extras is not None else ())
        if self.terrain is None:
            object.__setattr__(self, "terrain", normalize_room_terrain(self.sector, self.room_flags))
        elif not isinstance(self.terrain, TerrainType):
            object.__setattr__(self, "terrain", TerrainType(str(self.terrain).lower()))


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
    dialect: AreaDialect = AreaDialect.ROM

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
                    terrain=room.terrain,
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

    dialects = {getattr(a, "dialect", AreaDialect.ROM) for a in areas}
    merged_dialect = next(iter(dialects)) if len(dialects) == 1 else AreaDialect.ROM

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
        dialect=merged_dialect,
    )


__all__ = [
    "AreaData",
    "AreaDialect",
    "AreaHeader",
    "DEFAULT_TERRAIN_SCALES",
    "ROOM_INDOORS_FLAG",
    "has_indoors_flag",
    "normalize_room_terrain",
    "DIALECT_SECTOR_TABLES",
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
    "TERRAIN_LABELS",
    "TERRAIN_SCALES_1_1_2",
    "TerrainType",
    "UNIFORM_TERRAIN_SCALES",
    "build_terrain_scales",
    "compute_room_footprints",
    "get_terrain_label",
    "get_terrain_scale",
    "merge_areas",
    "normalize_sector",
    "parse_terrain_scales",
    "resolve_dialect",
    "scale_to_extent",
]
