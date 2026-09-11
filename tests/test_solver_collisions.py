"""Regression and verification tests ensuring zero coordinate collisions across layouts.

Prevents room coordinate collisions and ensures converging one-way funnels (e.g. school.are arena)
solve compactly with 0 cuts and 0 duplicate (x, y, z) coordinates (Task 1l).
"""

from collections.abc import Mapping
from pathlib import Path
import time
import pytest

from romutil.graph import solve_layout
from romutil.models import Direction, Exit, Room, RoomDef
from romutil.parser import Parser
from romutil.solver import solve, position_dummy_rooms

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def assert_no_room_collisions(rdb: Mapping[int, Room]) -> None:
    """Verify that for all non-dummy rooms in rdb, every room has a unique (x, y, z) tuple.

    If any collision is found, raises AssertionError detailing the colliding coordinates and VNUMs.
    """
    coords: dict[tuple[int, int, int], list[int]] = {}
    for vnum, room in rdb.items():
        if getattr(room, "dummy", False):
            continue
        assert room.x is not None and room.y is not None and room.z is not None, (
            f"Room {vnum} has unassigned coordinates: x={room.x}, y={room.y}, z={room.z}"
        )
        pos = (int(room.x), int(room.y), int(room.z))
        coords.setdefault(pos, []).append(vnum)

    collisions = {pos: vnums for pos, vnums in coords.items() if len(vnums) > 1}
    assert not collisions, f"Detected room coordinate collisions: {collisions}"


@pytest.mark.slow
@pytest.mark.integration
def test_school_are_zero_collisions_and_zero_cuts() -> None:
    """Verify school.are solves with 0 collisions, 0 cuts, and within 10 seconds."""
    filepath = FIXTURES_DIR / "areas" / "school.are"
    area = Parser().parse(filepath.read_text(encoding="latin-1"))
    rdb = {r.vnum: Room(r) for r in area.rooms}

    t0 = time.perf_counter()
    solved_rdb, exits = solve_layout(rdb, area, solver_timeout=30)
    elapsed = time.perf_counter() - t0

    assert_no_room_collisions(solved_rdb)
    cuts = sum(1 for e in exits if getattr(e, "cut", False))
    assert cuts == 0, f"Expected 0 cuts in school.are, got {cuts}"
    assert elapsed < 10.0, f"Solve time {elapsed:.2f}s exceeded 10.0s threshold"


@pytest.mark.slow
@pytest.mark.integration
def test_smurf_are_zero_collisions() -> None:
    """Verify smurf.are solves with 0 collisions and 0 cuts."""
    filepath = FIXTURES_DIR / "areas" / "smurf.are"
    area = Parser().parse(filepath.read_text(encoding="latin-1"))
    rdb = {r.vnum: Room(r) for r in area.rooms}

    solved_rdb, exits = solve_layout(rdb, area, solver_timeout=30)

    assert_no_room_collisions(solved_rdb)
    cuts = sum(1 for e in exits if getattr(e, "cut", False))
    assert cuts == 0, f"Expected 0 cuts in smurf.are, got {cuts}"


@pytest.mark.parametrize("funnel_dir", [Direction.up, Direction.north])
def test_converging_oneway_funnel_no_collisions(funnel_dir: Direction) -> None:
    """Verify that a 3x3 grid funneling via one-way exits to a single room has 0 collisions."""
    rooms: dict[int, Room] = {}
    exits: list[Exit] = []

    # 3x3 planar grid (vnums 1..9)
    for y in range(3):
        for x in range(3):
            vnum = y * 3 + x + 1
            r = Room(RoomDef(vnum=vnum, name=f"Room {vnum}", description=""))
            r.exits = []
            rooms[vnum] = r

    for y in range(3):
        for x in range(3):
            vnum = y * 3 + x + 1
            if x < 2:
                e_east = Exit(direction=Direction.east, src=vnum, dst=vnum + 1)
                e_west = Exit(direction=Direction.west, src=vnum + 1, dst=vnum)
                rooms[vnum].exits.append(e_east)
                rooms[vnum + 1].exits.append(e_west)
                exits.extend([e_east, e_west])
            if y < 2:
                e_north = Exit(direction=Direction.north, src=vnum, dst=vnum + 3)
                e_south = Exit(direction=Direction.south, src=vnum + 3, dst=vnum)
                rooms[vnum].exits.append(e_north)
                rooms[vnum + 3].exits.append(e_south)
                exits.extend([e_north, e_south])

    # Safe destination room
    dest_room = Room(RoomDef(vnum=10, name="Dest Room", description=""))
    dest_room.exits = []
    rooms[10] = dest_room

    # One-way funnel from all 9 grid rooms to destination room
    for vnum in range(1, 10):
        e_funnel = Exit(direction=funnel_dir, src=vnum, dst=10)
        rooms[vnum].exits.append(e_funnel)
        exits.append(e_funnel)

    model, results = solve(rooms, exits, timeout=10)
    for vnum, room in rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)
    cuts = sum(int(model.cut[i].value) for i in range(len(exits)))
    assert cuts == 0, f"Expected 0 cuts in converging funnel ({funnel_dir.name}), got {cuts}"


def assert_no_room_or_dummy_collisions(rdb: Mapping[int, Room]) -> None:
    """Verify that every room (both core rooms and dummy stubs) has a unique coordinate."""
    coords: dict[tuple[int, int, int], list[int]] = {}
    for vnum, room in rdb.items():
        assert room.x is not None and room.y is not None and room.z is not None, (
            f"Room {vnum} has unassigned coordinates: x={room.x}, y={room.y}, z={room.z}"
        )
        pos = (int(round(room.x)), int(round(room.y)), int(round(room.z)))
        coords.setdefault(pos, []).append(vnum)

    collisions = {pos: vnums for pos, vnums in coords.items() if len(vnums) > 1}
    assert not collisions, f"Detected coordinate collisions (including dummy stubs): {collisions}"


def test_core_room_dummy_stub_collision_avoidance() -> None:
    """Verify linear disjunctive separation prevents a core room from overlapping an affine dummy stub."""
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    dummy = Room(RoomDef(vnum=999, name="D999", description=""))
    dummy.dummy = True

    # r1 exits North to external dummy 999 (stub coordinate: (r1.x, r1.y + 1, r1.z))
    e_to_dummy = Exit(direction=Direction.north, src=1, dst=999)
    # r2 exits South to r1 (nominal delta (0, -1, 0) => r2.y = r1.y + 1)
    e_r2_to_r1 = Exit(direction=Direction.south, src=2, dst=1)

    r1.exits = [e_to_dummy]
    r2.exits = [e_r2_to_r1]
    rooms = {1: r1, 2: r2, 999: dummy}
    exits = [e_to_dummy, e_r2_to_r1]

    model, results = solve(rooms, exits, timeout=10)
    for vnum in (1, 2):
        rooms[vnum].x = int(round(model.x[vnum].value))
        rooms[vnum].y = int(round(model.y[vnum].value))
        rooms[vnum].z = int(round(model.z[vnum].value))

    # Position dummy stub using affine anchor
    position_dummy_rooms(rooms, exits)

    # Core room 2 and dummy stub 999 must NOT share coordinates
    assert_no_room_or_dummy_collisions(rooms)
    pos2 = (rooms[2].x, rooms[2].y, rooms[2].z)
    pos_dummy = (dummy.x, dummy.y, dummy.z)
    assert pos2 != pos_dummy


def test_vertical_dummy_stub_collision_avoidance() -> None:
    """Verify vertical disjunctive separation prevents a core room from colliding with a vertical dummy stub."""
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    dummy = Room(RoomDef(vnum=999, name="D999", description=""))
    dummy.dummy = True

    # r1 exits Up to dummy 999 (stub coordinate: (r1.x, r1.y, r1.z + 1))
    e_to_dummy = Exit(direction=Direction.up, src=1, dst=999)
    # r2 exits Down to r1 (nominal delta (0, 0, -1) => r2.z = r1.z + 1)
    e_r2_to_r1 = Exit(direction=Direction.down, src=2, dst=1)

    r1.exits = [e_to_dummy]
    r2.exits = [e_r2_to_r1]
    rooms = {1: r1, 2: r2, 999: dummy}
    exits = [e_to_dummy, e_r2_to_r1]

    model, results = solve(rooms, exits, timeout=10)
    for vnum in (1, 2):
        rooms[vnum].x = int(round(model.x[vnum].value))
        rooms[vnum].y = int(round(model.y[vnum].value))
        rooms[vnum].z = int(round(model.z[vnum].value))

    position_dummy_rooms(rooms, exits)
    assert_no_room_or_dummy_collisions(rooms)


@pytest.mark.slow
@pytest.mark.integration
def test_school_are_zero_dummy_collisions() -> None:
    """Verify school.are layout has zero room-to-dummy coordinate collisions."""
    filepath = FIXTURES_DIR / "areas" / "school.are"
    area = Parser().parse(filepath.read_text(encoding="latin-1"))
    rdb = {r.vnum: Room(r) for r in area.rooms}

    solved_rdb, exits = solve_layout(rdb, area, solver_timeout=30)
    assert_no_room_or_dummy_collisions(solved_rdb)


def test_multi_dummy_grid_no_collisions() -> None:
    """Verify a 2x2 grid with multiple external dummy stubs has zero collisions."""
    rooms: dict[int, Room] = {}
    exits: list[Exit] = []

    for i in range(1, 5):
        r = Room(RoomDef(vnum=i, name=f"R{i}", description=""))
        r.exits = []
        rooms[i] = r

    def add_bi(u: int, v: int, d: Direction):
        e1 = Exit(direction=d, src=u, dst=v)
        e2 = Exit(direction=d.invert(), src=v, dst=u)
        rooms[u].exits.append(e1)
        rooms[v].exits.append(e2)
        exits.extend([e1, e2])

    add_bi(1, 2, Direction.east)
    add_bi(3, 4, Direction.east)
    add_bi(1, 3, Direction.south)
    add_bi(2, 4, Direction.south)

    for i, d in [(1, Direction.north), (2, Direction.north), (3, Direction.west), (4, Direction.up)]:
        d_vnum = 1000 + i
        d_room = Room(RoomDef(vnum=d_vnum, name=f"D{d_vnum}", description=""))
        d_room.dummy = True
        rooms[d_vnum] = d_room
        e_dummy = Exit(direction=d, src=i, dst=d_vnum)
        rooms[i].exits.append(e_dummy)
        exits.append(e_dummy)

    model, results = solve(rooms, exits, timeout=10)
    for vnum, room in rooms.items():
        if getattr(room, "dummy", False):
            continue
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    position_dummy_rooms(rooms, exits)
    assert_no_room_or_dummy_collisions(rooms)


@pytest.mark.slow
@pytest.mark.integration
def test_collision_school_are_pairwise_unique_coordinates() -> None:
    """Verify school.are solves with pairwise unique coordinates and zero cuts."""
    test_school_are_zero_collisions_and_zero_cuts()


@pytest.mark.slow
@pytest.mark.integration
def test_collision_smurf_are_pairwise_unique_coordinates() -> None:
    """Verify smurf.are solves with pairwise unique coordinates and zero cuts."""
    test_smurf_are_zero_collisions()
