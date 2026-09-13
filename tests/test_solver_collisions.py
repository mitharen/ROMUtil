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
from romutil.solver import solve, position_dummy_rooms, add_room_separation_constraint, build_spatial_coordinate_buckets, find_spatial_room_collisions
from pyomo.environ import ConcreteModel, Var, ConstraintList, Integers, Boolean

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


def test_add_room_separation_constraint_planar() -> None:
    """Verify planar room separation constraint adds 4 directional variables and Big-M bounds."""
    m = ConcreteModel()
    m.Rooms = [1, 2]
    m.x = Var(m.Rooms, within=Integers)
    m.y = Var(m.Rooms, within=Integers)
    m.z = Var(m.Rooms, within=Integers)
    m.crossings = ConstraintList()
    m.Mx = 20
    m.My = 20
    m.Mz = 10

    coords = {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0)}
    next_rel = add_room_separation_constraint(
        m, 1, 2, relations=0, coords=coords, has_vertical_exits=False
    )

    assert next_rel == 1
    assert hasattr(m, 'room_rel_0')
    rel_var = getattr(m, 'room_rel_0')
    assert len(rel_var) == 4
    assert Direction.north in rel_var
    assert Direction.east in rel_var
    assert Direction.south in rel_var
    assert Direction.west in rel_var
    assert Direction.up not in rel_var
    assert Direction.down not in rel_var
    # 1 disjunctive sum constraint (sum >= 1) + 4 directional separation inequalities
    assert len(m.crossings) == 5


def test_add_room_separation_constraint_3d() -> None:
    """Verify 3D room separation constraint adds 6 directional variables and Big-M bounds."""
    m = ConcreteModel()
    m.Rooms = [1, 2]
    m.x = Var(m.Rooms, within=Integers)
    m.y = Var(m.Rooms, within=Integers)
    m.z = Var(m.Rooms, within=Integers)
    m.crossings = ConstraintList()
    m.Mx = 20
    m.My = 20
    m.Mz = 10

    coords = {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0)}
    next_rel = add_room_separation_constraint(
        m, 1, 2, relations=0, coords=coords, has_vertical_exits=True
    )

    assert next_rel == 1
    assert hasattr(m, 'room_rel_0')
    rel_var = getattr(m, 'room_rel_0')
    assert len(rel_var) == 6
    assert Direction.north in rel_var
    assert Direction.east in rel_var
    assert Direction.south in rel_var
    assert Direction.west in rel_var
    assert Direction.up in rel_var
    assert Direction.down in rel_var
    # 1 disjunctive sum constraint (sum >= 1) + 6 directional separation inequalities
    assert len(m.crossings) == 7


def test_core_room_point_collision_positive_separation_disconnected() -> None:
    """Verify that disconnected components that would otherwise collapse to identical

    coordinates are detected and separated into distinct coordinates.
    """
    # Path 1: 1 -> 2 (East)
    # Path 2: 3 -> 4 (East)
    # Both paths are disconnected; without collision cuts they would collapse to (0,0,0) and (1,0,0).
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    r3 = Room(RoomDef(vnum=3, name="R3", description=""))
    r4 = Room(RoomDef(vnum=4, name="R4", description=""))

    e12 = Exit(direction=Direction.east, src=1, dst=2)
    e21 = Exit(direction=Direction.west, src=2, dst=1)
    e34 = Exit(direction=Direction.east, src=3, dst=4)
    e43 = Exit(direction=Direction.west, src=4, dst=3)

    r1.exits = [e12]
    r2.exits = [e21]
    r3.exits = [e34]
    r4.exits = [e43]

    rooms = {1: r1, 2: r2, 3: r3, 4: r4}
    exits = [e12, e21, e34, e43]

    model, results = solve(rooms, exits, timeout=10)
    for vnum in (1, 2, 3, 4):
        rooms[vnum].x = int(round(model.x[vnum].value))
        rooms[vnum].y = int(round(model.y[vnum].value))
        rooms[vnum].z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)


def test_core_room_point_collision_positive_separation_parallel_branches() -> None:
    """Verify that parallel branches converging on the same relative delta are pushed apart."""
    # Room 1 exits East to Room 2 and East to Room 3.
    # Without collision constraints, Room 2 and Room 3 would both land at (1, 0, 0).
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    r3 = Room(RoomDef(vnum=3, name="R3", description=""))

    e12 = Exit(direction=Direction.east, src=1, dst=2)
    e21 = Exit(direction=Direction.west, src=2, dst=1)
    e13 = Exit(direction=Direction.east, src=1, dst=3)
    e31 = Exit(direction=Direction.west, src=3, dst=1)

    r1.exits = [e12, e13]
    r2.exits = [e21]
    r3.exits = [e31]

    rooms = {1: r1, 2: r2, 3: r3}
    exits = [e12, e21, e13, e31]

    model, results = solve(rooms, exits, timeout=10)
    for vnum in (1, 2, 3):
        rooms[vnum].x = int(round(model.x[vnum].value))
        rooms[vnum].y = int(round(model.y[vnum].value))
        rooms[vnum].z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)
    pos2 = (rooms[2].x, rooms[2].y, rooms[2].z)
    pos3 = (rooms[3].x, rooms[3].y, rooms[3].z)
    assert pos2 != pos3


def test_core_room_point_collision_deduplication_and_filtering() -> None:
    """Verify that dummy rooms are filtered out from core room collisions and that

    constrained pairs are not redundantly re-added.
    """
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    dummy = Room(RoomDef(vnum=999, name="D999", description=""))
    dummy.dummy = True

    e_to_dummy = Exit(direction=Direction.north, src=1, dst=999)
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

    position_dummy_rooms(rooms, exits)
    assert_no_room_or_dummy_collisions(rooms)
    # Core rooms 1 and 2 must have distinct coordinates
    assert (rooms[1].x, rooms[1].y, rooms[1].z) != (rooms[2].x, rooms[2].y, rooms[2].z)


def test_add_room_separation_constraint_coords_none() -> None:
    """Verify add_room_separation_constraint functions gracefully when coords is None."""
    m = ConcreteModel()
    m.Rooms = [1, 2]
    m.x = Var(m.Rooms, within=Integers)
    m.y = Var(m.Rooms, within=Integers)
    m.z = Var(m.Rooms, within=Integers)
    m.crossings = ConstraintList()

    next_rel = add_room_separation_constraint(m, 1, 2, relations=0, coords=None, has_vertical_exits=False)
    assert next_rel == 1
    assert hasattr(m, 'room_rel_0')
    assert len(m.room_rel_0) == 4


def test_core_room_point_collision_batch_capping() -> None:
    """Verify candidate batch capping triggers across iterations when many core rooms collide."""
    # 6 disconnected rooms with no exits: without cuts, all 6 collapse to (0, 0, 0)
    # 15 colliding pairs (> batch_cap of 5), exercising candidate batch capping
    rooms: dict[int, Room] = {}
    for i in range(1, 7):
        r = Room(RoomDef(vnum=i, name=f"R{i}", description=""))
        r.exits = []
        rooms[i] = r

    exits: list[Exit] = []
    model, results = solve(rooms, exits, timeout=15)

    for vnum, room in rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)
    assert len({(r.x, r.y, r.z) for r in rooms.values()}) == 6


def test_spatial_hash_room_collisions_single_and_multi_room_buckets() -> None:
    """Verify O(V) spatial hash bucketing accurately detects collisions across single

    and multi-room buckets (including 3+ coincident rooms generating all pairwise cuts).
    """
    # 1. Direct unit verification of build_spatial_coordinate_buckets
    # Coordinates with:
    # - Bucket at (0, 0, 0): 3 coincident rooms (101, 102, 103)
    # - Bucket at (5, 5, 0): 2 coincident rooms (201, 202)
    # - Bucket at (10, 10, 0): 1 isolated room (301)
    # - Room with None coordinates: room 401
    coords = {
        101: (0.0, 0.0, 0.0),
        102: (0.0, 0.0, 0.0),
        103: (0.0, 0.0, 0.0),
        201: (5.0, 5.0, 0.0),
        202: (5.0, 5.0, 0.0),
        301: (10.0, 10.0, 0.0),
        401: None,
    }
    rooms = [101, 102, 103, 201, 202, 301, 401]
    buckets = build_spatial_coordinate_buckets(rooms, coords)

    assert len(buckets) == 3
    assert buckets[(0.0, 0.0, 0.0)] == [101, 102, 103]
    assert buckets[(5.0, 5.0, 0.0)] == [201, 202]
    assert buckets[(10.0, 10.0, 0.0)] == [301]
    assert 401 not in [v for vnums in buckets.values() for v in vnums]

    # 2. Direct unit verification of find_spatial_room_collisions
    collisions = find_spatial_room_collisions(buckets)
    # 3 coincident rooms produce 3 pairs: (101, 102), (101, 103), (102, 103)
    # 2 coincident rooms produce 1 pair: (201, 202)
    # Single room bucket produces 0 pairs
    expected_collisions = [(101, 102), (101, 103), (102, 103), (201, 202)]
    assert collisions == expected_collisions

    # Verify filtering of already-constrained pairs
    constrained = {(101, 102), (201, 202)}
    filtered_collisions = find_spatial_room_collisions(buckets, constrained_room_collisions=constrained)
    assert filtered_collisions == [(101, 103), (102, 103)]

    # 3. Test 4 coincident rooms generating all 6 pairwise combinations
    coords_4 = {1: (1.0, 2.0, 3.0), 2: (1.0, 2.0, 3.0), 3: (1.0, 2.0, 3.0), 4: (1.0, 2.0, 3.0)}
    buckets_4 = build_spatial_coordinate_buckets([1, 2, 3, 4], coords_4)
    collisions_4 = find_spatial_room_collisions(buckets_4)
    assert collisions_4 == [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

    # 4. End-to-end solver integration: 3 disconnected rooms with no exits
    # All 3 rooms start at (0, 0, 0) and must be separated by pairwise cuts to unique coordinates
    solve_rooms: dict[int, Room] = {}
    for i in (1, 2, 3):
        r = Room(RoomDef(vnum=i, name=f"Room {i}", description=""))
        r.exits = []
        solve_rooms[i] = r

    model, results = solve(solve_rooms, [], timeout=10)
    for vnum, room in solve_rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(solve_rooms)
    assert len({(r.x, r.y, r.z) for r in solve_rooms.values()}) == 3


def test_spatial_hash_dummy_stub_collision_detection() -> None:
    """Verify O(D) spatial hash lookup detects dummy stub collisions and respects source room exclusions."""
    # 1. Direct unit verification of spatial hash lookup mechanics
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 2.0, 0.0),
        999: (1.0, 0.0, 0.0),  # Dummy stub positioned at room 2's coordinates
        998: (0.0, 0.0, 0.0),  # Dummy stub positioned at anchor room 1's coordinates
        997: (10.0, 10.0, 0.0),  # Dummy stub at unoccupied coordinate
    }
    non_dummy = [1, 2, 3]
    buckets = build_spatial_coordinate_buckets(non_dummy, coords)

    dummy_anchors = {
        999: (1, 1, 0, 0),    # Anchored to 1 with offset East (+1, 0, 0) => pos (1, 0, 0)
        998: (1, 0, 0, 0),    # Anchored to 1 with offset (0, 0, 0) => pos (0, 0, 0)
        997: (1, 10, 10, 0),  # Anchored to 1 with offset (+10, +10, 0) => pos (10, 10, 0)
    }

    # Simulate solver dummy loop logic using spatial hash lookup
    detected_dummy_collisions: list[tuple[int, int]] = []
    for dv, anchor in dummy_anchors.items():
        src_vnum, dx, dy, dz = anchor
        d_pos = coords[dv]
        if d_pos is None or d_pos not in buckets:
            continue
        for vnum in buckets[d_pos]:
            if vnum == src_vnum:
                continue
            detected_dummy_collisions.append((vnum, dv))

    # Core room 2 collides with dummy 999: (2, 999) detected
    # Core room 1 at (0, 0, 0) is the anchor for dummy 998: correctly skipped by vnum == src_vnum
    # Dummy 997 is at unoccupied coordinate: correctly skipped by d_pos not in buckets
    assert detected_dummy_collisions == [(2, 999)]

    # 2. End-to-end solve with 2 core rooms and 1 dummy stub
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    dummy = Room(RoomDef(vnum=999, name="D999", description=""))
    dummy.dummy = True

    e_to_dummy = Exit(direction=Direction.east, src=1, dst=999)
    e_r2_to_r1 = Exit(direction=Direction.west, src=2, dst=1)
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
    assert (rooms[2].x, rooms[2].y, rooms[2].z) != (dummy.x, dummy.y, dummy.z)


def test_spatial_hash_negative_disjoint_rooms_zero_cuts() -> None:
    """Verify that disjoint layouts produce single-room buckets, zero collisions, and zero cuts."""
    # 1. Direct unit verification: distinct coordinates yield empty collision pairs
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (2.0, 0.0, 0.0),
        4: (0.0, 1.0, 0.0),
    }
    rooms = [1, 2, 3, 4]
    buckets = build_spatial_coordinate_buckets(rooms, coords)
    assert len(buckets) == 4
    for pt, vnums in buckets.items():
        assert len(vnums) == 1

    collisions = find_spatial_room_collisions(buckets)
    assert collisions == []

    # Empty buckets verification
    empty_buckets: dict[tuple[float, float, float], list[int]] = {}
    assert find_spatial_room_collisions(empty_buckets) == []

    # Empty rooms sequence
    assert build_spatial_coordinate_buckets([], coords) == {}

    # 2. End-to-end solve with 3-room linear chain (1 <-> 2 <-> 3)
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    r3 = Room(RoomDef(vnum=3, name="R3", description=""))

    e12 = Exit(direction=Direction.east, src=1, dst=2)
    e21 = Exit(direction=Direction.west, src=2, dst=1)
    e23 = Exit(direction=Direction.east, src=2, dst=3)
    e32 = Exit(direction=Direction.west, src=3, dst=2)

    r1.exits = [e12]
    r2.exits = [e21, e23]
    r3.exits = [e32]

    chain_rooms = {1: r1, 2: r2, 3: r3}
    chain_exits = [e12, e21, e23, e32]

    model, results = solve(chain_rooms, chain_exits, timeout=10)
    for vnum, room in chain_rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(chain_rooms)
    cuts = sum(int(model.cut[i].value) for i in range(len(chain_exits)))
    assert cuts == 0
    # Coordinates must be strictly collinear and separated by exactly 1 unit along East/West
    assert chain_rooms[1].x is not None and chain_rooms[2].x is not None and chain_rooms[3].x is not None
    assert chain_rooms[1].y is not None and chain_rooms[2].y is not None and chain_rooms[3].y is not None
    assert chain_rooms[1].z is not None and chain_rooms[2].z is not None and chain_rooms[3].z is not None
    assert chain_rooms[2].x - chain_rooms[1].x == 1
    assert chain_rooms[3].x - chain_rooms[2].x == 1
    assert chain_rooms[1].y == chain_rooms[2].y == chain_rooms[3].y
    assert chain_rooms[1].z == chain_rooms[2].z == chain_rooms[3].z
