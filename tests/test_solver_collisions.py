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
from romutil.solver import solve, position_dummy_rooms, add_room_separation_constraint, build_spatial_coordinate_buckets, find_spatial_room_collisions, find_collinear_exit_room_penetrations, add_collinear_separation_constraint, add_exit_room_clearance_constraint
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


def test_find_collinear_exit_room_penetrations_cardinal_and_vertical() -> None:
    """Verify detection of collinear penetrations across East/West, North/South, and Up/Down."""
    # 1. East/West elongated exit: 1 <-> 2 (distance 4)
    # Room 1 at (0, 0, 0), Room 2 at (4, 0, 0)
    # Room 3 at (1, 0, 0), Room 4 at (2, 0, 0) penetrate
    e12 = Exit(direction=Direction.east, src=1, dst=2, distance=4)
    e21 = Exit(direction=Direction.west, src=2, dst=1, distance=4)
    exits = [e12, e21]
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (4.0, 0.0, 0.0),
        3: (1.0, 0.0, 0.0),
        4: (2.0, 0.0, 0.0),
    }
    rooms = [1, 2, 3, 4]
    penetrations = find_collinear_exit_room_penetrations(exits, coords, rooms=rooms)
    # Both exits 0 and 1 penetrate rooms 3 and 4:
    assert (0, 3) in penetrations
    assert (0, 4) in penetrations
    assert (1, 3) in penetrations
    assert (1, 4) in penetrations
    # Endpoints themselves (1 and 2) must not be reported:
    assert not any(w in (1, 2) for _, w in penetrations)

    # Test filtering with constrained_penetrations
    constrained = {(0, 3), (1, 3)}
    filtered = find_collinear_exit_room_penetrations(
        exits, coords, rooms=rooms, constrained_penetrations=constrained
    )
    assert (0, 3) not in filtered
    assert (1, 3) not in filtered
    assert (0, 4) in filtered
    assert (1, 4) in filtered

    # Test cut relaxation: exit 0 cut
    cuts = [True, False]
    cut_pen = find_collinear_exit_room_penetrations(exits, coords, cuts=cuts, rooms=rooms)
    assert not any(ex_idx == 0 for ex_idx, _ in cut_pen)
    assert (1, 3) in cut_pen
    assert (1, 4) in cut_pen

    # 2. North/South elongated exit: 10 <-> 20 (distance 3)
    # 10 at (0, 0, 0), 20 at (0, 3, 0), intermediate 30 at (0, 1, 0)
    e_ns = Exit(direction=Direction.north, src=10, dst=20, distance=3)
    coords_ns = {10: (0.0, 0.0, 0.0), 20: (0.0, 3.0, 0.0), 30: (0.0, 1.0, 0.0)}
    pen_ns = find_collinear_exit_room_penetrations([e_ns], coords_ns, rooms=[10, 20, 30])
    assert pen_ns == [(0, 30)]

    # 3. Up/Down vertical elongated exit: 100 <-> 200 (distance 3)
    # 100 at (2, 2, 0), 200 at (2, 2, 3), intermediate 300 at (2, 2, 2)
    e_vert = Exit(direction=Direction.up, src=100, dst=200, distance=3)
    coords_vert = {100: (2.0, 2.0, 0.0), 200: (2.0, 2.0, 3.0), 300: (2.0, 2.0, 2.0)}
    pen_vert = find_collinear_exit_room_penetrations([e_vert], coords_vert, rooms=[100, 200, 300])
    assert pen_vert == [(0, 300)]


def test_find_collinear_exit_room_penetrations_negative() -> None:
    """Verify non-penetrating rooms (adjacent, perpendicular, or outside segment) generate zero constraints."""
    # Exit 1 -> 2 along East from (0, 0, 0) to (3, 0, 0)
    e12 = Exit(direction=Direction.east, src=1, dst=2, distance=3)
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (3.0, 0.0, 0.0),
        # Perpendicularly adjacent to room 1:
        10: (0.0, 1.0, 0.0),
        # Perpendicular to intermediate coordinate (1, 0, 0):
        11: (1.0, 1.0, 0.0),
        # Collinear but beyond room 2:
        12: (4.0, 0.0, 0.0),
        # Collinear but before room 1:
        13: (-1.0, 0.0, 0.0),
        # Collinear in X/Y projection but at different elevation Z:
        14: (1.0, 0.0, 1.0),
        # Diagonal non-collinear:
        15: (2.0, 2.0, 0.0),
    }
    rooms = [1, 2, 10, 11, 12, 13, 14, 15]
    penetrations = find_collinear_exit_room_penetrations([e12], coords, rooms=rooms)
    assert penetrations == []

    # Distance 1 exit: no integer room can be between (0,0,0) and (1,0,0)
    e_short = Exit(direction=Direction.east, src=1, dst=2, distance=1)
    coords_short = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.5, 0.0, 0.0)}
    assert find_collinear_exit_room_penetrations([e_short], coords_short, rooms=[1, 2, 3]) == []

    # Empty exits or None coords
    assert find_collinear_exit_room_penetrations([], coords) == []
    assert find_collinear_exit_room_penetrations([e12], None) == []

    # Exit with None room coordinates
    coords_none = {1: (0.0, 0.0, 0.0), 2: None}
    assert find_collinear_exit_room_penetrations([e12], coords_none) == []


def test_add_collinear_separation_constraint_planar_and_3d() -> None:
    """Verify add_collinear_separation_constraint adds correct direction variables and Big-M bounds."""
    # 1. Planar model (has_vertical_exits=False)
    m = ConcreteModel()
    m.Rooms = [1, 2, 3]
    m.Exits = [0]
    m.x = Var(m.Rooms, within=Integers)
    m.y = Var(m.Rooms, within=Integers)
    m.z = Var(m.Rooms, within=Integers)
    m.cut = Var(m.Exits, within=Boolean)
    m.crossings = ConstraintList()
    m.Mx = 20
    m.My = 20
    m.Mz = 10

    ex = Exit(direction=Direction.east, src=1, dst=2, distance=3)
    coords = {1: (0.0, 0.0, 0.0), 2: (3.0, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    next_rel = add_collinear_separation_constraint(
        m, exit_idx=0, w=3, relations=0, ex=ex, coords=coords, has_vertical_exits=False
    )
    assert next_rel == 1
    assert hasattr(m, 'collinear_rel_0')
    rel_var = getattr(m, 'collinear_rel_0')
    assert len(rel_var) == 4
    assert Direction.north in rel_var
    assert Direction.east in rel_var
    assert Direction.south in rel_var
    assert Direction.west in rel_var
    assert Direction.up not in rel_var
    assert Direction.down not in rel_var
    # 1 sum constraint + (2 endpoints * 4 directions) = 9 constraints
    assert len(m.crossings) == 9

    # 2. 3D model (has_vertical_exits=True)
    m_3d = ConcreteModel()
    m_3d.Rooms = [1, 2, 3]
    m_3d.Exits = [0]
    m_3d.x = Var(m_3d.Rooms, within=Integers)
    m_3d.y = Var(m_3d.Rooms, within=Integers)
    m_3d.z = Var(m_3d.Rooms, within=Integers)
    m_3d.cut = Var(m_3d.Exits, within=Boolean)
    m_3d.crossings = ConstraintList()
    m_3d.Mx = 20
    m_3d.My = 20
    m_3d.Mz = 10

    next_rel_3d = add_collinear_separation_constraint(
        m_3d, exit_idx=0, w=3, relations=0, ex=ex, coords=None, has_vertical_exits=True
    )
    assert next_rel_3d == 1
    assert hasattr(m_3d, 'collinear_rel_0')
    rel_var_3d = getattr(m_3d, 'collinear_rel_0')
    assert len(rel_var_3d) == 6
    assert Direction.up in rel_var_3d
    assert Direction.down in rel_var_3d
    # 1 sum constraint + (2 endpoints * 6 directions) = 13 constraints
    assert len(m_3d.crossings) == 13

    # Alias check
    assert add_exit_room_clearance_constraint is add_collinear_separation_constraint


def test_collinear_exit_penetration_synthetic_planar_solve() -> None:
    """Verify solver resolves elongated exit piercing an intermediate room with 0 cuts and clean clearance."""
    # Room 1 <-> Room 2 connected by elongated East/West corridor of distance 3.
    # Parallel path: 1 -> 3 (North), 3 -> 4 (East), 4 -> 5 (South).
    # If 5 has distance 1, Room 5 would land at (1, 0, 0), piercing the 1 <-> 2 corridor!
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    r3 = Room(RoomDef(vnum=3, name="R3", description=""))
    r4 = Room(RoomDef(vnum=4, name="R4", description=""))
    r5 = Room(RoomDef(vnum=5, name="R5", description=""))

    e12 = Exit(direction=Direction.east, src=1, dst=2, distance=3)
    e21 = Exit(direction=Direction.west, src=2, dst=1, distance=3)

    e13 = Exit(direction=Direction.north, src=1, dst=3)
    e31 = Exit(direction=Direction.south, src=3, dst=1)
    e34 = Exit(direction=Direction.east, src=3, dst=4)
    e43 = Exit(direction=Direction.west, src=4, dst=3)
    e45 = Exit(direction=Direction.south, src=4, dst=5)
    e54 = Exit(direction=Direction.north, src=5, dst=4)

    r1.exits = [e12, e13]
    r2.exits = [e21]
    r3.exits = [e31, e34]
    r4.exits = [e43, e45]
    r5.exits = [e54]

    rooms = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5}
    exits = [e12, e21, e13, e31, e34, e43, e45, e54]

    model, results = solve(rooms, exits, timeout=15)
    for vnum, room in rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)
    cuts = sum(int(model.cut[i].value) for i in range(len(exits)))
    assert cuts == 0, f"Expected 0 cuts, got {cuts}"

    coords = {v: (r.x, r.y, r.z) for v, r in rooms.items()}
    # Collinear penetration check must be completely empty!
    pen = find_collinear_exit_room_penetrations(exits, coords, rooms=list(rooms.keys()))
    assert pen == [], f"Expected 0 collinear penetrations, got {pen}"


def test_collinear_exit_penetration_synthetic_3d_solve() -> None:
    """Verify solver resolves vertical elongated exit piercing an intermediate room in 3D."""
    # Room 1 <-> Room 2 connected by vertical Up/Down corridor of distance 3.
    # Path: 1 -> 3 (East), 3 -> 4 (Up), 4 -> 5 (West).
    # Room 5 lands at (0, 0, 1) piercing vertical corridor 1 <-> 2.
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    r3 = Room(RoomDef(vnum=3, name="R3", description=""))
    r4 = Room(RoomDef(vnum=4, name="R4", description=""))
    r5 = Room(RoomDef(vnum=5, name="R5", description=""))

    e12 = Exit(direction=Direction.up, src=1, dst=2, distance=3)
    e21 = Exit(direction=Direction.down, src=2, dst=1, distance=3)

    e13 = Exit(direction=Direction.east, src=1, dst=3)
    e31 = Exit(direction=Direction.west, src=3, dst=1)
    e34 = Exit(direction=Direction.up, src=3, dst=4)
    e43 = Exit(direction=Direction.down, src=4, dst=3)
    e45 = Exit(direction=Direction.west, src=4, dst=5)
    e54 = Exit(direction=Direction.east, src=5, dst=4)

    r1.exits = [e12, e13]
    r2.exits = [e21]
    r3.exits = [e31, e34]
    r4.exits = [e43, e45]
    r5.exits = [e54]

    rooms = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5}
    exits = [e12, e21, e13, e31, e34, e43, e45, e54]

    model, results = solve(rooms, exits, timeout=15)
    for vnum, room in rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)
    cuts = sum(int(model.cut[i].value) for i in range(len(exits)))
    assert cuts == 0

    coords = {v: (r.x, r.y, r.z) for v, r in rooms.items()}
    pen = find_collinear_exit_room_penetrations(exits, coords, rooms=list(rooms.keys()))
    assert pen == []


def test_add_collinear_separation_constraint_with_dummy_anchors() -> None:
    """Verify add_collinear_separation_constraint resolves dummy anchor offsets."""
    m = ConcreteModel()
    m.Rooms = [1, 3]  # Room 2 is a dummy room, not in m.Rooms
    m.Exits = [0]
    m.x = Var(m.Rooms, within=Integers)
    m.y = Var(m.Rooms, within=Integers)
    m.z = Var(m.Rooms, within=Integers)
    m.cut = Var(m.Exits, within=Boolean)
    m.crossings = ConstraintList()
    m.Mx = 20
    m.My = 20
    m.Mz = 10

    ex = Exit(direction=Direction.east, src=1, dst=2, distance=3)
    dummy_anchors = {2: (1, 3, 0, 0)}  # Anchor room 2 to room 1 with (+3, 0, 0)
    next_rel = add_collinear_separation_constraint(
        m, exit_idx=0, w=3, relations=0, ex=ex, dummy_anchors=dummy_anchors, has_vertical_exits=True
    )
    assert next_rel == 1
    assert len(m.crossings) == 13


def test_find_collinear_exit_room_penetrations_branches() -> None:
    """Verify edge cases: model coords, Pyomo cut variables, non-integer coords, empty vnums."""
    # 1. Pyomo model with Rooms attribute
    m = ConcreteModel()
    m.Rooms = [1, 2, 3]
    m.x = Var(m.Rooms, within=Integers)
    m.y = Var(m.Rooms, within=Integers)
    m.z = Var(m.Rooms, within=Integers)
    m.Exits = [0]
    m.cut = Var(m.Exits, within=Boolean)
    m.x[1].set_value(0)
    m.y[1].set_value(0)
    m.z[1].set_value(0)
    m.x[2].set_value(3)
    m.y[2].set_value(0)
    m.z[2].set_value(0)
    m.x[3].set_value(1)
    m.y[3].set_value(0)
    m.z[3].set_value(0)
    m.cut[0].set_value(0)

    ex = Exit(direction=Direction.east, src=1, dst=2, distance=3)
    pen = find_collinear_exit_room_penetrations([ex], m, cuts=m.cut)
    assert pen == [(0, 3)]

    # Cut is 1 with Pyomo Var
    m.cut[0].set_value(1)
    assert find_collinear_exit_room_penetrations([ex], m, cuts=m.cut) == []

    # 2. Non-integer floating point coordinates deviation > 1e-4
    coords_float = {1: (0.0, 0.0, 0.0), 2: (3.05, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    assert find_collinear_exit_room_penetrations([ex], coords_float) == []

    # 3. Object without Rooms or keys
    assert find_collinear_exit_room_penetrations([ex], "invalid_coords_object") == []


def test_collinear_exit_penetration_batch_capping() -> None:
    """Verify candidate batch capping triggers when many rooms penetrate an elongated corridor."""
    # Elongated Corridor 1 <-> 2 with distance 10
    r1 = Room(RoomDef(vnum=1, name="R1", description=""))
    r2 = Room(RoomDef(vnum=2, name="R2", description=""))
    e12 = Exit(direction=Direction.east, src=1, dst=2, distance=10)
    e21 = Exit(direction=Direction.west, src=2, dst=1, distance=10)
    r1.exits = [e12]
    r2.exits = [e21]

    rooms = {1: r1, 2: r2}
    exits = [e12, e21]

    # Create a parallel chain of 7 rooms 10..16: 1 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16
    # Each separated by 1 unit East, naturally landing at (1,0,0)..(7,0,0) collinear with 1<->2!
    prev = 1
    for i in range(10, 17):
        r = Room(RoomDef(vnum=i, name=f"R{i}", description=""))
        r.exits = []
        rooms[i] = r
        ef = Exit(direction=Direction.east, src=prev, dst=i, distance=1)
        er = Exit(direction=Direction.west, src=i, dst=prev, distance=1)
        rooms[prev].exits.append(ef)
        rooms[i].exits.append(er)
        exits.extend([ef, er])
        prev = i

    # Solve end-to-end: batch capping triggers because 7 rooms penetrate 1<->2, capped at 5 per batch
    model, results = solve(rooms, exits, timeout=20)
    for vnum, room in rooms.items():
        room.x = int(round(model.x[vnum].value))
        room.y = int(round(model.y[vnum].value))
        room.z = int(round(model.z[vnum].value))

    assert_no_room_collisions(rooms)
    solved_coords = {v: (r.x, r.y, r.z) for v, r in rooms.items()}
    assert find_collinear_exit_room_penetrations(exits, solved_coords, rooms=list(rooms.keys())) == []
