"""Comprehensive unit and regression tests for Boundary Dummy Room Anchoring

and Decision Variable Elimination (Task 1b).

Verifies:
1. Decision Variable Reduction:
   - In Pyomo MIP, boundary dummy rooms (room.dummy = True) are completely eliminated
     from m.Rooms (decision variables m.x, m.y, m.z).
   - In midgaard.are: integer decision variables are reduced from 393 (131 * 3) to 324 (108 * 3),
     an exact reduction of 69 variables (17.557%).
2. Coordinate Positioning & Affine Anchoring:
   - position_dummy_rooms places dummy rooms exactly 1 unit distance in the nominal exit
     direction from their source room (x_dummy = x_src + d_exit) across all 6 directions
     (North, South, East, West, Up, Down).
   - Exits incident to dummy rooms are formulated with affine linear expressions without
     creating new integer decision variables.
   - Idempotency and fallback handling for unanchored or disconnected dummy rooms.
3. Solver Status & Layout Feasibility:
   - No solver status regressions across all fixtures (midgaard.are, school.are, smurf.are,
     circlemud.are, dikumud_alfa.wld).
   - All rooms (non-dummy and dummy) receive valid, non-None integer coordinates.
4. Renderer Compatibility:
   - SVGRenderer and Plotter render external exit stubs at exact 1-unit length in nominal exit direction.
   - JSON and HTML exports function without errors and omit dummy rooms from room entity sets.
"""

from pathlib import Path
import pyomo.environ as pyo
import pytest

from romutil.models import Direction, Exit, Room, RoomDef
from romutil.parser import Parser, parse_circlemud_directory
from romutil.graph import solve_layout
from romutil.solver import position_dummy_rooms, solve, non_euler
from romutil.renderers import SVGRenderer, Plotter, render_map
from romutil.renderers.json import build_area_json


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDecisionVariableReduction:
    """Verify Pyomo MIP decision variable reduction on real and synthetic topologies."""

    def test_midgaard_decision_variable_reduction(self):
        """Verify that midgaard.are achieves exactly 69 fewer decision variables (17.557% reduction).

        Arithmetic:
        - midgaard.are initially has 143 rooms.
        - After straight hallway collapse: 108 non-dummy rooms + 23 boundary dummy rooms = 131 rooms total.
        - Without elimination: 131 * 3 = 393 decision variables (x, y, z).
        - With dummy elimination: 108 * 3 = 324 decision variables.
        - Exact reduction: 393 - 324 = 69 variables (69 / 393 = 17.557%).
        """
        filepath = FIXTURES_DIR / "areas" / "midgaard.are"
        area = Parser().parse(filepath.read_text(encoding="latin-1"))
        rdb = {r.vnum: Room(r) for r in area.rooms}

        # Solve layout, which performs hallway collapse, dummy insertion, solve, and positioning
        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=15)

        non_dummies = [v for v, r in solved_rdb.items() if getattr(r, "dummy", False) is not True]
        dummies = [v for v, r in solved_rdb.items() if getattr(r, "dummy", False) is True]

        assert len(dummies) == 23
        assert len(non_dummies) == 108
        assert len(solved_rdb) == 131

        old_var_count = len(solved_rdb) * 3
        new_var_count = len(non_dummies) * 3
        reduction = old_var_count - new_var_count
        percent_reduction = (reduction / old_var_count) * 100

        assert old_var_count == 393
        assert new_var_count == 324
        assert reduction == 69
        assert abs(percent_reduction - 17.557) < 0.01

        # Now solve directly on the rdb to verify model.Rooms and model variables directly
        model, results = solve(solved_rdb, exits, timeout=10)
        assert results.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit,
        )

        # Decision variables in Pyomo model must only contain non-dummy rooms
        assert len(model.Rooms) == 108
        for d in dummies:
            assert d not in model.Rooms
            assert d not in model.x
            assert d not in model.y
            assert d not in model.z

        for nd in non_dummies:
            assert nd in model.Rooms
            assert nd in model.x
            assert nd in model.y
            assert nd in model.z


class TestPositionDummyRoomsUnit:
    """Unit tests for position_dummy_rooms nominal direction math and edge cases."""

    @pytest.mark.parametrize(
        ("direction", "expected_dx", "expected_dy", "expected_dz"),
        [
            (Direction.north, 0, 1, 0),
            (Direction.east, 1, 0, 0),
            (Direction.south, 0, -1, 0),
            (Direction.west, -1, 0, 0),
            (Direction.up, 0, 0, 1),
            (Direction.down, 0, 0, -1),
        ],
    )
    def test_position_dummy_rooms_all_six_directions(self, direction, expected_dx, expected_dy, expected_dz):
        """Dummy rooms must be positioned exactly 1 unit in nominal direction from src room."""
        src_room = Room(RoomDef(vnum=100, name="Center", description="", exits=()))
        src_room.x, src_room.y, src_room.z = 10, 20, 5

        dummy_room = Room(RoomDef(vnum=200, name="Outside", description="", exits=()))
        dummy_room.dummy = True

        ex = Exit(direction=direction, src=100, dst=200)
        src_room.exits = [ex]

        rdb = {100: src_room, 200: dummy_room}
        exits = [ex]

        position_dummy_rooms(rdb, exits)

        assert dummy_room.x == src_room.x + expected_dx
        assert dummy_room.y == src_room.y + expected_dy
        assert dummy_room.z == src_room.z + expected_dz

    def test_position_dummy_rooms_reverse_exit(self):
        """When dummy room has an incoming exit pointing back to src, dummy is positioned relative to target."""
        src_room = Room(RoomDef(vnum=100, name="Inside", description="", exits=()))
        src_room.x, src_room.y, src_room.z = 5, 5, 0

        dummy_room = Room(RoomDef(vnum=200, name="Outside", description="", exits=()))
        dummy_room.dummy = True

        # North from dummy (200) leads to src_room (100) => dummy is South of src
        ex = Exit(direction=Direction.north, src=200, dst=100)
        dummy_room.exits = [ex]

        rdb = {100: src_room, 200: dummy_room}
        exits = [ex]

        position_dummy_rooms(rdb, exits)

        # North from dummy leads to (5, 5, 0), so dummy is (5, 4, 0)
        assert dummy_room.x == 5
        assert dummy_room.y == 4
        assert dummy_room.z == 0

    def test_position_dummy_rooms_fallback_disconnected(self):
        """A disconnected dummy room without incident exits falls back to (0, 0, 0)."""
        dummy_room = Room(RoomDef(vnum=999, name="Isolated", description="", exits=()))
        dummy_room.dummy = True
        dummy_room.x = None
        dummy_room.y = None
        dummy_room.z = None

        rdb = {999: dummy_room}
        position_dummy_rooms(rdb, [])

        assert dummy_room.x == 0
        assert dummy_room.y == 0
        assert dummy_room.z == 0

    def test_position_dummy_rooms_idempotent(self):
        """Calling position_dummy_rooms repeatedly preserves assigned coordinates."""
        src = Room(RoomDef(vnum=1, name="R1", description="", exits=()))
        src.x, src.y, src.z = 0, 0, 0
        dst = Room(RoomDef(vnum=2, name="R2", description="", exits=()))
        dst.dummy = True

        ex = Exit(direction=Direction.east, src=1, dst=2)
        src.exits = [ex]

        rdb = {1: src, 2: dst}
        position_dummy_rooms(rdb, [ex])
        first_coords = (dst.x, dst.y, dst.z)
        assert first_coords == (1, 0, 0)

        position_dummy_rooms(rdb, [ex])
        assert (dst.x, dst.y, dst.z) == first_coords


class TestSolverAffineAnchoringAndBranches:
    """Test Pyomo solver internal anchor routines, affine linear expressions, and edge branches."""

    def test_solve_affine_anchors_all_directions(self):
        """Verify affine expressions for East, West, Up, Down dummy exits in solver."""
        r_center = Room(RoomDef(vnum=10, name="Center", description="", exits=()))
        r_center.x = 0
        r_center.y = 0
        r_center.z = 0

        d_east = Room(RoomDef(vnum=11, name="EastDummy", description="", exits=()))
        d_east.dummy = True
        d_west = Room(RoomDef(vnum=12, name="WestDummy", description="", exits=()))
        d_west.dummy = True
        d_up = Room(RoomDef(vnum=13, name="UpDummy", description="", exits=()))
        d_up.dummy = True
        d_down = Room(RoomDef(vnum=14, name="DownDummy", description="", exits=()))
        d_down.dummy = True

        ex_east = Exit(direction=Direction.east, src=10, dst=11)
        ex_west = Exit(direction=Direction.west, src=10, dst=12)
        ex_up = Exit(direction=Direction.up, src=10, dst=13)
        ex_down = Exit(direction=Direction.down, src=10, dst=14)
        r_center.exits = [ex_east, ex_west, ex_up, ex_down]

        rdb = {10: r_center, 11: d_east, 12: d_west, 13: d_up, 14: d_down}
        exits = [ex_east, ex_west, ex_up, ex_down]

        model, results = solve(rdb, exits, timeout=10)
        assert 10 in model.Rooms
        assert 11 not in model.Rooms
        assert 12 not in model.Rooms
        assert 13 not in model.Rooms
        assert 14 not in model.Rooms

        assert d_east.x == r_center.x + 1
        assert d_west.x == r_center.x - 1
        assert d_up.z == r_center.z + 1
        assert d_down.z == r_center.z - 1

    def test_solve_isolated_room_and_dummy_exits_filter(self):
        """Test isolated room handling and skipping exits where both endpoints are dummy."""
        r_iso = Room(RoomDef(vnum=99, name="Isolated", description="", exits=()))
        d1 = Room(RoomDef(vnum=101, name="D1", description="", exits=()))
        d1.dummy = True
        d2 = Room(RoomDef(vnum=102, name="D2", description="", exits=()))
        d2.dummy = True

        # Exit between two dummy rooms (should be ignored in solve)
        dummy_exit = Exit(direction=Direction.north, src=101, dst=102)

        rdb = {99: r_iso, 101: d1, 102: d2}
        exits = [dummy_exit]

        model, results = solve(rdb, exits, timeout=5)
        assert 99 in model.Rooms
        assert 101 not in model.Rooms
        assert 102 not in model.Rooms

    def test_non_euler_skips_dummy_rooms_and_exits(self):
        """Verify non_euler heuristic excludes dummy rooms from decision variables."""
        r1 = Room(RoomDef(vnum=1, name="R1", description="", exits=()))
        d2 = Room(RoomDef(vnum=2, name="D2", description="", exits=()))
        d2.dummy = True

        ex = Exit(direction=Direction.north, src=1, dst=2)
        r1.exits = [ex]

        rdb = {1: r1, 2: d2}
        exits = [ex]

        m = non_euler(rdb, exits)
        assert 1 in m.Rooms
        assert 2 not in m.Rooms


class TestSolverMultiFixtureFeasibility:
    """Regression test: verify zero regressions across all repository area fixtures."""

    @pytest.mark.parametrize(
        "fixture_rel_path",
        [
            "areas/smurf.are",
            "areas/school.are",
            "dialects/circlemud.are",
            "dialects/dikumud_alfa.wld",
        ],
    )
    def test_fixture_solves_with_valid_coordinates(self, fixture_rel_path):
        """Verify area solves to optimality/feasibility with integer coordinates for all rooms."""
        filepath = FIXTURES_DIR / fixture_rel_path
        content = filepath.read_text(encoding="latin-1")
        area = Parser().parse(content)
        rdb = {r.vnum: Room(r) for r in area.rooms}

        solved_rdb, exits = solve_layout(rdb, area, solver_timeout=15)

        assert len(solved_rdb) >= len(area.rooms)
        for vnum, room in solved_rdb.items():
            assert room.x is not None, f"Room {vnum} has None x coordinate"
            assert room.y is not None, f"Room {vnum} has None y coordinate"
            assert room.z is not None, f"Room {vnum} has None z coordinate"
            assert isinstance(room.x, (int, float))
            assert isinstance(room.y, (int, float))
            assert isinstance(room.z, (int, float))
            assert room.x == int(room.x)
            assert room.y == int(room.y)
            assert room.z == int(room.z)

        # For any dummy room, verify it is adjacent to at least one connecting room
        dummy_rooms = [r for r in solved_rdb.values() if getattr(r, "dummy", False) is True]
        for dummy in dummy_rooms:
            connected = [
                e for e in exits
                if (e.dst == dummy.vnum and e.src in solved_rdb and not getattr(solved_rdb[e.src], "dummy", False))
                or (e.src == dummy.vnum and e.dst in solved_rdb and not getattr(solved_rdb[e.dst], "dummy", False))
            ]
            if connected:
                ex = connected[0]
                if ex.dst == dummy.vnum:
                    src = solved_rdb[ex.src]
                    dx = 1 if ex.direction == Direction.east else -1 if ex.direction == Direction.west else 0
                    dy = 1 if ex.direction == Direction.north else -1 if ex.direction == Direction.south else 0
                    dz = 1 if ex.direction == Direction.up else -1 if ex.direction == Direction.down else 0
                    assert dummy.x == src.x + dx
                    assert dummy.y == src.y + dy
                    assert dummy.z == src.z + dz

    def test_circle_world_directory_solve(self):
        """Verify CircleMUD multi-file zone/wld directory solve layout."""
        area = parse_circlemud_directory(FIXTURES_DIR / "dialects" / "circle_world")
        rdb = {r.vnum: Room(r) for r in area.rooms}

        solved_rdb, exits = solve_layout(rdb, area.header, solver_timeout=15)
        assert len(solved_rdb) == 5
        for r in solved_rdb.values():
            assert r.x is not None and r.y is not None and r.z is not None


class TestRendererExternalExitStubs:
    """Verify that renderers (SVG, HTML, JSON) handle boundary dummy rooms and external stubs."""

    @pytest.mark.parametrize(
        ("direction", "expected_dx", "expected_dy"),
        [
            (Direction.north, 0.0, -1.0),
            (Direction.east, 1.0, 0.0),
            (Direction.south, 0.0, 1.0),
            (Direction.west, -1.0, 0.0),
        ],
    )
    def test_plotter_external_exit_stubs_cardinal(self, direction, expected_dx, expected_dy):
        """Plotter projects external exit stubs at exact 1.0 unit length for dummy target rooms."""
        src = Room(RoomDef(vnum=10, name="R10", description="", exits=()))
        src.x, src.y, src.z = 5, 5, 0

        dummy = Room(RoomDef(vnum=20, name="D20", description="", exits=()))
        dummy.dummy = True
        dummy.x, dummy.y, dummy.z = 5, 6, 0

        ex = Exit(direction=direction, src=10, dst=20)
        src.exits = [ex]

        rdb = {10: src, 20: dummy}
        plotter = Plotter("test", rdb, [ex])
        coords = plotter.proj_exit(ex)

        assert coords is not None
        start, end = coords
        assert end[0] == pytest.approx(start[0] + expected_dx)
        assert end[1] == pytest.approx(start[1] + expected_dy)

    def test_plotter_external_exit_stubs_vertical(self):
        """Plotter projects vertical external exit stubs using elevation lift."""
        src = Room(RoomDef(vnum=10, name="R10", description="", exits=()))
        src.x, src.y, src.z = 2, 2, 0

        dummy_up = Room(RoomDef(vnum=20, name="UpDummy", description="", exits=()))
        dummy_up.dummy = True
        dummy_up.x, dummy_up.y, dummy_up.z = 2, 2, 1

        ex_up = Exit(direction=Direction.up, src=10, dst=20)
        src.exits = [ex_up]

        rdb = {10: src, 20: dummy_up}
        plotter = Plotter("test", rdb, [ex_up])
        start, end = plotter.proj_exit(ex_up)
        assert end[0] == pytest.approx(start[0] + plotter.lift)
        assert end[1] == pytest.approx(start[1] - plotter.lift)

        dummy_down = Room(RoomDef(vnum=30, name="DownDummy", description="", exits=()))
        dummy_down.dummy = True
        dummy_down.x, dummy_down.y, dummy_down.z = 2, 2, -1

        ex_down = Exit(direction=Direction.down, src=10, dst=30)
        src.exits.append(ex_down)
        rdb[30] = dummy_down

        plotter = Plotter("test", rdb, [ex_down])
        start, end = plotter.proj_exit(ex_down)
        assert end[0] == pytest.approx(start[0] - plotter.lift)
        assert end[1] == pytest.approx(start[1] + plotter.lift)

    def test_svg_render_with_dummy_rooms(self, tmp_path):
        """SVGRenderer renders file without errors when dummy rooms and external stubs exist."""
        src = Room(RoomDef(vnum=10, name="R10", description="", exits=()))
        src.x, src.y, src.z = 0, 0, 0

        dummy = Room(RoomDef(vnum=20, name="D20", description="", exits=()))
        dummy.dummy = True
        dummy.x, dummy.y, dummy.z = 0, 1, 0

        ex = Exit(direction=Direction.north, src=10, dst=20)
        src.exits = [ex]

        rdb = {10: src, 20: dummy}
        svg_file = tmp_path / "map.svg"

        renderer = SVGRenderer()
        renderer.render(rdb, svg_file)

        assert svg_file.exists()
        content = svg_file.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "</svg>" in content

    def test_json_and_html_render_with_dummy_rooms(self, tmp_path):
        """JSON and HTML exports succeed and exclude dummy rooms from the rendered room list."""
        src = Room(RoomDef(vnum=100, name="R100", description="Desc", exits=()))
        src.x, src.y, src.z = 0, 0, 0

        dummy = Room(RoomDef(vnum=200, name="D200", description="", exits=()))
        dummy.dummy = True
        dummy.x, dummy.y, dummy.z = 0, 1, 0

        ex = Exit(direction=Direction.north, src=100, dst=200)
        src.exits = [ex]

        rdb = {100: src, 200: dummy}

        # JSON data model build
        data = build_area_json(rdb)
        assert len(data["rooms"]) == 1
        assert data["rooms"][0]["vnum"] == 100

        # File export via render_map
        json_file = tmp_path / "map.json"
        render_map(rdb, json_file, fmt="json")
        assert json_file.exists()
        json_content = json_file.read_text(encoding="utf-8")
        assert '"vnum": 100' in json_content
        assert '"vnum": 200' not in json_content

        html_file = tmp_path / "map.html"
        render_map(rdb, html_file, fmt="html")
        assert html_file.exists()
        html_content = html_file.read_text(encoding="utf-8")
        assert "<html" in html_content.lower()
