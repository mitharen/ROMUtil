import io
import os
import pytest
import xml.etree.ElementTree as ET

from Mapper import Direction, Room, Exit, Plotter, restore_rooms, mfas, non_euler, solve, graph, main
import AreaParser

_sample_candidates = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../QuickMUD/area')),
    '/home/user/proj/QuickMUD/area',
]
SAMPLE_AREAS_DIR = next((p for p in _sample_candidates if os.path.isdir(p)), _sample_candidates[0])


class TestDirection:
    """Tests for Direction enum and inversion logic."""

    def test_direction_values(self):
        assert Direction.north == 0
        assert Direction.east == 1
        assert Direction.up == 2
        assert Direction.south == 3
        assert Direction.west == 4
        assert Direction.down == 5

    def test_direction_inversion(self):
        assert Direction.north.invert() == Direction.south
        assert Direction.south.invert() == Direction.north
        assert Direction.east.invert() == Direction.west
        assert Direction.west.invert() == Direction.east
        assert Direction.up.invert() == Direction.down
        assert Direction.down.invert() == Direction.up


class TestRoomAndExitModels:
    """Tests for Room and Exit domain models."""

    def test_room_initialization_with_exits(self):
        # Raw room format: (vnum, name, desc, [(dir_num, dst_vnum), ...])
        raw_room = (100, "Temple", "A grand temple.", [(0, 101), (1, 102)])
        room = Room(raw_room)
        assert room.vnum == 100
        assert room.name == "Temple"
        assert room.desc == "A grand temple."
        assert len(room.exits) == 2
        assert room.exits[0].src == 100
        assert room.exits[0].dst == 101
        assert room.exits[0].direction == Direction.north
        assert room.exits[1].src == 100
        assert room.exits[1].dst == 102
        assert room.exits[1].direction == Direction.east
        assert not room.dummy
        assert f"100: Temple" in repr(room)

    def test_room_initialization_no_exits(self):
        raw_room = (200, "Void", "Nothing here.", None)
        room = Room(raw_room)
        assert room.vnum == 200
        assert room.exits == []

    def test_room_replace_exit(self):
        raw_room = (100, "Hall", "A hall.", [(0, 101), (1, 102)])
        room = Room(raw_room)
        room.replace_exit(101, 105, distance=3)
        assert room.exits[0].dst == 105
        assert room.exits[0].distance == 4  # 1 + 3
        # Exit to 102 should remain untouched
        assert room.exits[1].dst == 102
        assert room.exits[1].distance == 1

    def test_exit_bidirectional_equality_and_hash(self):
        # Exit from 100 to 101 heading North
        e1 = Exit((0, 101), 100)
        # Symmetrical exit from 101 to 100 heading South
        e2 = Exit((2, 100), 101)
        # Exit in different direction
        e3 = Exit((1, 101), 100)
        # Exit to different destination
        e4 = Exit((0, 102), 100)

        assert e1 == e2
        assert hash(e1) == hash(e2)
        assert e1 != e3
        assert e1 != e4
        assert repr(e1) == "100 -> 101 (1 north)"

    def test_exit_contains_room(self):
        ex = Exit((0, 101), 100)
        assert 100 in ex
        assert 101 in ex
        assert 999 not in ex


class TestRestoreRooms:
    """Tests for reconstructing collapsed hallway rooms."""

    def test_restore_rooms_all_directions(self):
        parent = Room((100, "Parent", "Parent room", []))
        parent.x, parent.y, parent.z = 10, 10, 10

        r_n = Room((101, "North Room", "", []))
        r_e = Room((102, "East Room", "", []))
        r_s = Room((103, "South Room", "", []))
        r_w = Room((104, "West Room", "", []))
        r_u = Room((105, "Up Room", "", []))
        r_d = Room((106, "Down Room", "", []))

        parent.fixups = [
            (r_n, Direction.north, 2),
            (r_e, Direction.east, 3),
            (r_s, Direction.south, 1),
            (r_w, Direction.west, 4),
            (r_u, Direction.up, 5),
            (r_d, Direction.down, 2),
        ]

        restored = restore_rooms(parent)
        assert len(restored) == 6
        assert (r_n.x, r_n.y, r_n.z) == (10, 12, 10)
        assert (r_e.x, r_e.y, r_e.z) == (13, 10, 10)
        assert (r_s.x, r_s.y, r_s.z) == (10, 9, 10)
        assert (r_w.x, r_w.y, r_w.z) == (6, 10, 10)
        assert (r_u.x, r_u.y, r_u.z) == (10, 10, 15)
        assert (r_d.x, r_d.y, r_d.z) == (10, 10, 8)


class TestPlotter:
    """Tests for SVG map generation and coordinate projections."""

    def test_plotter_projections(self):
        r1 = Room((1, "Room 1", "Desc 1", [(1, 2)]))
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room((2, "Room 2", "Desc 2", [(3, 1)]))
        r2.x, r2.y, r2.z = 1, 0, 0
        ex = r1.exits[0]

        rdb = {1: r1, 2: r2}
        plotter = Plotter("dummy.svg", rdb, [ex])
        plotter.x_max = 1
        plotter.y_max = 0
        plotter.z_max = 0

        # proj_room
        pr = plotter.proj_room(r1)
        assert pr is not None
        assert len(pr) == 2

        # proj_exit (internal exit)
        pe = plotter.proj_exit(ex)
        assert pe is not None
        assert len(pe) == 2

    def test_plotter_proj_exit_external_directions(self):
        r1 = Room((1, "Room 1", "Desc 1", []))
        r1.x, r1.y, r1.z = 0, 0, 0
        plotter = Plotter("dummy.svg", {1: r1}, [])
        plotter.x_max = 0
        plotter.y_max = 0
        plotter.z_max = 0

        for d in [Direction.north, Direction.east, Direction.south, Direction.west, Direction.up, Direction.down]:
            ex = Exit((d.value, 999), 1)
            pe = plotter.proj_exit(ex)
            assert pe is not None
            assert len(pe) == 2

    def test_plotter_projection_with_none_coords(self):
        r1 = Room((1, "Room 1", "Desc", []))
        r1.x, r1.y, r1.z = None, 0, 0
        plotter = Plotter("dummy.svg", {1: r1}, [])
        assert plotter.proj_room(r1) is None

    def test_plotter_plot_generates_valid_svg(self, tmp_path):
        out_svg = str(tmp_path / "map.svg")
        r1 = Room((1, "Start", "Starting Room\nSecond Line", [(1, 2)]))
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room((2, "End", "Ending Room", [(3, 1)]))
        r2.x, r2.y, r2.z = 2, 0, 0
        ex = r1.exits[0]

        plotter = Plotter(out_svg, {1: r1, 2: r2}, [ex])
        plotter.plot()

        assert os.path.exists(out_svg)
        assert os.path.getsize(out_svg) > 0

        # Verify XML/SVG parse
        tree = ET.parse(out_svg)
        root = tree.getroot()
        assert "svg" in root.tag.lower()


class TestGraphAndCorridorCollapse:
    """Tests for hallway condensation and dummy room injection."""

    def test_corridor_collapsing(self, tmp_path):
        # 3 rooms in a straight line: 1 <-> 2 <-> 3
        # Room 2 has exactly 2 opposite exits (west to 1, east to 3)
        r1 = Room((1, "West Room", "", [(1, 2)]))       # east to 2
        r2 = Room((2, "Corridor Room", "", [(3, 1), (1, 3)])) # west to 1, east to 3
        r3 = Room((3, "East Room", "", [(3, 2)]))       # west to 2
        rdb = {1: r1, 2: r2, 3: r3}

        out_svg = str(tmp_path / "corridor.svg")
        graph(rdb, out_svg, ("test.are", "Test Corridor", "", (1, 3)))

        # After graph() solve, room 2 was collapsed and restored, all 3 rooms exist with coordinates
        assert 1 in rdb
        assert 2 in rdb
        assert 3 in rdb
        assert r1.x < r2.x < r3.x
        assert os.path.exists(out_svg)

    def test_disconnected_room_handled(self, caplog):
        # Room with no exits
        r1 = Room((1, "Isolated", "No exits", []))
        graph({1: r1}, "dummy.svg", ("test.are", "Test", "", (1, 1)))
        assert "Ignoring disconnected room" in caplog.text


class TestSolver:
    """Tests for MILP optimization solver and feedback arc set."""

    def test_mfas_acyclic_and_cyclic(self):
        # Cyclic graph: 1 -> 2 -> 3 -> 1
        edges = [(1, 2), (2, 3), (3, 1)]
        cuts = mfas(edges)
        # Should cut at least 1 edge to break cycle
        assert len(cuts) >= 1
        assert any(e in edges for e in cuts)

    def test_solve_simple_layout(self):
        # 2 rooms connected bidirectionally
        r1 = Room((1, "R1", "Desc 1", [(1, 2)]))  # East to 2
        r2 = Room((2, "R2", "Desc 2", [(3, 1)]))  # West to 1
        rdb = {1: r1, 2: r2}
        exits = [r1.exits[0], r2.exits[0]]

        model, result = solve(rdb, exits)
        assert result.solver.termination_condition.name in ("optimal", "feasible")
        # R2 must be to the east of R1
        assert model.x[2].value > model.x[1].value
        assert model.y[2].value == model.y[1].value
        assert model.z[2].value == model.z[1].value

    def test_solve_vertical_levels(self):
        # 2 rooms connected Up / Down
        r1 = Room((10, "Ground", "Floor 1", [(4, 20)]))  # Up to 20
        r2 = Room((20, "Tower", "Floor 2", [(5, 10)]))   # Down to 10
        rdb = {10: r1, 20: r2}
        exits = [r1.exits[0], r2.exits[0]]

        model, result = solve(rdb, exits)
        assert result.solver.termination_condition.name in ("optimal", "feasible")
    def test_solve_one_way_and_no_exit_room(self):
        # Room 1 has a one-way exit east to Room 2
        # Room 2 has no exits
        # Room 3 is isolated (no exits)
        r1 = Room((1, "R1", "Desc 1", [(1, 2)]))
        r2 = Room((2, "R2", "Desc 2", []))
        r3 = Room((3, "Isolated", "Desc 3", []))
        rdb = {1: r1, 2: r2, 3: r3}
        exits = [r1.exits[0]]

        model, result = solve(rdb, exits)
        assert result.solver.termination_condition.name in ("optimal", "feasible")
        assert model.x[1].value is not None
        assert model.x[2].value is not None


class TestMapperIntegration:
    """Integration tests for Mapper.main and CLI flows."""

    def test_graph_undefined_destination_minus_one(self, tmp_path):
        # Room has exit pointing to -1 (incomplete area)
        r1 = Room((10, "Incomplete", "Desc", [(0, -1)]))
        rdb = {10: r1}
        out_svg = str(tmp_path / "minus_one.svg")
        graph(rdb, out_svg, ("test.are", "Test Incomplete", "", (10, 10)))
        assert os.path.exists(out_svg)

    def test_mapper_main_single_area(self, tmp_path):
        smurf_file = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        if not os.path.exists(smurf_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "smurf_out")
        with open(smurf_file, 'r', encoding='latin-1') as f:
            with pytest.raises(SystemExit) as exc_info:
                main([f], outbase)
            assert exc_info.value.code == 0

        # Verify at least one SVG was generated
        svgs = list(tmp_path.glob("smurf_out*.svg"))
        assert len(svgs) >= 1
        assert svgs[0].stat().st_size > 0

    def test_mapper_main_no_rooms_negative(self, tmp_path, caplog):
        # social.are has no #ROOMS section
        social_file = os.path.join(SAMPLE_AREAS_DIR, "social.are")
        if not os.path.exists(social_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "social_out")
        with open(social_file, 'r', encoding='latin-1') as f:
            with pytest.raises(SystemExit) as exc_info:
                main([f], outbase)
            assert exc_info.value.code == 0
        assert "No rooms to plot" in caplog.text

    def test_mapper_main_corrupt_file_negative(self, tmp_path, caplog):
        corrupt_content = io.StringIO("INVALID CONTENT WITHOUT ROOMS OR HEADER\n")
        outbase = str(tmp_path / "corrupt_out")
        with pytest.raises(SystemExit) as exc_info:
            main([corrupt_content], outbase)
        assert exc_info.value.code == 0
        assert "No rooms to plot" in caplog.text

    def test_cli_execution(self, tmp_path, monkeypatch):
        smurf_file = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        if not os.path.exists(smurf_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "cli_smurf")
        monkeypatch.setattr("sys.argv", ["Mapper.py", smurf_file, "-outbase", outbase, "-d"])
        import runpy
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("Mapper", run_name="__main__")
        assert exc.value.code == 0
