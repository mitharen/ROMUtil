import io
import os
import pytest
import xml.etree.ElementTree as ET

from romutil.models import Direction, Room, Exit, RoomDef, ExitDef, AreaHeader
from romutil.plotter import Plotter
from romutil.solver import non_euler, solve
from romutil.graph import restore_rooms, mfas, graph
from romutil.cli import main, cli

_CANDIDATE_AREAS_DIRS = [
    os.environ.get('QUICKMUD_AREA_DIR', ''),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../QuickMUD/area')),
    '/home/user/proj/QuickMUD/area',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../proj/QuickMUD/area')),
]
SAMPLE_AREAS_DIR = next((d for d in _CANDIDATE_AREAS_DIRS if d and os.path.isdir(d)), _CANDIDATE_AREAS_DIRS[1])


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
        r_def = RoomDef(
            vnum=100,
            name="Temple",
            description="A grand temple.",
            exits=(
                ExitDef(direction=0, dst_vnum=101),
                ExitDef(direction=1, dst_vnum=102),
            ),
        )
        room = Room(r_def)
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

        # Keyword argument initialization
        room_kw = Room(
            vnum=100,
            name="Temple",
            desc="A grand temple.",
            exits=[
                ExitDef(direction=0, dst_vnum=101),
                ExitDef(direction=1, dst_vnum=102),
            ],
        )
        assert room_kw.vnum == 100
        assert len(room_kw.exits) == 2

        # Raw tuple/list must raise TypeError
        with pytest.raises(TypeError):
            Room((100, "Temple", "A grand temple.", [(0, 101), (1, 102)]))  # type: ignore

    def test_room_initialization_no_exits(self):
        r_def = RoomDef(vnum=200, name="Void", description="Nothing here.", exits=())
        room = Room(r_def)
        assert room.vnum == 200
        assert room.exits == []

    def test_room_replace_exit(self):
        r_def = RoomDef(
            vnum=100,
            name="Hall",
            description="A hall.",
            exits=(
                ExitDef(direction=0, dst_vnum=101),
                ExitDef(direction=1, dst_vnum=102),
            ),
        )
        room = Room(r_def)
        room.replace_exit(101, 105, distance=3)
        assert room.exits[0].dst == 105
        assert room.exits[0].distance == 4  # 1 + 3
        # Exit to 102 should remain untouched
        assert room.exits[1].dst == 102
        assert room.exits[1].distance == 1

    def test_exit_bidirectional_equality_and_hash(self):
        # Exit from 100 to 101 heading North
        e1 = Exit(ExitDef(direction=0, dst_vnum=101), source=100)
        # Symmetrical exit from 101 to 100 heading South
        e2 = Exit(ExitDef(direction=2, dst_vnum=100), source=101)
        # Exit in different direction
        e3 = Exit(ExitDef(direction=1, dst_vnum=101), source=100)
        # Exit to different destination
        e4 = Exit(ExitDef(direction=0, dst_vnum=102), source=100)

        assert e1 == e2
        assert hash(e1) == hash(e2)
        assert e1 != e3
        assert e1 != e4
        assert repr(e1) == "100 -> 101 (1 north)"

        # Raw tuple must raise TypeError
        with pytest.raises(TypeError):
            Exit((0, 101), source=100)  # type: ignore

    def test_exit_contains_room(self):
        ex = Exit(ExitDef(direction=0, dst_vnum=101), source=100)
        assert 100 in ex
        assert 101 in ex
        assert 999 not in ex


class TestRestoreRooms:
    """Tests for reconstructing collapsed hallway rooms."""

    def test_restore_rooms_all_directions(self):
        parent = Room(RoomDef(vnum=100, name="Parent", description="Parent room", exits=()))
        parent.x, parent.y, parent.z = 10, 10, 10

        r_n = Room(RoomDef(vnum=101, name="North Room", description="", exits=()))
        r_e = Room(RoomDef(vnum=102, name="East Room", description="", exits=()))
        r_s = Room(RoomDef(vnum=103, name="South Room", description="", exits=()))
        r_w = Room(RoomDef(vnum=104, name="West Room", description="", exits=()))
        r_u = Room(RoomDef(vnum=105, name="Up Room", description="", exits=()))
        r_d = Room(RoomDef(vnum=106, name="Down Room", description="", exits=()))

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
        r1 = Room(RoomDef(vnum=1, name="Room 1", description="Desc 1", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room(RoomDef(vnum=2, name="Room 2", description="Desc 2", exits=(ExitDef(direction=3, dst_vnum=1),)))
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
        r1 = Room(RoomDef(vnum=1, name="Room 1", description="Desc 1", exits=()))
        r1.x, r1.y, r1.z = 0, 0, 0
        plotter = Plotter("dummy.svg", {1: r1}, [])
        plotter.x_max = 0
        plotter.y_max = 0
        plotter.z_max = 0

        for d in [Direction.north, Direction.east, Direction.south, Direction.west, Direction.up, Direction.down]:
            ex = Exit(ExitDef(direction=d.value, dst_vnum=999), source=1)
            pe = plotter.proj_exit(ex)
            assert pe is not None
            assert len(pe) == 2

    def test_plotter_projection_with_none_coords(self):
        r1 = Room(RoomDef(vnum=1, name="Room 1", description="Desc", exits=()))
        r1.x, r1.y, r1.z = None, 0, 0
        plotter = Plotter("dummy.svg", {1: r1}, [])
        assert plotter.proj_room(r1) is None

    def test_plotter_plot_generates_valid_svg(self, tmp_path):
        out_svg = str(tmp_path / "map.svg")
        r1 = Room(RoomDef(vnum=1, name="Start", description="Starting Room\nSecond Line", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room(RoomDef(vnum=2, name="End", description="Ending Room", exits=(ExitDef(direction=3, dst_vnum=1),)))
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
        r1 = Room(RoomDef(vnum=1, name="West Room", description="", exits=(ExitDef(direction=1, dst_vnum=2),)))       # east to 2
        r2 = Room(RoomDef(vnum=2, name="Corridor Room", description="", exits=(ExitDef(direction=3, dst_vnum=1), ExitDef(direction=1, dst_vnum=3)))) # west to 1, east to 3
        r3 = Room(RoomDef(vnum=3, name="East Room", description="", exits=(ExitDef(direction=3, dst_vnum=2),)))       # west to 2
        rdb = {1: r1, 2: r2, 3: r3}

        out_svg = str(tmp_path / "corridor.svg")
        graph(rdb, out_svg, AreaHeader(filename="test.are", name="Test Corridor", builder="", vnum_min=1, vnum_max=3))

        # After graph() solve, room 2 was collapsed and restored, all 3 rooms exist with coordinates
        assert 1 in rdb
        assert 2 in rdb
        assert 3 in rdb
        assert r1.x < r2.x < r3.x
        assert os.path.exists(out_svg)

    def test_disconnected_room_handled(self, caplog):
        # Room with no exits
        r1 = Room(RoomDef(vnum=1, name="Isolated", description="No exits", exits=()))
        graph({1: r1}, "dummy.svg", AreaHeader(filename="test.are", name="Test", builder="", vnum_min=1, vnum_max=1))
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
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc 1", exits=(ExitDef(direction=1, dst_vnum=2),)))  # East to 2
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc 2", exits=(ExitDef(direction=3, dst_vnum=1),)))  # West to 1
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
        r1 = Room(RoomDef(vnum=10, name="Ground", description="Floor 1", exits=(ExitDef(direction=4, dst_vnum=20),)))  # Up to 20
        r2 = Room(RoomDef(vnum=20, name="Tower", description="Floor 2", exits=(ExitDef(direction=5, dst_vnum=10),)))   # Down to 10
        rdb = {10: r1, 20: r2}
        exits = [r1.exits[0], r2.exits[0]]

        model, result = solve(rdb, exits)
        assert result.solver.termination_condition.name in ("optimal", "feasible")
    def test_solve_one_way_and_no_exit_room(self):
        # Room 1 has a one-way exit east to Room 2
        # Room 2 has no exits
        # Room 3 is isolated (no exits)
        r1 = Room(RoomDef(vnum=1, name="R1", description="Desc 1", exits=(ExitDef(direction=1, dst_vnum=2),)))
        r2 = Room(RoomDef(vnum=2, name="R2", description="Desc 2", exits=()))
        r3 = Room(RoomDef(vnum=3, name="Isolated", description="Desc 3", exits=()))
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
        r1 = Room(RoomDef(vnum=10, name="Incomplete", description="Desc", exits=(ExitDef(direction=0, dst_vnum=-1),)))
        rdb = {10: r1}
        out_svg = str(tmp_path / "minus_one.svg")
        graph(rdb, out_svg, AreaHeader(filename="test.are", name="Test Incomplete", builder="", vnum_min=10, vnum_max=10))
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
        monkeypatch.setattr("sys.argv", ["romutil", smurf_file, "-outbase", outbase, "-d"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

    def test_cli_default_outbase(self, tmp_path, monkeypatch):
        fake_area = tmp_path / "dummy.are"
        fake_area.write_text("#AREA\ndummy.are~\nDummy~\nBuilder~\n1 10\n#$\n")
        monkeypatch.setattr("sys.argv", ["romutil", str(fake_area)])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

    def test_cli_module_run(self, tmp_path, monkeypatch):
        import sys
        import runpy
        smurf_file = os.path.join(SAMPLE_AREAS_DIR, "smurf.are")
        if not os.path.exists(smurf_file):
            pytest.skip("QuickMUD area files not found")

        outbase = str(tmp_path / "cli_module_smurf")
        monkeypatch.setattr("sys.argv", ["romutil", smurf_file, "-outbase", outbase])
        monkeypatch.delitem(sys.modules, "romutil.cli", raising=False)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("romutil.cli", run_name="__main__")
        assert exc.value.code == 0
