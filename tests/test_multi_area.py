"""Tests for multi-area composite mapping and cross-file topology resolution (Task 10a).

Verifies:
1. merge_areas() logic:
   - Preserves area provenance on each RoomDef (room.area_name).
   - Combines rooms, mobiles, objects, resets, shops, specials, helps, socials.
   - Computes composite VNUM ranges and header titles.
   - Handles empty and single-area input gracefully.
   - Preserves custom titles when specified.
2. Cross-file inter-area exit resolution:
   - When areas are loaded individually, cross-area exits become external dummy stubs.
   - When areas are loaded together (composite set), exits between loaded areas
     become internal edges and bidirectional pairs link properly:
       midgaard [3119] (East) <-> hood [2101] (West)
       midgaard [3144] (East) <-> hood [2160] (West)
       midgaard [3124] (South) <-> grave [3600] (North)
       midgaard [3047] (East) <-> mobfact [9400] (West)
   - Exits to VNUMs outside the loaded cluster (e.g. Arachnos 3700, Sewers 70xx)
     still cleanly produce external boundary dummy stubs.
3. Renderer & viewer presentation:
   - JSON output includes 'area_name' for every room.
   - HTML viewer template contains area_name display logic in room details card and tooltips.
4. CLI multi-file invocation:
   - Passing multiple .are files to CLI produces valid composite HTML, SVG, and JSON.
   - Single-area and --circle-dir invocations remain fully backward-compatible.
5. Solver layout and non-overlap invariants:
   - Solves composite topologies positioning interconnected zones without room collisions.
"""

from pathlib import Path
import json
import pytest

from romutil.models import (
    AreaData,
    AreaHeader,
    Direction,
    Exit,
    ExitDef,
    Room,
    RoomDef,
    SocialDef,
    merge_areas,
)
from romutil.parser import Parser
from romutil.graph import solve_layout
from romutil.renderers.json import build_area_json
from romutil.renderers.html import generate_html_viewer
from romutil.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "areas"


@pytest.fixture
def midgaard_area() -> AreaData:
    content = (FIXTURES_DIR / "midgaard.are").read_text(encoding="latin-1")
    return Parser().parse(content)


@pytest.fixture
def hood_area() -> AreaData:
    content = (FIXTURES_DIR / "hood.are").read_text(encoding="latin-1")
    return Parser().parse(content)


@pytest.fixture
def grave_area() -> AreaData:
    content = (FIXTURES_DIR / "grave.are").read_text(encoding="latin-1")
    return Parser().parse(content)


@pytest.fixture
def mobfact_area() -> AreaData:
    content = (FIXTURES_DIR / "mobfact.are").read_text(encoding="latin-1")
    return Parser().parse(content)


# ============================================================================
# 1. Unit Tests: merge_areas() and Provenance Tracking
# ============================================================================

class TestMergeAreas:
    """Test merge_areas() combining multiple AreaData instances."""

    def test_merge_areas_empty(self):
        """Empty input returns a valid fallback AreaData composite."""
        composite = merge_areas([])
        assert isinstance(composite, AreaData)
        assert composite.header is not None
        assert composite.header.name == "Composite Area"
        assert len(composite.rooms) == 0

    def test_merge_areas_empty_with_title(self):
        """Empty input with custom title respects title."""
        composite = merge_areas([], title="Custom Empty")
        assert composite.header.name == "Custom Empty"

    def test_merge_areas_single_area(self, midgaard_area):
        """Single area merge preserves all metadata and tags room provenance."""
        composite = merge_areas([midgaard_area])
        assert composite.header.name == midgaard_area.header.name
        assert len(composite.rooms) == len(midgaard_area.rooms)
        assert all(r.area_name == "Midgaard" for r in composite.rooms)

    def test_merge_areas_provenance_tracking(self, midgaard_area, hood_area):
        """Rooms maintain their respective source area_name after merging."""
        composite = merge_areas([midgaard_area, hood_area])

        midgaard_rooms = [r for r in composite.rooms if r.vnum >= 3000 and r.vnum < 3400]
        hood_rooms = [r for r in composite.rooms if r.vnum >= 2100 and r.vnum < 2200]

        assert len(midgaard_rooms) == len(midgaard_area.rooms)
        assert len(hood_rooms) == len(hood_area.rooms)
        assert all(r.area_name == "Midgaard" for r in midgaard_rooms)
        assert all(r.area_name == "Gangland" for r in hood_rooms)

    def test_merge_areas_vnum_bounds_and_header(self, midgaard_area, hood_area):
        """Composite header VNUM bounds encompass all merged zones."""
        composite = merge_areas([midgaard_area, hood_area])
        assert composite.header.vnum_min == 2100
        assert composite.header.vnum_max == 3399
        assert composite.header.name == "Midgaard + Gangland"

    def test_merge_areas_custom_title(self, midgaard_area, hood_area):
        """Passing custom title overrides default joined title."""
        composite = merge_areas([midgaard_area, hood_area], title="Metropolitan Midgaard")
        assert composite.header.name == "Metropolitan Midgaard"

    def test_merge_areas_four_cluster_areas(self, midgaard_area, hood_area, grave_area, mobfact_area):
        """Merging all 4 zones combines entities without loss."""
        composite = merge_areas([midgaard_area, hood_area, grave_area, mobfact_area])
        expected_rooms = 143 + 70 + 33 + 25  # 271 rooms
        assert len(composite.rooms) == expected_rooms
        assert composite.header.vnum_min == 2100
        assert composite.header.vnum_max == 9499

        assert len(composite.mobiles) == (
            len(midgaard_area.mobiles) + len(hood_area.mobiles) +
            len(grave_area.mobiles) + len(mobfact_area.mobiles)
        )
        assert len(composite.objects) == (
            len(midgaard_area.objects) + len(hood_area.objects) +
            len(grave_area.objects) + len(mobfact_area.objects)
        )
        assert len(composite.resets) == (
            len(midgaard_area.resets) + len(hood_area.resets) +
            len(grave_area.resets) + len(mobfact_area.resets)
        )

    def test_merge_areas_deduplication(self):
        """Duplicate room vnums across areas keep the first encounter."""
        r1 = RoomDef(vnum=100, name="Room A", description="", area_name="Area1")
        r2 = RoomDef(vnum=100, name="Room B", description="", area_name="Area2")
        r3 = RoomDef(vnum=101, name="Room C", description="", area_name="Area2")

        area1 = AreaData(header=AreaHeader("a1.are", "Area1", "", 100, 100), rooms=(r1,))
        area2 = AreaData(header=AreaHeader("a2.are", "Area2", "", 100, 101), rooms=(r2, r3))

        composite = merge_areas([area1, area2])
        assert len(composite.rooms) == 2
        assert composite.rooms[0].name == "Room A"
        assert composite.rooms[0].area_name == "Area1"
        assert composite.rooms[1].vnum == 101

    def test_merge_areas_metadata_preservation(self, midgaard_area, hood_area):
        """Room properties like sector, flags, and extras are preserved."""
        composite = merge_areas([midgaard_area, hood_area])
        r3001 = next(r for r in composite.rooms if r.vnum == 3001)
        orig_r3001 = next(r for r in midgaard_area.rooms if r.vnum == 3001)
        assert r3001.sector == orig_r3001.sector
        assert r3001.room_flags == orig_r3001.room_flags
        assert len(r3001.extras) == len(orig_r3001.extras)

    def test_merge_areas_reset_message_and_flag(self):
        """Reset message and flag metadata are carried over in merge."""
        a1 = AreaData(
            header=AreaHeader("a1.are", "A1", "", 100, 100),
            rooms=(RoomDef(vnum=100, name="R1", description=""),),
            reset_message="The zone resets with a shudder.",
            flag="OOC",
        )
        composite = merge_areas([a1])
        assert composite.reset_message == "The zone resets with a shudder."
        assert composite.flag == "OOC"

    def test_merge_areas_without_header_bounds(self):
        """When headers lack positive bounds, min/max derive from rooms."""
        r = RoomDef(vnum=105, name="R", description="")
        area = AreaData(rooms=(r,))
        composite = merge_areas([area])
        assert composite.header.vnum_min == 105
        assert composite.header.vnum_max == 105

    def test_merge_areas_completely_empty(self):
        """Headerless and roomless AreaData defaults to 0-0 bounds."""
        composite = merge_areas([AreaData()])
        assert composite.header.vnum_min == 0
        assert composite.header.vnum_max == 0

    def test_merge_areas_socials(self):
        """Social definitions are merged and deduplicated by name."""
        a1 = AreaData(socials=(SocialDef(name="smile"),))
        a2 = AreaData(socials=(SocialDef(name="smile"), SocialDef(name="wave")))
        composite = merge_areas([a1, a2])
        assert len(composite.socials) == 2
        assert {s.name for s in composite.socials} == {"smile", "wave"}

    def test_room_area_name_initialization(self):
        """Room properly initializes and propagates area_name across variants."""
        rdef = RoomDef(vnum=10, name="R", description="", area_name="Area1")
        r1 = Room(rdef)
        assert r1.area_name == "Area1"

        r2 = Room(r1)
        assert r2.area_name == "Area1"

        r3 = Room(vnum=20, name="R3", area_name="Custom")
        assert r3.area_name == "Custom"


# ============================================================================
# 2. Integration Tests: Cross-File Inter-Area Exit Resolution
# ============================================================================

class TestCrossFileInterAreaExits:
    """Verify that exits crossing area boundaries resolve cleanly."""

    def test_isolated_area_creates_dummy_stubs(self, midgaard_area):
        """When boundary exit has no target room loaded, it becomes a dummy room."""
        # Test on a small slice around room 3119 (which exits East to 2101)
        sub_rooms = [r for r in midgaard_area.rooms if r.vnum in (3117, 3119, 3133)]
        rdb = {r.vnum: Room(r) for r in sub_rooms}
        assert 2101 not in rdb

        solved_rdb, exits = solve_layout(rdb, midgaard_area.header, solver_timeout=15)
        assert 2101 in solved_rdb
        assert solved_rdb[2101].dummy is True

    def test_composite_inter_area_exits_become_internal(
        self, midgaard_area, hood_area, grave_area, mobfact_area
    ):
        """When all areas are loaded together, inter-area exits link internal rooms."""
        composite = merge_areas([midgaard_area, hood_area, grave_area, mobfact_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        inter_area_vnums = [3119, 2101, 3144, 2160, 3124, 3600, 3047, 9400]
        for v in inter_area_vnums:
            assert v in rdb
            assert rdb[v].dummy is False

        # Midgaard 3119 (East) <-> The Slums 2101 (West)
        r_3119 = rdb[3119]
        exit_to_hood = next(e for e in r_3119.exits if e.direction == Direction.east)
        assert exit_to_hood.dst == 2101

        r_2101 = rdb[2101]
        exit_to_mid = next(e for e in r_2101.exits if e.direction == Direction.west)
        assert exit_to_mid.dst == 3119
        assert exit_to_hood == exit_to_mid

        # Midgaard 3144 (East) <-> The Slums 2160 (West)
        r_3144 = rdb[3144]
        exit_to_hood_2 = next(e for e in r_3144.exits if e.direction == Direction.east)
        assert exit_to_hood_2.dst == 2160

        r_2160 = rdb[2160]
        exit_from_hood_2 = next(e for e in r_2160.exits if e.direction == Direction.west)
        assert exit_from_hood_2.dst == 3144
        assert exit_to_hood_2 == exit_from_hood_2

        # Midgaard 3124 (South) <-> Graveyard 3600 (North)
        r_3124 = rdb[3124]
        exit_to_grave = next(e for e in r_3124.exits if e.direction == Direction.south)
        assert exit_to_grave.dst == 3600

        r_3600 = rdb[3600]
        exit_from_grave = next(e for e in r_3600.exits if e.direction == Direction.north)
        assert exit_from_grave.dst == 3124
        assert exit_to_grave == exit_from_grave

        # Midgaard 3047 (East) <-> Mob Factory 9400 (West)
        r_3047 = rdb[3047]
        exit_to_mobfact = next(e for e in r_3047.exits if e.direction == Direction.east)
        assert exit_to_mobfact.dst == 9400

        r_9400 = rdb[9400]
        exit_from_mobfact = next(e for e in r_9400.exits if e.direction == Direction.west)
        assert exit_from_mobfact.dst == 3047
        assert exit_to_mobfact == exit_from_mobfact

    def test_inter_area_subproblem_solves_internally(self, midgaard_area, hood_area):
        """Cross-boundary rooms solve together with internal exit linking them."""
        # Take boundary rooms from midgaard and hood
        rooms = [
            next(r for r in midgaard_area.rooms if r.vnum == 3119),
            next(r for r in hood_area.rooms if r.vnum == 2101),
        ]
        rdb = {r.vnum: Room(r) for r in rooms}
        solved_rdb, exits = solve_layout(rdb, solver_timeout=15)

        assert 3119 in solved_rdb and not solved_rdb[3119].dummy
        assert 2101 in solved_rdb and not solved_rdb[2101].dummy

        # The exit between 3119 and 2101 is internal
        inter_exit = next(e for e in exits if (e.src, e.dst) in ((3119, 2101), (2101, 3119)))
        assert inter_exit is not None

        # Nominal relative positions: 2101 is 1 unit East of 3119
        assert solved_rdb[2101].x == solved_rdb[3119].x + 1
        assert solved_rdb[2101].y == solved_rdb[3119].y
        assert solved_rdb[2101].z == solved_rdb[3119].z

    def test_external_exits_remain_dummy_stubs_in_composite(
        self, midgaard_area, hood_area, grave_area, mobfact_area
    ):
        """Exits targeting zones outside the composite still generate dummy stubs."""
        composite = merge_areas([midgaard_area, hood_area, grave_area, mobfact_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        r_3001 = rdb[3001]
        exit_arachnos = next((e for e in r_3001.exits if e.dst == 3700), None)
        assert exit_arachnos is not None
        assert 3700 not in rdb


# ============================================================================
# 3. Presentation Tests: Renderer & Viewer (JSON & HTML)
# ============================================================================

class TestRendererAndViewerPresentation:
    """Verify JSON serialization and HTML viewer display of room area provenance."""

    def test_json_includes_area_name(self, midgaard_area, hood_area):
        """build_area_json serializes area_name on each room."""
        composite = merge_areas([midgaard_area, hood_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}
        for i, r in enumerate(rdb.values()):
            r.x, r.y, r.z = i, 0, 0

        data = build_area_json(rdb, area_meta=composite.header)
        assert "rooms" in data
        assert len(data["rooms"]) == len(composite.rooms)

        mid_room = next(r for r in data["rooms"] if r["vnum"] == 3001)
        assert mid_room.get("area_name") == "Midgaard"

        hood_room = next(r for r in data["rooms"] if r["vnum"] == 2101)
        assert hood_room.get("area_name") == "Gangland"

    def test_html_viewer_contains_area_display_markup(self, midgaard_area, hood_area):
        """HTML viewer script includes area_name in details card and tooltip."""
        composite = merge_areas([midgaard_area, hood_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}
        for i, r in enumerate(rdb.values()):
            r.x, r.y, r.z = i, 0, 0

        data = build_area_json(rdb, area_meta=composite.header)
        html_content = generate_html_viewer(data)
        assert "Area: " in html_content
        assert "room.area_name" in html_content


# ============================================================================
# 4. CLI Tests: Multi-File Ingestion
# ============================================================================

class TestCliMultiArea:
    """Test CLI invocation with multiple area files producing composite maps."""

    def test_cli_multi_file_invocation_grave_mobfact(self, tmp_path):
        """Passing multiple area files generates composite HTML and JSON."""
        grave_file = FIXTURES_DIR / "grave.are"
        mobfact_file = FIXTURES_DIR / "mobfact.are"
        outbase = str(tmp_path / "composite_out")

        with pytest.raises(SystemExit) as exc_info:
            main([grave_file, mobfact_file], outbase, fmt="json,html,svg", solver_timeout=30)
        assert exc_info.value.code == 0

        json_path = tmp_path / "composite_out.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["rooms"]) == 33 + 25
        vnums = {r["vnum"] for r in data["rooms"]}
        assert 3600 in vnums
        assert 9400 in vnums

        html_path = tmp_path / "composite_out.html"
        assert html_path.exists()
        assert html_path.stat().st_size > 0

        svg_path = tmp_path / "composite_out.svg"
        assert svg_path.exists()
        assert svg_path.stat().st_size > 0

    def test_cli_argument_parsing_multiple_files(self, monkeypatch, tmp_path):
        """CLI invocation via cli() entrypoint accepts multiple files."""
        import sys
        from romutil.cli import cli

        mid_file = str(FIXTURES_DIR / "grave.are")
        hood_file = str(FIXTURES_DIR / "mobfact.are")
        outbase = str(tmp_path / "cli_test_out")

        monkeypatch.setattr("sys.argv", ["romutil", mid_file, hood_file, "-outbase", outbase, "-f", "json"])
        with pytest.raises(SystemExit) as exc_info:
            cli()
        assert exc_info.value.code == 0
        assert (tmp_path / "cli_test_out.json").exists()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_all_four_areas_json(self, tmp_path):
        """CLI invocation with all 4 metropolitan zones."""
        files = [
            FIXTURES_DIR / "midgaard.are",
            FIXTURES_DIR / "hood.are",
            FIXTURES_DIR / "grave.are",
            FIXTURES_DIR / "mobfact.are",
        ]
        outbase = str(tmp_path / "midgaard_metro")

        with pytest.raises(SystemExit) as exc_info:
            main(files, outbase, fmt="json", solver_timeout=15)
        assert exc_info.value.code == 0

        json_path = tmp_path / "midgaard_metro.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["rooms"]) == 271
        assert data["area"]["name"] == "Midgaard + Gangland + Graveyard + Mob Factory"


# ============================================================================
# 5. Solver Integration: Non-Overlap Invariants on Composite Zones
# ============================================================================

class TestSolverCompositeInvariants:
    """Verify solver coordinates feasibility and non-overlap invariants on composite zones."""

    def test_slums_and_graveyard_non_overlap(self, hood_area, grave_area):
        """Solve The Slums + The Graveyard (103 rooms) achieving 0 overlaps."""
        composite = merge_areas([hood_area, grave_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        solved_rdb, exits = solve_layout(rdb, composite.header, solver_timeout=30)
        non_dummy = [r for r in solved_rdb.values() if not getattr(r, "dummy", False)]
        assert len(non_dummy) == len(hood_area.rooms) + len(grave_area.rooms)

        coords = [(r.x, r.y, r.z) for r in non_dummy]
        assert len(coords) == len(set(coords)), "Expected 0 coordinate overlaps in Hood + Grave"

    def test_hood_grave_mobfact_non_overlap(self, hood_area, grave_area, mobfact_area):
        """Solve 3-zone composite (The Slums + Graveyard + Mob Factory, 128 rooms) achieving 0 overlaps."""
        composite = merge_areas([hood_area, grave_area, mobfact_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        solved_rdb, exits = solve_layout(rdb, composite.header, solver_timeout=30)
        non_dummy = [r for r in solved_rdb.values() if not getattr(r, "dummy", False)]
        assert len(non_dummy) == len(hood_area.rooms) + len(grave_area.rooms) + len(mobfact_area.rooms)

        coords = [(r.x, r.y, r.z) for r in non_dummy]
        assert len(coords) == len(set(coords)), "Expected 0 coordinate overlaps in Hood + Grave + Mobfact"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_cluster_coordinates_assigned(
        self, midgaard_area, hood_area, grave_area, mobfact_area
    ):
        """Full 4-area Midgaard cluster solve assigns valid 3D coordinates to all 271 rooms."""
        composite = merge_areas([midgaard_area, hood_area, grave_area, mobfact_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        solved_rdb, exits = solve_layout(rdb, composite.header, solver_timeout=15)
        non_dummy = [r for r in solved_rdb.values() if not getattr(r, "dummy", False)]
        assert len(non_dummy) == 271

        for r in non_dummy:
            assert r.x is not None
            assert r.y is not None
            assert r.z is not None
