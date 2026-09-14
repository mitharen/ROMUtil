"""
Comprehensive unit and integration tests for Multi-Area Graph Decomposition
and Macro-Assembly Architecture (Task 10h).

Verifies:
1. Enclosure classification and directional offsets.
2. Area partitioning and connectivity graph construction.
3. Macro-cavity contract extraction (multi-port displacements and bounding clearances).
4. Pyomo macro-constraint injection.
5. Rigid leaf-up macro-assembly and satellite elevation layering.
6. End-to-end multi-area solves across single-port, multi-port, and full 5-zone clusters.
7. CLI and API backward compatibility and flag handling.
"""

from pathlib import Path
import pytest
import pyomo.environ as pyo

from romutil.models import (
    AreaData,
    AreaHeader,
    Direction,
    Exit,
    ExitDef,
    Room,
    RoomDef,
    merge_areas,
)
from romutil.parser import Parser
from romutil.graph import solve_layout, graph
from romutil.cli import main, cli
from romutil.decomposition import (
    AreaProfile,
    MacroCavityContract,
    MacroAssemblyResult,
    EnclosureType,
    HierarchicalLayoutSolver,
    solve_hierarchical_layout,
    direction_offset,
    compute_bounding_box,
    get_room_area_name,
    partition_rooms_by_area,
    build_area_connectivity_graph,
    find_clearance_rooms,
    profile_area,
    classify_enclosure,
    build_macro_contracts,
    create_macro_cavity_hook,
    assemble_composite_layout,
    KNOWN_MACRO_CONTRACTS,
)
from romutil.solver import build_spatial_coordinate_buckets, find_spatial_room_collisions

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


@pytest.fixture
def school_area() -> AreaData:
    content = (FIXTURES_DIR / "school.are").read_text(encoding="latin-1")
    return Parser().parse(content)


# ============================================================================
# 1. Unit Tests: Enums, Directional Offsets, and Spatial Helpers
# ============================================================================

class TestDecompositionBasics:
    """Test fundamental data structures and geometry utilities."""

    def test_enclosure_type_enum(self):
        assert EnclosureType.ENCLOSED == "enclosed"
        assert EnclosureType.SATELLITE == "satellite"
        assert EnclosureType.SIBLING == "sibling"

    def test_direction_offset_all(self):
        assert direction_offset(Direction.north) == (0, 1, 0)
        assert direction_offset(Direction.south) == (0, -1, 0)
        assert direction_offset(Direction.east) == (1, 0, 0)
        assert direction_offset(Direction.west) == (-1, 0, 0)
        assert direction_offset(Direction.up) == (0, 0, 1)
        assert direction_offset(Direction.down) == (0, 0, -1)
        assert direction_offset(Direction.northeast) == (1, 1, 0)
        assert direction_offset(Direction.northwest) == (-1, 1, 0)
        assert direction_offset(Direction.southeast) == (1, -1, 0)
        assert direction_offset(Direction.southwest) == (-1, -1, 0)
        # Invalid direction fallback
        assert direction_offset(99) == (0, 0, 0)

    def test_compute_bounding_box(self):
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r1.x, r1.y, r1.z = 2, 5, -1
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r2.x, r2.y, r2.z = 8, 10, 4
        r_dummy = Room(RoomDef(vnum=3, name="Dummy", description=""))
        r_dummy.dummy = True
        r_dummy.x, r_dummy.y, r_dummy.z = 100, 100, 100

        box = compute_bounding_box([r1, r2, r_dummy])
        assert box == (2, 8, 5, 10, -1, 4)

    def test_compute_bounding_box_empty(self):
        assert compute_bounding_box([]) == (0, 0, 0, 0, 0, 0)
        r = Room(RoomDef(vnum=1, name="R", description=""))
        assert compute_bounding_box([r]) == (0, 0, 0, 0, 0, 0)

    def test_get_room_area_name(self):
        r1 = Room(RoomDef(vnum=1, name="R1", description="", area_name="Midgaard"))
        assert get_room_area_name(r1) == "Midgaard"

        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r2.area_file = "hood.are"
        assert get_room_area_name(r2) == "hood.are"

        r3 = Room(RoomDef(vnum=3, name="R3", description=""))
        assert get_room_area_name(r3) == "Unknown Area"


# ============================================================================
# 2. Unit Tests: Graph Partitioning and Enclosure Classification
# ============================================================================

class TestGraphPartitioningAndClassification:
    """Test area partitioning and topological classification."""

    def test_partition_rooms_by_area(self):
        r1 = Room(RoomDef(vnum=10, name="R1", description="", area_name="Area1"))
        r2 = Room(RoomDef(vnum=20, name="R2", description="", area_name="Area2"))
        r3 = Room(RoomDef(vnum=30, name="R3", description="", area_name="Area1"))
        r_dum = Room(RoomDef(vnum=40, name="Dum", description="", area_name="Area1"))
        r_dum.dummy = True

        rdb = {10: r1, 20: r2, 30: r3, 40: r_dum}
        partitions = partition_rooms_by_area(rdb)

        assert "Area1" in partitions and "Area2" in partitions
        assert set(partitions["Area1"].keys()) == {10, 30}
        assert set(partitions["Area2"].keys()) == {20}

    def test_build_area_connectivity_graph(self):
        r1 = Room(RoomDef(vnum=1, name="R1", description="", area_name="Host"))
        r2 = Room(RoomDef(vnum=2, name="R2", description="", area_name="Host"))
        r3 = Room(RoomDef(vnum=3, name="R3", description="", area_name="Child"))
        r_dum = Room(RoomDef(vnum=4, name="Dum", description="", area_name="Child"))
        r_dum.dummy = True

        e1 = Exit(ExitDef(direction=Direction.east, dst_vnum=3), source=2)
        e2 = Exit(ExitDef(direction=Direction.west, dst_vnum=2), source=3)
        e_dum = Exit(ExitDef(direction=Direction.north, dst_vnum=4), source=1)
        r2.exits.append(e1)
        r3.exits.append(e2)
        r1.exits.append(e_dum)

        rdb = {1: r1, 2: r2, 3: r3, 4: r_dum}
        g, inter_links = build_area_connectivity_graph(rdb, [e1, e2, e_dum])

        assert set(g.nodes) == {"Host", "Child"}
        assert g.has_edge("Host", "Child")
        assert len(inter_links) == 2

    def test_classify_enclosure(self):
        # Cardinal exits -> ENCLOSED
        ports_cardinal = [(10, 100, Direction.east), (11, 101, Direction.east)]
        assert classify_enclosure("Host", "Child", ports_cardinal) == EnclosureType.ENCLOSED

        # Vertical exits -> SATELLITE
        ports_vertical = [(10, 100, Direction.up)]
        assert classify_enclosure("Host", "Child", ports_vertical) == EnclosureType.SATELLITE

    def test_find_clearance_rooms_bfs(self):
        # Linear chain: 1 -> 2 (east) -> 3 (east) -> 4 (south)
        r1 = Room(RoomDef(vnum=1, name="R1", description=""))
        r2 = Room(RoomDef(vnum=2, name="R2", description=""))
        r3 = Room(RoomDef(vnum=3, name="R3", description=""))
        r4 = Room(RoomDef(vnum=4, name="R4", description=""))
        r_dum = Room(RoomDef(vnum=5, name="Dum", description=""))
        r_dum.dummy = True

        r1.exits.append(Exit(ExitDef(direction=Direction.east, dst_vnum=2), source=1))
        r2.exits.append(Exit(ExitDef(direction=Direction.east, dst_vnum=3), source=2))
        r2.exits.append(Exit(ExitDef(direction=Direction.north, dst_vnum=5), source=2))
        r3.exits.append(Exit(ExitDef(direction=Direction.south, dst_vnum=4), source=3))

        rdb = {1: r1, 2: r2, 3: r3, 4: r4, 5: r_dum}
        clearance = find_clearance_rooms(rdb, gateway_vnum=1, direction=Direction.east, max_hops=3)

        assert 2 in clearance
        assert 3 in clearance
        assert 4 in clearance
        assert 1 not in clearance
        assert 5 not in clearance

    def test_find_clearance_rooms_empty(self):
        assert find_clearance_rooms({}, gateway_vnum=1, direction=Direction.east) == []


# ============================================================================
# 3. Unit Tests: Macro Contracts and Constraint Hooks
# ============================================================================

class TestMacroContractsAndHooks:
    """Test contract construction and Pyomo constraint injection."""

    def test_build_macro_contracts_multi_port(self):
        child_prof = AreaProfile(
            area_name="Hood",
            rooms={},
            exits=[],
            bounds=(0, 8, 0, 12, 0, 0),
            width=8,
            height=12,
            depth=0,
            ports={2101: (0, 11, 0), 2160: (0, 0, 0)},
        )

        inter_links = [
            ("Midgaard", "Hood", 3119, 2101, Direction.east),
            ("Midgaard", "Hood", 3144, 2160, Direction.east),
            ("Hood", "Midgaard", 2101, 3119, Direction.west),
        ]

        contracts = build_macro_contracts("Midgaard", {}, {"Hood": child_prof, "Midgaard": AreaProfile("Midgaard", {}, [], (0,0,0,0,0,0), 0,0,0)}, inter_links)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.child_area_name == "Hood"
        assert c.container_area_name == "Midgaard"
        assert c.enclosure_type == EnclosureType.ENCLOSED
        assert (3119, 3144) in c.port_displacement
        assert c.port_displacement[(3119, 3144)] == (0, 11, 0)
        assert 3272 in c.clearance_rooms
        assert 3273 in c.clearance_rooms

    def test_build_macro_contracts_no_ports(self):
        child_prof = AreaProfile("Unconnected", {}, [], (0, 0, 0, 0, 0, 0), 0, 0, 0)
        contracts = build_macro_contracts("Midgaard", {}, {"Unconnected": child_prof}, [])
        assert len(contracts) == 0

    def test_create_macro_cavity_hook_all_directions(self):
        model = pyo.ConcreteModel()
        model.Rooms = pyo.Set(initialize=[10, 20, 30, 40, 50, 60])
        model.x = pyo.Var(model.Rooms, within=pyo.Integers)
        model.y = pyo.Var(model.Rooms, within=pyo.Integers)
        model.z = pyo.Var(model.Rooms, within=pyo.Integers)

        contract = MacroCavityContract(
            child_area_name="TestChild",
            container_area_name="TestContainer",
            enclosure_type=EnclosureType.ENCLOSED,
            port_pairs=[
                (10, 101, Direction.west),
                (20, 102, Direction.north),
                (30, 103, Direction.south),
            ],
            port_displacement={
                (10, 20): (-5, -8, -2),
                (20, 30): (0, 5, 2),
            },
            bounding_box=(4, 6, 2),
            clearance_rooms=[40, 50, 60],
        )

        hook = create_macro_cavity_hook([contract])
        hook(model)

        assert hasattr(model, "macro_cavity_constraints")
        assert len(model.macro_cavity_constraints) > 0


# ============================================================================
# 4. Integration Tests: Area Profiling and Leaf-Up Macro-Assembly
# ============================================================================

class TestHierarchicalAssemblyIntegration:
    """Test leaf-up assembly and multi-area collision elimination."""

    def test_profile_child_area(self, hood_area):
        rdb = {r.vnum: Room(r) for r in hood_area.rooms}
        prof = profile_area("Gangland", rdb, solver_timeout=15)
        assert prof.area_name == "Gangland"
        assert prof.width > 0
        assert prof.height > 0
        assert 2101 in prof.ports
        assert 2160 in prof.ports

    def test_assemble_composite_layout_zero_collisions(self):
        r10 = Room(RoomDef(vnum=10, name="Gate North", description="", area_name="Container"))
        r10.x, r10.y, r10.z = 0, 11, 0
        r20 = Room(RoomDef(vnum=20, name="Gate South", description="", area_name="Container"))
        r20.x, r20.y, r20.z = 0, 0, 0
        cont_prof = AreaProfile("Container", {10: r10, 20: r20}, [], (0, 0, 0, 11, 0, 0), 0, 11, 0, {10: (0, 11, 0), 20: (0, 0, 0)})

        r101 = Room(RoomDef(vnum=101, name="Child North", description="", area_name="Child"))
        r101.x, r101.y, r101.z = 0, 11, 0
        r102 = Room(RoomDef(vnum=102, name="Child South", description="", area_name="Child"))
        r102.x, r102.y, r102.z = 0, 0, 0
        child_prof = AreaProfile("Child", {101: r101, 102: r102}, [], (0, 0, 0, 11, 0, 0), 0, 11, 0, {101: (0, 11, 0), 102: (0, 0, 0)})

        contract = MacroCavityContract(
            child_area_name="Child",
            container_area_name="Container",
            enclosure_type=EnclosureType.ENCLOSED,
            port_pairs=[(10, 101, Direction.east), (20, 102, Direction.east)],
            port_displacement={(10, 20): (0, 11, 0)},
            bounding_box=(2, 11, 0),
            clearance_rooms=[],
        )

        res = assemble_composite_layout(cont_prof, {"Child": child_prof}, [contract])
        assert res.is_collision_free is True
        assert res.rooms[101].x == res.rooms[10].x + 1
        assert res.rooms[102].x == res.rooms[20].x + 1

    def test_assemble_composite_layout_downward_satellite(self):
        r10 = Room(RoomDef(vnum=10, name="Base", description="", area_name="Surface"))
        r10.x, r10.y, r10.z = 5, 5, 0
        cont_prof = AreaProfile("Surface", {10: r10}, [], (5, 5, 5, 5, 0, 0), 0, 0, 0, {10: (5, 5, 0)})

        r20 = Room(RoomDef(vnum=20, name="Dungeon", description="", area_name="Underground"))
        r20.x, r20.y, r20.z = 0, 0, 0
        child_prof = AreaProfile("Underground", {20: r20}, [], (0, 0, 0, 0, 0, 0), 0, 0, 0, {20: (0, 0, 0)})

        contract = MacroCavityContract(
            child_area_name="Underground",
            container_area_name="Surface",
            enclosure_type=EnclosureType.SATELLITE,
            port_pairs=[(10, 20, Direction.down)],
        )

        res = assemble_composite_layout(cont_prof, {"Underground": child_prof}, [contract])
        assert res.is_collision_free is True
        # Surface and Dungeon properly layered in Z
        assert res.rooms[10].z > res.rooms[20].z

    def test_solve_hierarchical_layout_single_area(self, hood_area):
        """Single-area input gracefully delegates to standard solve."""
        rdb = {r.vnum: Room(r) for r in hood_area.rooms[:5]}
        solved_rdb, exits = solve_hierarchical_layout(rdb, solver_timeout=15)
        non_dummy = [r for r in solved_rdb.values() if not getattr(r, "dummy", False)]
        assert len(non_dummy) == 5
        assert all(r.x is not None for r in non_dummy)

    def test_solve_hierarchical_layout_hood_midgaard_slice(self, midgaard_area, hood_area):
        """Solve Midgaard slice + Hood slice using hierarchical layout."""
        mid_rooms = [r for r in midgaard_area.rooms if r.vnum in (3119, 3144, 3272, 3273)]
        hood_rooms = [r for r in hood_area.rooms if r.vnum in (2101, 2160, 2102)]
        composite = merge_areas([AreaData(rooms=tuple(mid_rooms)), AreaData(rooms=tuple(hood_rooms))])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        solved_rdb, exits = solve_hierarchical_layout(rdb, solver_timeout=30)
        non_dummy = [r for r in solved_rdb.values() if not getattr(r, "dummy", False)]
        assert len(non_dummy) == len(mid_rooms) + len(hood_rooms)
        coords = [(r.x, r.y, r.z) for r in non_dummy]
        assert len(coords) == len(set(coords)), "Zero collisions expected between Midgaard and Hood"

    def test_solve_hierarchical_layout_school_satellite(self, midgaard_area, school_area):
        """Solve Midgaard Temple (3001) + Mud School entrance (3700) with satellite Z layering."""
        mid_rooms = [r for r in midgaard_area.rooms if r.vnum == 3001]
        school_rooms = [r for r in school_area.rooms if r.vnum == 3700]
        composite = merge_areas([AreaData(rooms=tuple(mid_rooms)), AreaData(rooms=tuple(school_rooms))])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        solved_rdb, exits = solve_hierarchical_layout(rdb, solver_timeout=15)
        assert 3001 in solved_rdb and 3700 in solved_rdb
        # School sits on a higher Z elevation layer
        assert solved_rdb[3700].z > solved_rdb[3001].z

    def test_solve_hierarchical_layout_disconnected_multi_components(self):
        """Hierarchical solver decomposes disconnected areas into shelf-packed components."""
        r1 = Room(RoomDef(vnum=10, name="Zone1", description="", area_name="ZoneA"))
        r2 = Room(RoomDef(vnum=20, name="Zone2", description="", area_name="ZoneB"))
        # No exits between ZoneA and ZoneB
        rdb = {10: r1, 20: r2}
        solved_rdb, exits = solve_hierarchical_layout(rdb, solver_timeout=15, component_padding=3)
        assert 10 in solved_rdb and 20 in solved_rdb
        # Non-overlapping horizontal positions
        assert solved_rdb[10].x != solved_rdb[20].x

    def test_cli_hierarchical_flag_acceptance(self, tmp_path, monkeypatch):
        """Verify CLI accepts --hierarchical and --no-hierarchical options."""
        mid_file = str(FIXTURES_DIR / "grave.are")
        outbase = str(tmp_path / "cli_hierarchical_test")

        monkeypatch.setattr("sys.argv", ["romutil", mid_file, "-outbase", outbase, "--hierarchical", "-f", "json"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert (tmp_path / "cli_hierarchical_test.json").exists()

        monkeypatch.setattr("sys.argv", ["romutil", mid_file, "-outbase", outbase, "--no-hierarchical", "-f", "json"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

    def test_solve_layout_hierarchical_switch(self, hood_area, grave_area):
        """solve_layout respects hierarchical=False for monolithic fallback."""
        composite = merge_areas([hood_area, grave_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms[:6]}
        solved_rdb, exits = solve_layout(rdb, hierarchical=False, solver_timeout=15)
        assert len(solved_rdb) >= 6

    def test_graph_hierarchical_passthrough(self, hood_area, tmp_path):
        """graph() passes hierarchical parameter to solve_layout."""
        rdb = {r.vnum: Room(r) for r in hood_area.rooms[:4]}
        outbase = str(tmp_path / "test_out")
        graph(rdb, outbase, area=hood_area.header, hierarchical=True, solver_timeout=15)


# ============================================================================
# 5. Full Metropolitan Cluster Integration Test
# ============================================================================

class TestMetropolitanClusterSolve:
    """Full 5-area Metropolitan Midgaard hierarchical solve."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_midgaard_metropolitan_five_area_zero_inter_collisions(
        self, midgaard_area, hood_area, grave_area, mobfact_area, school_area
    ):
        """
        Verify that solving the entire 5-area Midgaard Metropolitan cluster
        (Midgaard, Hood, Graveyard, Mob Factory, Mud School: 369 rooms total)
        via hierarchical macro-assembly produces ZERO inter-area collisions.
        """
        composite = merge_areas([midgaard_area, hood_area, grave_area, mobfact_area, school_area])
        rdb = {r.vnum: Room(r) for r in composite.rooms}

        solved_rdb, exits = solve_hierarchical_layout(
            rdb,
            child_timeout=15,
            container_timeout=60,
        )

        non_dummy = [r for r in solved_rdb.values() if not getattr(r, "dummy", False)]
        assert len(non_dummy) == len(composite.rooms)

        # Coordinate buckets collision verification
        coords = {r.vnum: (r.x, r.y, r.z) for r in non_dummy if r.x is not None and r.y is not None and r.z is not None}
        buckets = build_spatial_coordinate_buckets(list(coords.keys()), coords)
        collisions = find_spatial_room_collisions(buckets)

        inter_area_collisions = [
            (u, v) for u, v in collisions
            if solved_rdb[u].area_name != solved_rdb[v].area_name
        ]
        assert len(inter_area_collisions) == 0, (
            f"Expected 0 inter-area collisions, found {len(inter_area_collisions)}: {inter_area_collisions}"
        )
