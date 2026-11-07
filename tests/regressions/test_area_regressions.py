"""Historical area layout defect regression test suite.

Verifies end-to-end spatial layout invariants across authentic ROM area files:
- Sewer (sewer.are): parallel branch coordinate isolation and corridor non-penetration.
- Thalos (thalos.are): split-pair boundary port alignment, elevation, and exit shear.
- Midgaard (midgaard.are): Main Street collinearity, Clan Road elevation, zero collisions.
- Moria (moria.are): orthogonal bridge series elasticity, typed clearance, zero crossings.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from romutil.layout import HierarchicalLayoutSolver, LayoutGraph, Segment3D, segments_intersect_3d
from romutil.parser import parse_file
from romutil.types import LayoutOptions, SolverStatus, WorldGraph
from tests.regressions.invariants import (
    assert_axis_aligned_exits,
    assert_corridor_non_penetration,
    assert_zero_coordinate_collisions,
)

pytestmark = [pytest.mark.regression]

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "areas"


class TestSewerRegression:
    """Regression suite verifying spatial decomposition and branch isolation for sewer.are."""

    def test_sewer_branch_isolation_and_clearance(self) -> None:
        """Sewer solves with 177 distinct room coordinates and isolated parallel branches."""
        area = parse_file(FIXTURES_DIR / "sewer.are")
        graph = WorldGraph.from_area_data(area)
        assert len(graph.rooms) == 177

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(graph, LayoutOptions(timeout_seconds=30))

        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(layout.coordinates) == 177

        # Invariant 1: Zero coordinate collisions across all 177 rooms
        assert_zero_coordinate_collisions(layout)

        # Invariant 2: Parallel cut branches do not overlap
        coords = layout.coordinates
        assert coords[7219] != coords[7231]
        assert coords[7217] != coords[7230]
        assert coords[7218] != coords[7229]

        # Invariant 3: Corridor between rooms 7011 and 7021 avoids interior room 7012
        assert_corridor_non_penetration(layout, graph)


class TestThalosRegression:
    """Regression suite verifying split-pair boundary port alignment for thalos.are."""

    def test_thalos_split_pair_boundary_alignment_and_elevation(self) -> None:
        """Thalos split-pair boundary ports solve on-axis with planar elevation and zero exit shear."""
        area = parse_file(FIXTURES_DIR / "thalos.are")
        graph = WorldGraph.from_area_data(area)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(graph, LayoutOptions(timeout_seconds=30))

        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        coords = layout.coordinates

        # Invariant 1: Zero coordinate collisions
        assert_zero_coordinate_collisions(layout)

        # Invariant 2: Split-pair boundary ports align on-axis and at planar elevation z=2
        assert 5243 in coords and 5241 in coords
        c5243 = coords[5243]
        c5241 = coords[5241]

        assert c5241.x - c5243.x >= 1
        assert c5241.y == c5243.y, f"Split-pair boundary ports differ in y: {c5243} vs {c5241}"
        assert c5243.z == 2 and c5241.z == 2, f"Expected elevation z=2, got {c5243.z} and {c5241.z}"

        # Invariant 3: Key cardinal exit corridors have zero perpendicular shear
        designated_exits = [
            (5243, 5241),
            (5201, 5239),
            (5239, 5240),
            (5240, 5241),
            (5214, 5211),
            (5211, 5210),
            (5210, 5209),
        ]
        assert_axis_aligned_exits(layout, graph, designated_exits)

        # Invariant 4: No corridor penetrations across Thalos
        assert_corridor_non_penetration(layout, graph)


class TestMidgaardRegression:
    """Regression suite verifying Main Street collinearity and Clan Road elevation in midgaard.are."""

    def test_midgaard_main_street_collinearity_and_clan_road_elevation(self) -> None:
        """Midgaard solves with Main Street collinear at y=17, Clan Road elevated at y=21, and zero collisions."""
        area = parse_file(FIXTURES_DIR / "midgaard.are")
        graph = WorldGraph.from_area_data(area)
        assert len(graph.rooms) == 143

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(graph, LayoutOptions(timeout_seconds=30))

        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(layout.coordinates) == 143
        coords = layout.coordinates

        # Invariant 1: Zero coordinate collisions across all 143 rooms
        assert_zero_coordinate_collisions(layout)

        # Invariant 2: Main Street collinearity along y=17 at ground elevation z=0
        main_street_vnums = [3053, 3012, 3013, 3014, 3015, 3016, 3040]
        for v in main_street_vnums:
            assert v in coords
            assert coords[v].y == 17, f"Main Street room {v} at y={coords[v].y}, expected 17"
            assert coords[v].z == 0, f"Main Street room {v} at z={coords[v].z}, expected 0"

        # Invariant 3: Main Street exit segments have zero perpendicular shear
        main_street_exits = [
            (3012, 3013),
            (3013, 3014),
            (3014, 3015),
            (3015, 3016),
        ]
        assert_axis_aligned_exits(layout, graph, main_street_exits)

        # Invariant 4: Clan Road routes north of Market Square and Temple Square (elevation y=21)
        clan_road_rooms = [3258, 3266, 3315, 3301, 3302, 3303]
        for v in clan_road_rooms:
            assert v in coords
            assert coords[v].y == 21, f"Clan Road room {v} at y={coords[v].y}, expected 21"

        assert coords[3315].y > coords[3005].y, "Clan Road must route north of Temple Square"
        assert coords[3258].y != coords[3001].y, "Clan Road must not penetrate Cleric Inner Sanctum latitude"

        # Invariant 5: Key boundary port pairs do not collide
        assert coords[3009] != coords[3004]
        assert coords[3028] != coords[3049]
        assert coords[3029] != coords[3203]

        # Invariant 6: Strict corridor non-penetration across all Midgaard rooms
        assert_corridor_non_penetration(layout, graph)


class TestMoriaRegression:
    """Regression suite verifying orthogonal bridge elasticity and optimality for moria.are."""

    def test_moria_orthogonal_bridge_elasticity_and_optimality(self) -> None:
        """Moria solves to OPTIMAL status with 121 rooms, 0 collisions, and 0 corridor crossings."""
        area = parse_file(FIXTURES_DIR / "moria.are")
        graph = WorldGraph.from_area_data(area)
        assert len(graph.rooms) == 121

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(graph, LayoutOptions(timeout_seconds=20))

        # Invariant 1: Solves to OPTIMAL status
        assert layout.diagnostics.status == SolverStatus.OPTIMAL
        assert len(layout.coordinates) == 121
        coords = layout.coordinates

        # Invariant 2: Zero coordinate collisions across all 121 rooms
        assert_zero_coordinate_collisions(layout)

        # Invariant 3: Strict corridor non-penetration
        assert_corridor_non_penetration(layout, graph)

        # Invariant 4: Zero corridor crossings across all active planar edges
        simplified = solver.simplifier.simplify(
            LayoutGraph.from_world_graph(graph), LayoutOptions(timeout_seconds=10)
        )
        segments: list[tuple[int, int, Segment3D]] = []
        for e in simplified.graph.edges:
            if e.cut or e.src not in coords or e.dst not in coords:
                continue
            segments.append((e.src, e.dst, Segment3D(coords[e.src], coords[e.dst])))

        crossings: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(len(segments)):
            u1, v1, s1 = segments[i]
            for j in range(i + 1, len(segments)):
                u2, v2, s2 = segments[j]
                if len({u1, v1, u2, v2}) < 4:
                    continue
                if segments_intersect_3d(s1, s2):
                    crossings.append(((u1, v1), (u2, v2)))

        assert crossings == [], f"Found corridor crossings in Moria: {crossings}"
