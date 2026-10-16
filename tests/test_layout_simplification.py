"""Unit and regression tests for layout simplification and warp cycle filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from romutil.layout import (
    HierarchicalLayoutSolver,
    LayoutEdge,
    LayoutGraph,
    WarpSeverer,
)
from romutil.models import Direction, Exit, Room
from romutil.parser import parse_file
from romutil.types import LayoutOptions, WorldGraph


def _make_room(vnum: int, exits: list[Exit] | None = None) -> Room:
    """Helper to construct a Room with specified exits."""
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(exits or []),
    )


class TestWarpSevererExternalExitFiltering:
    """Tests verifying that out-of-area dangling exits do not create synthetic warp cycles."""

    def test_multiple_rooms_exiting_to_same_external_vnum(self) -> None:
        """Rooms exiting to the same external vnum do not form a cycle and trigger no chord cuts."""
        # Rooms 1 and 2 connect East/West, and both have an Up exit to external vnum 9999
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=int(Direction.east)),
            LayoutEdge(src=2, dst=1, direction=int(Direction.west)),
            LayoutEdge(src=1, dst=9999, direction=int(Direction.up)),
            LayoutEdge(src=2, dst=9999, direction=int(Direction.up)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        # External node 9999 must not form an undirected cycle 1 - 9999 - 2 - 1
        assert len(simplified.severed_chords) == 0
        # External exits must remain preserved in the graph edges
        assert len(simplified.graph.edges) == 4
        assert any(e.dst == 9999 and not e.cut for e in simplified.graph.edges)

    def test_disconnected_rooms_with_identical_external_exit(self) -> None:
        """Multiple disconnected rooms with one-way exits to the same external vnum remain uncut."""
        rooms = {
            10: _make_room(10),
            20: _make_room(20),
            30: _make_room(30),
        }
        edges = [
            LayoutEdge(src=10, dst=8888, direction=int(Direction.north)),
            LayoutEdge(src=20, dst=8888, direction=int(Direction.north)),
            LayoutEdge(src=30, dst=8888, direction=int(Direction.north)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 0
        assert len(simplified.graph.edges) == 3

    def test_external_self_loops_and_antiparallel_edges_ignored(self) -> None:
        """Exits to and from non-existent external vnums are ignored by 1-cycle and 2-cycle checks."""
        rooms = {1: _make_room(1)}
        edges = [
            # Dangling self loop on external room
            LayoutEdge(src=9999, dst=9999, direction=int(Direction.north)),
            # Antiparallel pair involving external room
            LayoutEdge(src=1, dst=9999, direction=int(Direction.north)),
            LayoutEdge(src=9999, dst=1, direction=int(Direction.north)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 0
        assert len(simplified.graph.edges) == 3

    def test_genuine_in_area_3cycle_warp_severed(self) -> None:
        """Genuine in-area 3-cycle with all North steps is detected and severed."""
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=int(Direction.north)),
            LayoutEdge(src=2, dst=3, direction=int(Direction.north)),
            LayoutEdge(src=3, dst=1, direction=int(Direction.north)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.cut is True
        assert {severed.src, severed.dst}.issubset({1, 2, 3})

    def test_genuine_in_area_4cycle_warp_all_north_east_severed(self) -> None:
        """Genuine in-area 4-cycle with all North/East steps is detected and severed."""
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
            4: _make_room(4),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=int(Direction.north)),
            LayoutEdge(src=2, dst=3, direction=int(Direction.east)),
            LayoutEdge(src=3, dst=4, direction=int(Direction.north)),
            LayoutEdge(src=4, dst=1, direction=int(Direction.east)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.cut is True
        assert {severed.src, severed.dst}.issubset({1, 2, 3, 4})

    def test_non_euclidean_2cycle_severed(self) -> None:
        """Non-Euclidean 2-cycle (both North) has minimal edge severed."""
        rooms = {1: _make_room(1), 2: _make_room(2)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=int(Direction.north), weight=2),
            LayoutEdge(src=2, dst=1, direction=int(Direction.north), weight=1),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].cut is True
        assert simplified.severed_chords[0].weight == 1

    def test_2cycle_invalid_direction_skipped(self) -> None:
        """Antiparallel pair with invalid direction is safely skipped."""
        rooms = {1: _make_room(1), 2: _make_room(2)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=99),
            LayoutEdge(src=2, dst=1, direction=int(Direction.north)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 0

    def test_midgaard_warp_severer_and_poor_alley_square_connection(self) -> None:
        """In midgaard.are, WarpSeverer severs 0 chords and Poor Alley / Common Square remain connected."""
        midgaard_path = Path("tests/fixtures/areas/midgaard.are")
        area = parse_file(midgaard_path)
        world_graph = WorldGraph.from_area_data(area)
        layout_graph = LayoutGraph.from_world_graph(world_graph)

        severer = WarpSeverer()
        simplified = severer.simplify(layout_graph)

        # Midgaard has out-of-area exits (e.g. to 1219) but zero internal non-Euclidean warp cycles
        assert len(simplified.severed_chords) == 0

        # Street connection between Eastern End of Poor Alley (3024) and The Common Square (3025)
        edges_3024_3025 = [
            e
            for e in simplified.graph.edges
            if (e.src, e.dst) in ((3024, 3025), (3025, 3024))
        ]
        assert len(edges_3024_3025) == 2
        for edge in edges_3024_3025:
            assert edge.cut is False

        # Solve layout and verify exact unit offset between 3024 and 3025 along East
        solver = HierarchicalLayoutSolver.production()
        solution = solver.solve(world_graph, LayoutOptions(timeout_seconds=30))

        c3024 = solution.coordinates[3024]
        c3025 = solution.coordinates[3025]
        assert c3025.x - c3024.x == 1
        assert c3025.y - c3024.y == 0
        assert c3025.z - c3024.z == 0
