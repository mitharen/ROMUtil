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
from romutil.layout.ir import CorridorSegment
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


class TestWarpSevererStratifiedScoring:
    """Tests verifying stratified chord scoring in WarpSeverer."""

    def test_multiple_rooms_exiting_to_same_external_vnum(self) -> None:
        """Rooms exiting to the same external vnum form a warp severed at a one-way chord."""
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

        # One-way chord to external 9999 is severed, preserving bidirectional street 1 <-> 2
        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.dst == 9999
        assert severed.cut is True
        # Bidirectional street 1 <-> 2 remains intact
        assert any(e.src == 1 and e.dst == 2 and not e.cut for e in simplified.graph.edges)
        assert any(e.src == 2 and e.dst == 1 and not e.cut for e in simplified.graph.edges)

    def test_disconnected_rooms_with_identical_external_exit(self) -> None:
        """Multiple disconnected rooms with one-way exits to the same external vnum form no cycles."""
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

    def test_external_self_loops_and_antiparallel_edges_severed_symmetrically(self) -> None:
        """Non-Euclidean self-loops and antiparallel cycles involving external vnums are naturally severed."""
        rooms = {1: _make_room(1)}
        edges = [
            # Dangling self loop on external room
            LayoutEdge(src=9999, dst=9999, direction=int(Direction.north)),
            # Antiparallel pair involving external room (both North is non-Euclidean)
            LayoutEdge(src=1, dst=9999, direction=int(Direction.north)),
            LayoutEdge(src=9999, dst=1, direction=int(Direction.north)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 2
        assert all(c.cut for c in simplified.severed_chords)

    def test_cycle_with_oneway_and_bidirectional_severs_oneway(self) -> None:
        """When a warp cycle contains one-way and bidirectional edges, one-way exit is severed."""
        # 1 <-> 2 (bidirectional)
        # 2 <-> 3 (bidirectional)
        # 3 -> 1 (one-way ladder/chute)
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=int(Direction.north)),
            LayoutEdge(src=2, dst=1, direction=int(Direction.south)),
            LayoutEdge(src=2, dst=3, direction=int(Direction.east)),
            LayoutEdge(src=3, dst=2, direction=int(Direction.west)),
            LayoutEdge(src=3, dst=1, direction=int(Direction.down)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)
        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        # Severed chord must be the one-way exit 3 -> 1
        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].src == 3
        assert simplified.severed_chords[0].dst == 1
        # Bidirectional street edges 1 <-> 2 and 2 <-> 3 remain unsevered
        active_pairs = {(e.src, e.dst) for e in simplified.graph.edges if not e.cut}
        assert (1, 2) in active_pairs and (2, 1) in active_pairs
        assert (2, 3) in active_pairs and (3, 2) in active_pairs

    def test_bidirectional_cycle_severs_peripheral_over_arterial(self) -> None:
        """When all edges in a cycle are bidirectional, lower-degree peripheral edges are severed first."""
        # 1 and 4 are high-degree arterial intersections (degree 4).
        # 2 is degree 2, 3 is degree 2 (peripheral street).
        # Cycle 1 - 2 - 3 - 4 - 1 with all North steps (warp).
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
            4: _make_room(4),
            10: _make_room(10),
            11: _make_room(11),
            12: _make_room(12),
            13: _make_room(13),
        }
        edges = [
            # Cycle edges (all bidirectional, but directions form a warp: all North)
            LayoutEdge(src=1, dst=2, direction=int(Direction.north)),
            LayoutEdge(src=2, dst=1, direction=int(Direction.south)),
            LayoutEdge(src=2, dst=3, direction=int(Direction.north)),
            LayoutEdge(src=3, dst=2, direction=int(Direction.south)),
            LayoutEdge(src=3, dst=4, direction=int(Direction.north)),
            LayoutEdge(src=4, dst=3, direction=int(Direction.south)),
            LayoutEdge(src=4, dst=1, direction=int(Direction.north)),
            LayoutEdge(src=1, dst=4, direction=int(Direction.south)),
            # Additional degree for 1 and 4 (arterial hubs)
            LayoutEdge(src=1, dst=10, direction=int(Direction.east)),
            LayoutEdge(src=10, dst=1, direction=int(Direction.west)),
            LayoutEdge(src=1, dst=11, direction=int(Direction.west)),
            LayoutEdge(src=11, dst=1, direction=int(Direction.east)),
            LayoutEdge(src=4, dst=12, direction=int(Direction.east)),
            LayoutEdge(src=12, dst=4, direction=int(Direction.west)),
            LayoutEdge(src=4, dst=13, direction=int(Direction.west)),
            LayoutEdge(src=13, dst=4, direction=int(Direction.east)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)
        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        # In the cycle 1 - 2 - 3 - 4 - 1:
        # deg(1) = 4, deg(4) = 4, deg(2) = 2, deg(3) = 2.
        # deg_sum(2, 3) = 4 (peripheral).
        # deg_sum(1, 4) = 8 (arterial street connecting two hubs).
        # Severer must cut (2, 3), protecting arterial edge (1, 4).
        severed_pairs = {(min(e.src, e.dst), max(e.src, e.dst)) for e in simplified.severed_chords}
        assert (2, 3) in severed_pairs
        assert (1, 4) not in severed_pairs

    def test_cycle_prefers_atomic_edge_over_corridor_macro_edge(self) -> None:
        """When candidate bidirectional edges have different weights, atomic edge is severed first."""
        # 1 - 2 - 3 - 1 where (1, 2) is a macro-corridor edge (weight 3, is_corridor=True)
        # and (2, 3), (3, 1) are atomic edges (weight 1). All bidirectional.
        rooms = {1: _make_room(1), 2: _make_room(2), 3: _make_room(3)}
        corridor = CorridorSegment(
            start_vnum=1,
            end_vnum=2,
            direction=int(Direction.north),
            intermediate_vnums=(100, 101),
        )
        edges = [
            LayoutEdge(
                src=1,
                dst=2,
                direction=int(Direction.north),
                weight=3,
                metadata={"corridor": corridor, "is_corridor": True},
            ),
            LayoutEdge(src=2, dst=3, direction=int(Direction.north), weight=1),
            LayoutEdge(src=3, dst=2, direction=int(Direction.south), weight=1),
            LayoutEdge(src=3, dst=1, direction=int(Direction.north), weight=1),
            LayoutEdge(src=1, dst=3, direction=int(Direction.south), weight=1),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)
        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        # Macro corridor (1, 2) must not be severed; an atomic edge (2, 3) or (1, 3) must be severed
        severed_pairs = {(min(e.src, e.dst), max(e.src, e.dst)) for e in simplified.severed_chords}
        assert (1, 2) not in severed_pairs

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
        """In midgaard.are, WarpSeverer severs one-way warp to 1219 while Poor Alley / Common Square remain connected."""
        midgaard_path = Path("tests/fixtures/areas/midgaard.are")
        area = parse_file(midgaard_path)
        world_graph = WorldGraph.from_area_data(area)
        layout_graph = LayoutGraph.from_world_graph(world_graph)

        severer = WarpSeverer()
        simplified = severer.simplify(layout_graph)

        # Midgaard has a warp converging on external node 1219, severed at the one-way exit
        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.dst == 1219
        assert severed.src in (3150, 3161, 3170)
        assert severed.cut is True

        # Street connection between Eastern End of Poor Alley (3024) and The Common Square (3025)
        edges_3024_3025 = [
            e
            for e in simplified.graph.edges
            if (e.src, e.dst) in ((3024, 3025), (3025, 3024))
        ]
        assert len(edges_3024_3025) == 2
        for edge in edges_3024_3025:
            assert edge.cut is False

        # Solve layout and verify on-axis alignment between 3024 and 3025 along East
        solver = HierarchicalLayoutSolver.production()
        solution = solver.solve(world_graph, LayoutOptions(timeout_seconds=30))

        c3024 = solution.coordinates[3024]
        c3025 = solution.coordinates[3025]
        assert c3025.x - c3024.x >= 1
        assert c3025.y - c3024.y == 0
        assert c3025.z - c3024.z == 0
