"""Comprehensive unit tests for non-Euclidean warp cycle detection and minimal chord severing."""

from __future__ import annotations

from pathlib import Path

import pytest

from romutil.layout import (
    ChainedSimplifier,
    CorridorSegment,
    GraphSimplifier,
    HallwayCompactor,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
    WarpSeverer,
    direction_to_displacement,
    is_euclidean_satisfiable,
)
from romutil.models import Direction, Exit, Room
from romutil.parser import parse_file
from romutil.types import LayoutOptions, WorldGraph


def _make_room(vnum: int, exits: list[Exit] | None = None) -> Room:
    """Helper to construct a Room with given exits."""
    return Room(
        vnum=vnum,
        name=f"Room {vnum}",
        description="",
        exits=tuple(exits or []),
    )


class TestDirectionToDisplacement:
    """Unit tests for direction_to_displacement."""

    def test_all_enum_directions(self) -> None:
        expected = {
            Direction.north: (0, 1, 0),
            Direction.south: (0, -1, 0),
            Direction.east: (1, 0, 0),
            Direction.west: (-1, 0, 0),
            Direction.up: (0, 0, 1),
            Direction.down: (0, 0, -1),
            Direction.northeast: (1, 1, 0),
            Direction.northwest: (-1, 1, 0),
            Direction.southeast: (1, -1, 0),
            Direction.southwest: (-1, -1, 0),
        }
        for direction, offset in expected.items():
            assert direction_to_displacement(direction) == offset
            assert direction_to_displacement(int(direction)) == offset

    def test_invalid_direction_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            direction_to_displacement(-1)
        with pytest.raises(ValueError):
            direction_to_displacement(10)
        with pytest.raises(ValueError):
            direction_to_displacement(99)


class TestIsEuclideanSatisfiable:
    """Unit tests for is_euclidean_satisfiable."""

    def test_empty_cycle(self) -> None:
        assert is_euclidean_satisfiable([]) is False

    def test_single_vector(self) -> None:
        assert is_euclidean_satisfiable([(0, 0, 0)]) is True
        assert is_euclidean_satisfiable([(0, 1, 0)]) is False
        assert is_euclidean_satisfiable([(0, -1, 0)]) is False
        assert is_euclidean_satisfiable([(1, 0, 0)]) is False
        assert is_euclidean_satisfiable([(-1, 0, 0)]) is False
        assert is_euclidean_satisfiable([(0, 0, 1)]) is False
        assert is_euclidean_satisfiable([(0, 0, -1)]) is False

    def test_standard_euclidean_square(self) -> None:
        """Square (N, E, S, W): closes with uniform lengths L_e = 1."""
        square = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
        assert is_euclidean_satisfiable(square) is True

    def test_elastic_rectangle(self) -> None:
        """Elastic rectangle (2 North, 1 East, 1 South, 1 West): stretch South to 2."""
        elastic = [(0, 1, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
        assert is_euclidean_satisfiable(elastic) is True

    def test_non_euclidean_loop_4_north(self) -> None:
        """4 North exits returning to start: purely positive motion on Y."""
        loop = [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
        assert is_euclidean_satisfiable(loop) is False

    def test_3d_warp_loop_pure_up(self) -> None:
        """3D warp loop (pure Up exits forming cycle): purely positive motion on Z."""
        loop = [(0, 0, 1), (0, 0, 1), (0, 0, 1)]
        assert is_euclidean_satisfiable(loop) is False

    def test_axis_aligned_negative_motion_only(self) -> None:
        """Pure negative motion along axes must fail."""
        assert is_euclidean_satisfiable([(0, -1, 0), (0, -1, 0)]) is False
        assert is_euclidean_satisfiable([(-1, 0, 0), (-1, 0, 0)]) is False
        assert is_euclidean_satisfiable([(0, 0, -1), (0, 0, -1)]) is False

    def test_axis_aligned_stretchable_3d(self) -> None:
        """Cycle with 2 Up, 1 Down can stretch Down to 2 to close."""
        vectors = [(0, 0, 1), (0, 0, 1), (0, 0, -1)]
        assert is_euclidean_satisfiable(vectors) is True

    def test_diagonal_satisfiable_cycles(self) -> None:
        """Diagonal loops that close directly or stretch."""
        # Simple diagonal reciprocal
        assert is_euclidean_satisfiable([(1, 1, 0), (-1, -1, 0)]) is True
        # Diagonal couple: NE (1, 1, 0), NW (-1, 1, 0), S (0, -1, 0)
        # Stretches S to 2: (1, 1, 0) + (-1, 1, 0) + 2 * (0, -1, 0) = (0, 0, 0)
        assert is_euclidean_satisfiable([(1, 1, 0), (-1, 1, 0), (0, -1, 0)]) is True
        # NE, NW, 2 S
        assert is_euclidean_satisfiable([(1, 1, 0), (-1, 1, 0), (0, -1, 0), (0, -1, 0)]) is True

    def test_diagonal_unsatisfiable_cycles(self) -> None:
        """Diagonal cycles where axes couple and require non-positive or impossible lengths."""
        # NE, NW, SE: requires L_1 = 0, so no positive solution exists
        assert is_euclidean_satisfiable([(1, 1, 0), (-1, 1, 0), (1, -1, 0)]) is False
        # Vectors strictly inside halfspace x + y > 0
        halfspace = [(1, 0, 0), (0, 1, 0), (-1, 2, 0), (2, -1, 0)]
        assert is_euclidean_satisfiable(halfspace) is False


class TestWarpSeverer:
    """Unit tests for WarpSeverer simplifier."""

    def test_protocol_conformance(self) -> None:
        severer = WarpSeverer()
        assert isinstance(severer, GraphSimplifier)

    def test_empty_graph(self) -> None:
        severer = WarpSeverer()
        empty_lg = LayoutGraph(rooms={}, edges=())
        simplified = severer.simplify(empty_lg, LayoutOptions())
        assert simplified.graph.rooms == {}
        assert simplified.graph.edges == ()
        assert simplified.severed_chords == ()

    def test_graph_without_edges(self) -> None:
        severer = WarpSeverer()
        rooms = {1: _make_room(1), 2: _make_room(2)}
        lg = LayoutGraph(rooms=rooms, edges=())
        simplified = severer.simplify(lg)
        assert simplified.graph.rooms == rooms
        assert simplified.graph.edges == ()
        assert simplified.severed_chords == ()

    def test_non_euclidean_self_loop(self) -> None:
        """Self loop with North exit must be severed."""
        rooms = {1: _make_room(1, [Exit(direction=0, dst_vnum=1)])}
        edges = [LayoutEdge(src=1, dst=1, direction=0)]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].src == 1
        assert simplified.severed_chords[0].dst == 1
        assert simplified.severed_chords[0].cut is True
        assert len(simplified.graph.edges) == 0

    def test_self_loop_with_invalid_direction_severed(self) -> None:
        """Self loop with invalid direction must also be severed."""
        rooms = {1: _make_room(1, [Exit(direction=99, dst_vnum=1)])}
        edges = [LayoutEdge(src=1, dst=1, direction=99)]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].cut is True

    def test_standard_euclidean_square_no_cuts(self) -> None:
        """Standard Euclidean square (N, E, S, W): 0 chords cut."""
        # 1 -> 2 (East), 2 -> 3 (North), 3 -> 4 (West), 4 -> 1 (South)
        rooms = {
            1: _make_room(1, [Exit(direction=1, dst_vnum=2)]),
            2: _make_room(2, [Exit(direction=0, dst_vnum=3)]),
            3: _make_room(3, [Exit(direction=3, dst_vnum=4)]),
            4: _make_room(4, [Exit(direction=2, dst_vnum=1)]),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=1),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=4, direction=3),
            LayoutEdge(src=4, dst=1, direction=2),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg, LayoutOptions())

        assert len(simplified.severed_chords) == 0
        assert len(simplified.graph.edges) == 4

    def test_elastic_rectangle_no_cuts(self) -> None:
        """Elastic rectangle (2 North, 1 East, 1 South, 1 West): 0 chords cut."""
        # 1 -> 2 (North), 2 -> 3 (North), 3 -> 4 (East), 4 -> 5 (South), 5 -> 1 (West)
        rooms = {
            1: _make_room(1, [Exit(direction=0, dst_vnum=2)]),
            2: _make_room(2, [Exit(direction=0, dst_vnum=3)]),
            3: _make_room(3, [Exit(direction=1, dst_vnum=4)]),
            4: _make_room(4, [Exit(direction=2, dst_vnum=5)]),
            5: _make_room(5, [Exit(direction=3, dst_vnum=1)]),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=4, direction=1),
            LayoutEdge(src=4, dst=5, direction=2),
            LayoutEdge(src=5, dst=1, direction=3),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 0
        assert len(simplified.graph.edges) == 5

    def test_non_euclidean_loop_4_north_severed(self) -> None:
        """Non-Euclidean loop (4 North exits returning to start): exactly 1 chord severed."""
        # 1 -> 2 (North), 2 -> 3 (North), 3 -> 4 (North), 4 -> 1 (North)
        rooms = {
            1: _make_room(1, [Exit(direction=0, dst_vnum=2)]),
            2: _make_room(2, [Exit(direction=0, dst_vnum=3)]),
            3: _make_room(3, [Exit(direction=0, dst_vnum=4)]),
            4: _make_room(4, [Exit(direction=0, dst_vnum=1)]),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=4, direction=0),
            LayoutEdge(src=4, dst=1, direction=0),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.cut is True
        assert len(simplified.graph.edges) == 3
        # The remaining edges must form a path (no cycles left)
        assert severed not in simplified.graph.edges

    def test_3d_warp_loop_pure_up_severed(self) -> None:
        """3D warp loop (pure Up exits forming cycle): exactly 1 chord severed."""
        # 1 -> 2 (Up), 2 -> 3 (Up), 3 -> 1 (Up)
        rooms = {
            1: _make_room(1, [Exit(direction=4, dst_vnum=2)]),
            2: _make_room(2, [Exit(direction=4, dst_vnum=3)]),
            3: _make_room(3, [Exit(direction=4, dst_vnum=1)]),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=4),
            LayoutEdge(src=2, dst=3, direction=4),
            LayoutEdge(src=3, dst=1, direction=4),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.cut is True
        assert len(simplified.graph.edges) == 2

    def test_sever_prefers_non_compacted_edge(self) -> None:
        """WarpSeverer should prefer severing a weight-1 edge over a compacted macro-edge."""
        # Triangle 1 -> 2 (weight 3), 2 -> 3 (weight 1), 3 -> 1 (weight 1), all pure North
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=0, weight=3),
            LayoutEdge(src=2, dst=3, direction=0, weight=1),
            LayoutEdge(src=3, dst=1, direction=0, weight=1),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        # Should have severed one of the weight=1 edges, not the weight=3 edge
        assert severed.weight == 1
        assert (severed.src, severed.dst) != (1, 2)

    def test_sever_prefers_peripheral_low_degree_nodes(self) -> None:
        """WarpSeverer should prefer severing a peripheral low-degree chord over higher-degree nodes."""
        # Cycle: 1 - 2 - 3 - 4 - 1 (all pure North)
        # Hubs: 1 and 2 also connect to external nodes 10, 20 (deg 3)
        # Nodes 3 and 4 have degree 2
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
            4: _make_room(4),
            10: _make_room(10),
            20: _make_room(20),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=4, direction=0),
            LayoutEdge(src=4, dst=1, direction=0),
            # Branches on 1 and 2
            LayoutEdge(src=1, dst=10, direction=1),
            LayoutEdge(src=2, dst=20, direction=1),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        # (3, 4) has degrees 2+2=4, whereas other cycle edges have 3+2=5 or 3+3=6
        assert {severed.src, severed.dst} == {3, 4}

    def test_bidirectional_loop_severing(self) -> None:
        """Reciprocal edges between endpoints of severed chord are both severed."""
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
        }
        # Non-Euclidean pure Up in one direction, pure Down in other
        edges = [
            LayoutEdge(src=1, dst=2, direction=4),
            LayoutEdge(src=2, dst=1, direction=5),
            LayoutEdge(src=2, dst=3, direction=4),
            LayoutEdge(src=3, dst=2, direction=5),
            LayoutEdge(src=3, dst=1, direction=4),
            LayoutEdge(src=1, dst=3, direction=5),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        # One chord pair is severed, which contains 2 directed edges
        assert len(simplified.severed_chords) == 2
        for severed in simplified.severed_chords:
            assert severed.cut is True
            assert severed not in simplified.graph.edges

    def test_reverse_edge_traversal(self) -> None:
        """Cycle where one edge is traversed against its directed orientation."""
        # 1 -> 2 (North), 2 -> 3 (North), 1 -> 3 (South)
        # Undirected cycle: 1 - 2 - 3 - 1
        # Tracing 1 -> 2 (N), 2 -> 3 (N), 3 -> 1 (reverse of 1 -> 3 which is S, so -S = N)
        # Cycle is pure North -> non-Euclidean warp!
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=1, dst=3, direction=2),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].cut is True

    def test_cycle_with_invalid_direction_skipped(self) -> None:
        """A cycle with an invalid direction is safely skipped."""
        rooms = {1: _make_room(1), 2: _make_room(2), 3: _make_room(3)}
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=99),  # invalid direction
            LayoutEdge(src=3, dst=1, direction=0),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        # Cycle skipped, 0 severed chords
        assert len(simplified.severed_chords) == 0

    def test_multiple_non_euclidean_warps(self) -> None:
        """Two disjoint non-Euclidean loops both have a chord severed."""
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
            10: _make_room(10),
            11: _make_room(11),
            12: _make_room(12),
        }
        edges = [
            # Loop 1: pure North on 1, 2, 3
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=1, direction=0),
            # Loop 2: pure Up on 10, 11, 12
            LayoutEdge(src=10, dst=11, direction=4),
            LayoutEdge(src=11, dst=12, direction=4),
            LayoutEdge(src=12, dst=10, direction=4),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        assert len(simplified.severed_chords) == 2
        severed_nodes = [{e.src, e.dst} for e in simplified.severed_chords]
        # One from loop 1, one from loop 2
        assert any(nodes.issubset({1, 2, 3}) for nodes in severed_nodes)
        assert any(nodes.issubset({10, 11, 12}) for nodes in severed_nodes)

    def test_sever_prefers_parallel_convergent_exits_over_through_path(self) -> None:
        """WarpSeverer prioritizes severing parallel-convergent one-way exits entering a common sink."""
        # Collinear through-path: 1 -> 2 (South), 2 -> 3 (South)
        # Parallel drop exits: 1 -> 99 (West), 2 -> 99 (West), 3 -> 99 (West)
        # Non-Euclidean cycles are created: [1, 2, 99], [2, 3, 99].
        # The parallel entries into 99 must be severed instead of through-path edges (1, 2) or (2, 3).
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
            99: _make_room(99),
        }
        edges = [
            LayoutEdge(src=1, dst=2, direction=2),
            LayoutEdge(src=2, dst=3, direction=2),
            LayoutEdge(src=1, dst=99, direction=3),
            LayoutEdge(src=2, dst=99, direction=3),
            LayoutEdge(src=3, dst=99, direction=3),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        severer = WarpSeverer()
        simplified = severer.simplify(lg)

        active_pairs = {(e.src, e.dst) for e in simplified.graph.edges}
        assert (1, 2) in active_pairs
        assert (2, 3) in active_pairs

        severed_pairs = {(e.src, e.dst) for e in simplified.severed_chords}
        assert (1, 99) in severed_pairs or (2, 99) in severed_pairs or (3, 99) in severed_pairs
        assert len(simplified.severed_chords) == 2

    def test_sewer_area_parallel_convergent_pit_chords(self) -> None:
        """Verify parallel drop exits into pit sink 7114 are severed, preserving the eastern ledge."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "areas"
        area = parse_file(fixtures_dir / "sewer.are")
        world_graph = WorldGraph.from_area_data(area)
        layout_graph = LayoutGraph.from_world_graph(world_graph)

        severer = WarpSeverer()
        simplified = severer.simplify(layout_graph)

        severed_pairs = {(c.src, c.dst) for c in simplified.severed_chords}
        assert (7117, 7114) in severed_pairs
        assert (7118, 7114) in severed_pairs
        assert (7119, 7114) in severed_pairs

        active_pairs = {(e.src, e.dst) for e in simplified.graph.edges}
        assert (7116, 7117) in active_pairs
        assert (7117, 7118) in active_pairs
        assert (7118, 7119) in active_pairs
        assert (7119, 7120) in active_pairs


class TestChainedSimplifier:
    """Unit tests for ChainedSimplifier."""

    def test_protocol_conformance(self) -> None:
        chain = ChainedSimplifier([])
        assert isinstance(chain, GraphSimplifier)

    def test_empty_chain(self) -> None:
        chain = ChainedSimplifier([])
        rooms = {1: _make_room(1)}
        lg = LayoutGraph(rooms=rooms, edges=())
        simplified = chain.simplify(lg)
        assert simplified.graph.rooms == rooms
        assert simplified.corridors == ()
        assert simplified.severed_chords == ()

    def test_chain_hallways_and_warps(self) -> None:
        """Combines HallwayCompactor and WarpSeverer: corridors and severed chords both populated."""
        # Corridor: 1 <-> 2 <-> 3 (North / South, room 2 is degree-2 collinear)
        # Warp loop: 10 -> 11 -> 12 -> 10 (pure Up)
        rooms = {
            1: _make_room(1, [Exit(direction=0, dst_vnum=2)]),
            2: _make_room(2, [Exit(direction=2, dst_vnum=1), Exit(direction=0, dst_vnum=3)]),
            3: _make_room(3, [Exit(direction=2, dst_vnum=2)]),
            10: _make_room(10, [Exit(direction=4, dst_vnum=11)]),
            11: _make_room(11, [Exit(direction=4, dst_vnum=12)]),
            12: _make_room(12, [Exit(direction=4, dst_vnum=10)]),
        }
        edges = [
            # Hallway edges
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
            # Warp loop edges
            LayoutEdge(src=10, dst=11, direction=4),
            LayoutEdge(src=11, dst=12, direction=4),
            LayoutEdge(src=12, dst=10, direction=4),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges)

        chain = ChainedSimplifier([HallwayCompactor(), WarpSeverer()])
        simplified = chain.simplify(lg, LayoutOptions())

        # Verify corridors populated by HallwayCompactor
        assert len(simplified.corridors) == 1
        corridor = simplified.corridors[0]
        assert corridor.start_vnum == 1
        assert corridor.end_vnum == 3
        assert corridor.intermediate_vnums == (2,)

        # Verify room 2 collapsed from simplified graph
        assert 2 not in simplified.graph.rooms
        assert set(simplified.graph.rooms.keys()) == {1, 3, 10, 11, 12}

        # Verify severed chords populated by WarpSeverer
        assert len(simplified.severed_chords) == 1
        severed = simplified.severed_chords[0]
        assert severed.cut is True
        assert {severed.src, severed.dst}.issubset({10, 11, 12})
        assert severed not in simplified.graph.edges


class TestWarpSevererVirtualAndCanonicalCycles:
    """Verify WarpSeverer virtual-room deduplication, canonical cycle basis, and sentinel handling."""

    def test_virtual_room_dedup_keeps_one_per_direction_and_preserves_virtual_rooms(self) -> None:
        """Hybrid incoming edges (1 --N--> 9000, 2 --E--> 9000, 3 --E--> 9000) keep 1->9000 and best E->9000."""
        rooms = {
            1: _make_room(1),
            2: _make_room(2),
            3: _make_room(3),
            9000: _make_room(9000),
        }
        edges = [
            LayoutEdge(src=1, dst=9000, direction=int(Direction.north)),
            LayoutEdge(src=2, dst=9000, direction=int(Direction.east)),
            LayoutEdge(src=3, dst=9000, direction=int(Direction.east)),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges, virtual_rooms=frozenset({9000}))

        simplified = WarpSeverer().simplify(lg)
        assert simplified.graph.virtual_rooms == frozenset({9000})
        active = {(e.src, e.dst, e.direction) for e in simplified.graph.edges if not e.cut}
        assert (1, 9000, int(Direction.north)) in active
        assert (2, 9000, int(Direction.east)) in active
        assert (3, 9000, int(Direction.east)) not in active
        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].src == 3
        assert simplified.severed_chords[0].dst == 9000
        assert simplified.severed_chords[0].cut is True

    def test_virtual_room_dedup_prioritizes_degree_and_weight(self) -> None:
        """Higher-weight and higher-degree sources win over lower-vnum leaves when deduplicating."""
        rooms = {i: _make_room(i) for i in (1, 2, 3, 20, 21, 9000)}
        edges = [
            # Source 2 has higher degree (20, 21) than source 1
            LayoutEdge(src=1, dst=9000, direction=int(Direction.east), weight=1),
            LayoutEdge(src=2, dst=9000, direction=int(Direction.east), weight=1),
            LayoutEdge(src=2, dst=20, direction=int(Direction.south)),
            LayoutEdge(src=2, dst=21, direction=int(Direction.north)),
            # Source 3 has higher weight than source 2
            LayoutEdge(src=3, dst=9000, direction=int(Direction.east), weight=3),
        ]
        lg = LayoutGraph(rooms=rooms, edges=edges, virtual_rooms=frozenset({9000}))
        simplified = WarpSeverer().simplify(lg)

        kept_east = [
            e for e in simplified.graph.edges
            if e.dst == 9000 and e.direction == int(Direction.east) and not e.cut
        ]
        assert len(kept_east) == 1
        assert kept_east[0].src == 3
        severed_srcs = {e.src for e in simplified.severed_chords}
        assert severed_srcs == {1, 2}

    def test_canonical_cycle_basis_invariant_under_added_articulation_wing(self) -> None:
        """Attaching a low-VNUM biconnected articulation wing does not alter warp cuts in another block."""
        # Biconnected warp block on {10, 11, 12, 13}: 10 <-> 11 <-> 12 <-> 13 (Euclidean square) + chord 10 -> 12 (North)
        base_rooms = {v: _make_room(v) for v in (10, 11, 12, 13)}
        base_edges = [
            LayoutEdge(src=10, dst=11, direction=int(Direction.north)),
            LayoutEdge(src=11, dst=10, direction=int(Direction.south)),
            LayoutEdge(src=11, dst=12, direction=int(Direction.east)),
            LayoutEdge(src=12, dst=11, direction=int(Direction.west)),
            LayoutEdge(src=12, dst=13, direction=int(Direction.south)),
            LayoutEdge(src=13, dst=12, direction=int(Direction.north)),
            LayoutEdge(src=13, dst=10, direction=int(Direction.west)),
            LayoutEdge(src=10, dst=13, direction=int(Direction.east)),
            # Non-Euclidean one-way chord creating warp cycles inside {10, 11, 12, 13}
            LayoutEdge(src=10, dst=12, direction=int(Direction.north)),
        ]
        base_lg = LayoutGraph(rooms=base_rooms, edges=base_edges)
        base_cuts = {
            (e.src, e.dst, e.direction)
            for e in WarpSeverer().simplify(base_lg).severed_chords
        }
        assert len(base_cuts) >= 1

        # Attach an unrelated Euclidean biconnected wing with very low VNUMs {1, 2, 3, 4} at articulation 10
        wing_rooms = {**base_rooms, **{v: _make_room(v) for v in (1, 2, 3, 4)}}
        wing_edges = list(base_edges) + [
            LayoutEdge(src=10, dst=1, direction=int(Direction.down)),
            LayoutEdge(src=1, dst=10, direction=int(Direction.up)),
            LayoutEdge(src=1, dst=2, direction=int(Direction.north)),
            LayoutEdge(src=2, dst=1, direction=int(Direction.south)),
            LayoutEdge(src=2, dst=3, direction=int(Direction.east)),
            LayoutEdge(src=3, dst=2, direction=int(Direction.west)),
            LayoutEdge(src=3, dst=4, direction=int(Direction.south)),
            LayoutEdge(src=4, dst=3, direction=int(Direction.north)),
            LayoutEdge(src=4, dst=1, direction=int(Direction.west)),
            LayoutEdge(src=1, dst=4, direction=int(Direction.east)),
        ]
        # Reverse insertion order to verify order independence as well
        wing_lg = LayoutGraph(rooms=wing_rooms, edges=list(reversed(wing_edges)))
        wing_cuts = {
            (e.src, e.dst, e.direction)
            for e in WarpSeverer().simplify(wing_lg).severed_chords
        }
        assert wing_cuts == base_cuts

    def test_negative_sentinel_exits_do_not_sever_cross_area_oneway_chain(self) -> None:
        """Shared -1 sentinel exits across disconnected subgraphs do not sever a one-way connecting chain."""
        # Area A (100 <-> 101 with -1 stub), Area B (200 <-> 201 with -1 stub), linked via 101 -> 300 -> 201
        r100 = _make_room(100, [Exit(Direction.east, 101)])
        r101 = _make_room(
            101,
            [
                Exit(Direction.west, 100),
                Exit(Direction.north, -1),
                Exit(Direction.down, 300),
            ],
        )
        r300 = _make_room(300, [Exit(Direction.down, 201)])
        r201 = _make_room(
            201,
            [
                Exit(Direction.east, 200),
                Exit(Direction.west, -1),
            ],
        )
        r200 = _make_room(200, [Exit(Direction.west, 201)])

        rooms = {r.vnum: r for r in (r100, r101, r300, r201, r200)}
        wg = WorldGraph(rooms=rooms, exits=tuple(ex for r in rooms.values() for ex in r.exits))
        lg = LayoutGraph.from_world_graph(wg)

        simplified = WarpSeverer().simplify(lg)
        assert simplified.severed_chords == ()
        active_pairs = {(e.src, e.dst) for e in simplified.graph.edges if not e.cut}
        assert (101, 300) in active_pairs
        assert (300, 201) in active_pairs


class TestGlobalAxisSccWarpDetection:
    """Verify global per-axis Tarjan SCC warp verification and single-pair parallel edge cuts."""

    def test_global_scc_z_axis_warp_missed_by_cycle_basis(self) -> None:
        """Detects and severs a multi-cycle Z-axis contradiction whose basis cycles are each satisfiable."""
        import networkx as nx

        from romutil.layout.simplification.warps import (
            _extract_canonical_cycle_basis,
            _find_global_axis_warp_cycle,
        )

        rooms = {v: _make_room(v) for v in (1, 2, 3, 4, 5, 6)}
        edges = (
            LayoutEdge(src=1, dst=2, direction=int(Direction.up)),
            LayoutEdge(src=2, dst=3, direction=int(Direction.up)),
            LayoutEdge(src=3, dst=1, direction=int(Direction.down)),
            LayoutEdge(src=1, dst=4, direction=int(Direction.up)),
            LayoutEdge(src=4, dst=5, direction=int(Direction.north)),
            LayoutEdge(src=5, dst=3, direction=int(Direction.south)),
            LayoutEdge(src=2, dst=6, direction=int(Direction.east)),
            LayoutEdge(src=6, dst=4, direction=int(Direction.west)),
        )
        g = nx.Graph()
        g.add_nodes_from(rooms)
        emap: dict[tuple[int, int], list[LayoutEdge]] = {}
        for edge in edges:
            g.add_edge(edge.src, edge.dst)
            emap.setdefault((min(edge.src, edge.dst), max(edge.src, edge.dst)), []).append(edge)

        basis_cycles = _extract_canonical_cycle_basis(g)
        assert len(basis_cycles) == 3
        for cycle in basis_cycles:
            vecs = []
            for idx in range(len(cycle)):
                u, v = cycle[idx], cycle[(idx + 1) % len(cycle)]
                es = emap[(min(u, v), max(u, v))]
                fwd = [e for e in es if e.src == u and e.dst == v]
                chosen = fwd[0] if fwd else es[0]
                disp = direction_to_displacement(chosen.direction)
                if not fwd:
                    disp = (-disp[0], -disp[1], -disp[2])
                vecs.append(disp)
            assert is_euclidean_satisfiable(vecs) is True

        scc_cycle = _find_global_axis_warp_cycle(rooms, edges)
        assert scc_cycle is not None
        assert len(scc_cycle) == 5

        lg = LayoutGraph(rooms=rooms, edges=edges)
        simplified = WarpSeverer().simplify(lg)
        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].cut is True
        assert _find_global_axis_warp_cycle(rooms, simplified.graph.edges) is None

    def test_global_scc_x_and_y_axis_warps_missed_by_cycle_basis(self) -> None:
        """Detects and resolves multi-cycle X-axis and Y-axis SCC warps missed by single-tree cycle bases."""
        from romutil.layout.simplification.warps import _find_global_axis_warp_cycle

        rooms = {v: _make_room(v) for v in (1, 2, 3, 4, 5, 6)}
        configs = [
            (Direction.east, Direction.west, Direction.north, Direction.south, Direction.up, Direction.down),
            (Direction.north, Direction.south, Direction.east, Direction.west, Direction.up, Direction.down),
        ]
        for d_pos, d_neg, a_pos, a_neg, b_pos, b_neg in configs:
            edges = (
                LayoutEdge(src=1, dst=2, direction=int(d_pos)),
                LayoutEdge(src=2, dst=3, direction=int(d_pos)),
                LayoutEdge(src=3, dst=1, direction=int(d_neg)),
                LayoutEdge(src=1, dst=4, direction=int(d_pos)),
                LayoutEdge(src=4, dst=5, direction=int(a_pos)),
                LayoutEdge(src=5, dst=3, direction=int(a_neg)),
                LayoutEdge(src=2, dst=6, direction=int(b_pos)),
                LayoutEdge(src=6, dst=4, direction=int(b_neg)),
            )
            assert _find_global_axis_warp_cycle(rooms, edges) is not None
            simplified = WarpSeverer().simplify(LayoutGraph(rooms=rooms, edges=edges))
            assert len(simplified.severed_chords) == 1
            assert _find_global_axis_warp_cycle(rooms, simplified.graph.edges) is None

    def test_global_scc_parallel_multi_edge_severs_single_edge_and_preserves_connectivity(self) -> None:
        """Parallel multi-edge contradictions cut only one conflicting edge, keeping endpoints connected."""
        from romutil.layout.simplification.warps import _find_global_axis_warp_cycle

        rooms = {1: _make_room(1), 2: _make_room(2), 3: _make_room(3)}
        # Parallel 1 -> 2 (north) and 1 -> 2 (up) form a 2-edge Y/Z contradiction with zero graph cycles.
        # An additional invalid-direction edge and an extra tree branch 2 -> 3 verify clean handling.
        e_corridor = LayoutEdge(
            src=1,
            dst=2,
            direction=int(Direction.north),
            weight=2,
            metadata={"is_corridor": True},
        )
        e_conflict = LayoutEdge(src=1, dst=2, direction=int(Direction.up), weight=1)
        e_invalid = LayoutEdge(src=2, dst=3, direction=999, weight=1)
        e_branch = LayoutEdge(src=2, dst=3, direction=int(Direction.east), weight=1)

        lg = LayoutGraph(
            rooms=rooms,
            edges=(e_corridor, e_conflict, e_invalid, e_branch),
        )
        assert _find_global_axis_warp_cycle(rooms, lg.edges) is not None

        simplified = WarpSeverer().simplify(lg)
        assert len(simplified.severed_chords) == 1
        assert simplified.severed_chords[0].src == 1
        assert simplified.severed_chords[0].dst == 2
        assert simplified.severed_chords[0].direction == int(Direction.up)
        assert simplified.severed_chords[0].cut is True

        active_1_2 = [e for e in simplified.graph.edges if {e.src, e.dst} == {1, 2} and not e.cut]
        assert len(active_1_2) == 1
        assert active_1_2[0].direction == int(Direction.north)
        assert _find_global_axis_warp_cycle(rooms, simplified.graph.edges) is None
