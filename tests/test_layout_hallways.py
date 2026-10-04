"""Comprehensive unit tests for degree-2 collinear hallway compaction and expansion."""

from __future__ import annotations

import pytest

from romutil.layout import (
    CorridorSegment,
    GraphSimplifier,
    HallwayCompactor,
    HierarchicalLayoutSolver,
    IdentityLayoutAssembler,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeBuilder,
    TreeSolution,
    TreeSolver,
    TrivialTreeBuilder,
    expand_corridor_coordinates,
)
from romutil.models import AreaHeader, ExitDef, RoomDef
from romutil.types import (
    Coord3D,
    LayoutOptions,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


def _make_bidirectional_corridor(
    vnums: list[int],
    direction_forward: int = 0,
    direction_backward: int = 2,
) -> tuple[dict[int, RoomDef], list[LayoutEdge]]:
    """Helper to construct bidirectional linear chain of rooms."""
    rooms: dict[int, RoomDef] = {}
    edges: list[LayoutEdge] = []

    for i, vnum in enumerate(vnums):
        exits: list[ExitDef] = []
        if i > 0:
            exits.append(ExitDef(direction=direction_backward, dst_vnum=vnums[i - 1]))
            edges.append(
                LayoutEdge(
                    src=vnum,
                    dst=vnums[i - 1],
                    direction=direction_backward,
                    weight=1,
                )
            )
        if i < len(vnums) - 1:
            exits.append(ExitDef(direction=direction_forward, dst_vnum=vnums[i + 1]))
            edges.append(
                LayoutEdge(
                    src=vnum,
                    dst=vnums[i + 1],
                    direction=direction_forward,
                    weight=1,
                )
            )
        rooms[vnum] = RoomDef(vnum=vnum, name=f"Room {vnum}", description="", exits=tuple(exits))

    return rooms, edges


class TestHallwayCompactor:
    """Unit tests for HallwayCompactor simplification stage."""

    def test_protocol_conformance(self) -> None:
        compactor = HallwayCompactor()
        assert isinstance(compactor, GraphSimplifier)

    def test_empty_graph(self) -> None:
        compactor = HallwayCompactor()
        empty_lg = LayoutGraph(rooms={}, edges=())
        simplified = compactor.simplify(empty_lg, LayoutOptions())
        assert simplified.graph.rooms == {}
        assert simplified.graph.edges == ()
        assert simplified.corridors == ()

    def test_straight_chain_compaction_north(self) -> None:
        """1 <-> 2 <-> 3 North: room 2 contracted, macro-edge weight 2, direction North."""
        rooms, edges = _make_bidirectional_corridor([1, 2, 3], direction_forward=0, direction_backward=2)
        lg = LayoutGraph(rooms=rooms, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg, LayoutOptions())

        assert set(simplified.graph.rooms.keys()) == {1, 3}
        assert 2 not in simplified.graph.rooms

        # Macro-edge created with weight 2 and direction North (0)
        assert len(simplified.graph.edges) == 1
        macro_edge = simplified.graph.edges[0]
        assert macro_edge.src == 1
        assert macro_edge.dst == 3
        assert macro_edge.direction == 0
        assert macro_edge.weight == 2

        # CorridorSegment recorded
        assert len(simplified.corridors) == 1
        corridor = simplified.corridors[0]
        assert corridor.start_vnum == 1
        assert corridor.end_vnum == 3
        assert corridor.direction == 0
        assert corridor.intermediate_vnums == (2,)

    def test_longer_chain_compaction(self) -> None:
        """1 <-> 2 <-> 3 <-> 4 <-> 5 East: 2, 3, 4 contracted, macro-edge weight 4."""
        rooms, edges = _make_bidirectional_corridor([1, 2, 3, 4, 5], direction_forward=1, direction_backward=3)
        lg = LayoutGraph(rooms=rooms, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {1, 5}
        assert len(simplified.graph.edges) == 1
        macro_edge = simplified.graph.edges[0]
        assert macro_edge.src == 1
        assert macro_edge.dst == 5
        assert macro_edge.direction == 1
        assert macro_edge.weight == 4

        assert len(simplified.corridors) == 1
        corridor = simplified.corridors[0]
        assert corridor.start_vnum == 1
        assert corridor.end_vnum == 5
        assert corridor.direction == 1
        assert corridor.intermediate_vnums == (2, 3, 4)

    def test_reverse_vnum_canonical_orientation(self) -> None:
        """300 <-> 200 <-> 100 North: start is 100, end is 300, direction South."""
        # 300 forward (North) to 200, 200 forward (North) to 100
        rooms, edges = _make_bidirectional_corridor([300, 200, 100], direction_forward=0, direction_backward=2)
        lg = LayoutGraph(rooms=rooms, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {100, 300}
        assert len(simplified.corridors) == 1
        corridor = simplified.corridors[0]
        # start_vnum < end_vnum: 100 < 300, direction South (2)
        assert corridor.start_vnum == 100
        assert corridor.end_vnum == 300
        assert corridor.direction == 2
        assert corridor.intermediate_vnums == (200,)

    def test_corners_are_preserved(self) -> None:
        """Room 2 turning from North to East must NOT be compacted."""
        # 1 <-> 2 North/South, 2 <-> 3 East/West
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        r2 = RoomDef(
            vnum=2,
            name="R2 Corner",
            description="",
            exits=(
                ExitDef(direction=2, dst_vnum=1),
                ExitDef(direction=1, dst_vnum=3),
            ),
        )
        r3 = RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=3, dst_vnum=2),))

        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
            LayoutEdge(src=2, dst=3, direction=1),
            LayoutEdge(src=3, dst=2, direction=3),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        # Nothing should be compacted
        assert set(simplified.graph.rooms.keys()) == {1, 2, 3}
        assert simplified.corridors == ()
        assert len(simplified.graph.edges) == 4

    def test_junctions_are_not_compacted(self) -> None:
        """Room 2 with degree > 2 is a junction and must never be compacted."""
        # 2 connected to 1 (South), 3 (North), 4 (East)
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        r2 = RoomDef(
            vnum=2,
            name="R2 Junction",
            description="",
            exits=(
                ExitDef(direction=2, dst_vnum=1),
                ExitDef(direction=0, dst_vnum=3),
                ExitDef(direction=1, dst_vnum=4),
            ),
        )
        r3 = RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=2, dst_vnum=2),))
        r4 = RoomDef(vnum=4, name="R4", description="", exits=(ExitDef(direction=3, dst_vnum=2),))

        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
            LayoutEdge(src=2, dst=4, direction=1),
            LayoutEdge(src=4, dst=2, direction=3),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3, 4: r4}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {1, 2, 3, 4}
        assert simplified.corridors == ()

    def test_dead_ends_are_not_compacted(self) -> None:
        """Rooms of degree 1 (dead ends) are never compacted."""
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        r2 = RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=2, dst_vnum=1),))
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {1, 2}
        assert simplified.corridors == ()

    def test_one_way_exit_rejects_candidate(self) -> None:
        """One-way exits violate bidirectional hallway condition."""
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        # Room 2 exits to 3, but has no exit back to 1
        r2 = RoomDef(vnum=2, name="R2", description="", exits=(ExitDef(direction=0, dst_vnum=3),))
        r3 = RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=2, dst_vnum=2),))

        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {1, 2, 3}
        assert simplified.corridors == ()

    def test_inconsistent_reverse_direction_rejects_candidate(self) -> None:
        """Reciprocal edge has non-inverted direction."""
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        r2 = RoomDef(
            vnum=2,
            name="R2",
            description="",
            exits=(
                ExitDef(direction=1, dst_vnum=1),  # East instead of South!
                ExitDef(direction=0, dst_vnum=3),
            ),
        )
        r3 = RoomDef(vnum=3, name="R3", description="", exits=(ExitDef(direction=2, dst_vnum=2),))

        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=1),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {1, 2, 3}
        assert simplified.corridors == ()

    def test_invalid_direction_int_rejects_candidate(self) -> None:
        """Unrecognized direction integer outside [0, 9] is rejected."""
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=99, dst_vnum=2),))
        r2 = RoomDef(vnum=2, name="R2", description="", exits=())
        r3 = RoomDef(vnum=3, name="R3", description="", exits=())
        edges = [
            LayoutEdge(src=1, dst=2, direction=99),
            LayoutEdge(src=2, dst=1, direction=99),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)
        assert simplified.corridors == ()

    def test_external_neighbor_rejects_candidate(self) -> None:
        """Room 2 connected to an external room not in graph.rooms is not compacted."""
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        r2 = RoomDef(
            vnum=2,
            name="R2",
            description="",
            exits=(
                ExitDef(direction=2, dst_vnum=1),
                ExitDef(direction=0, dst_vnum=9999),  # 9999 is external
            ),
        )
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
            LayoutEdge(src=2, dst=9999, direction=0),
            LayoutEdge(src=9999, dst=2, direction=2),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)
        assert simplified.corridors == ()

    def test_room_with_self_loop_not_compacted(self) -> None:
        """Room with a self-loop (degree 2 with 1 neighbor in nx.Graph) is not compacted."""
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=1),))
        edges = [LayoutEdge(src=1, dst=1, direction=0)]
        lg = LayoutGraph(rooms={1: r1}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)
        assert simplified.corridors == ()

    def test_two_disjoint_chains(self) -> None:
        """Two separate hallways in the same graph are both compacted."""
        rooms1, edges1 = _make_bidirectional_corridor([1, 2, 3], direction_forward=0, direction_backward=2)
        rooms2, edges2 = _make_bidirectional_corridor([10, 20, 30], direction_forward=1, direction_backward=3)

        combined_rooms = {**rooms1, **rooms2}
        combined_edges = edges1 + edges2
        lg = LayoutGraph(rooms=combined_rooms, edges=combined_edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        assert set(simplified.graph.rooms.keys()) == {1, 3, 10, 30}
        assert len(simplified.corridors) == 2
        assert simplified.corridors[0].intermediate_vnums == (2,)
        assert simplified.corridors[1].intermediate_vnums == (20,)

    def test_two_chains_meeting_at_corner(self) -> None:
        """Chains meeting at corner room: corner is kept, intermediate rooms compacted."""
        # 1 <-> 2 <-> 3 North; 3 <-> 4 <-> 5 East
        r1 = RoomDef(vnum=1, name="R1", description="", exits=(ExitDef(direction=0, dst_vnum=2),))
        r2 = RoomDef(
            vnum=2,
            name="R2",
            description="",
            exits=(ExitDef(direction=2, dst_vnum=1), ExitDef(direction=0, dst_vnum=3)),
        )
        r3 = RoomDef(
            vnum=3,
            name="R3 Corner",
            description="",
            exits=(ExitDef(direction=2, dst_vnum=2), ExitDef(direction=1, dst_vnum=4)),
        )
        r4 = RoomDef(
            vnum=4,
            name="R4",
            description="",
            exits=(ExitDef(direction=3, dst_vnum=3), ExitDef(direction=1, dst_vnum=5)),
        )
        r5 = RoomDef(vnum=5, name="R5", description="", exits=(ExitDef(direction=3, dst_vnum=4),))

        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
            LayoutEdge(src=3, dst=4, direction=1),
            LayoutEdge(src=4, dst=3, direction=3),
            LayoutEdge(src=4, dst=5, direction=1),
            LayoutEdge(src=5, dst=4, direction=3),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3, 4: r4, 5: r5}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)

        # Rooms 2 and 4 compacted; 1, 3 (corner), 5 remain
        assert set(simplified.graph.rooms.keys()) == {1, 3, 5}
        assert len(simplified.corridors) == 2
        assert simplified.corridors[0].start_vnum == 1
        assert simplified.corridors[0].end_vnum == 3
        assert simplified.corridors[0].intermediate_vnums == (2,)

        assert simplified.corridors[1].start_vnum == 3
        assert simplified.corridors[1].end_vnum == 5
        assert simplified.corridors[1].intermediate_vnums == (4,)

    def test_pure_cycle_of_degree_2_nodes(self) -> None:
        """A closed loop of collinear degree-2 nodes has no non-collinear boundary."""
        # 1 <-> 2 <-> 3 <-> 1 (all North/South)
        r1 = RoomDef(
            vnum=1,
            name="R1",
            description="",
            exits=(ExitDef(direction=0, dst_vnum=2), ExitDef(direction=2, dst_vnum=3)),
        )
        r2 = RoomDef(
            vnum=2,
            name="R2",
            description="",
            exits=(ExitDef(direction=2, dst_vnum=1), ExitDef(direction=0, dst_vnum=3)),
        )
        r3 = RoomDef(
            vnum=3,
            name="R3",
            description="",
            exits=(ExitDef(direction=2, dst_vnum=2), ExitDef(direction=0, dst_vnum=1)),
        )
        edges = [
            LayoutEdge(src=1, dst=2, direction=0),
            LayoutEdge(src=2, dst=1, direction=2),
            LayoutEdge(src=2, dst=3, direction=0),
            LayoutEdge(src=3, dst=2, direction=2),
            LayoutEdge(src=3, dst=1, direction=0),
            LayoutEdge(src=1, dst=3, direction=2),
        ]
        lg = LayoutGraph(rooms={1: r1, 2: r2, 3: r3}, edges=edges)

        compactor = HallwayCompactor()
        simplified = compactor.simplify(lg)
        # Should not compact into nothingness
        assert simplified.corridors == ()


class TestCoordinateExpansion:
    """Unit tests for expand_corridor_coordinates interpolation."""

    def test_single_room_interpolation_north(self) -> None:
        corridor = CorridorSegment(start_vnum=1, end_vnum=3, direction=0, intermediate_vnums=(2,))
        coords = expand_corridor_coordinates(
            corridor=corridor,
            start_coord=Coord3D(0, 0, 0),
            end_coord=Coord3D(0, 2, 0),
        )
        assert coords == {2: Coord3D(0, 1, 0)}

    def test_multi_room_interpolation_east(self) -> None:
        corridor = CorridorSegment(
            start_vnum=10,
            end_vnum=50,
            direction=1,
            intermediate_vnums=(20, 30, 40),
        )
        coords = expand_corridor_coordinates(
            corridor=corridor,
            start_coord=Coord3D(10, 5, 0),
            end_coord=Coord3D(14, 5, 0),
        )
        assert coords == {
            20: Coord3D(11, 5, 0),
            30: Coord3D(12, 5, 0),
            40: Coord3D(13, 5, 0),
        }

    def test_multi_room_interpolation_vertical_down(self) -> None:
        corridor = CorridorSegment(
            start_vnum=100,
            end_vnum=400,
            direction=5,
            intermediate_vnums=(200, 300),
        )
        coords = expand_corridor_coordinates(
            corridor=corridor,
            start_coord=Coord3D(0, 0, 9),
            end_coord=Coord3D(0, 0, 6),
        )
        assert coords == {
            200: Coord3D(0, 0, 8),
            300: Coord3D(0, 0, 7),
        }

    def test_empty_corridor_segment(self) -> None:
        corridor = CorridorSegment(start_vnum=1, end_vnum=2, direction=0, intermediate_vnums=())
        coords = expand_corridor_coordinates(
            corridor=corridor,
            start_coord=Coord3D(0, 0, 0),
            end_coord=Coord3D(0, 1, 0),
        )
        assert coords == {}


class TestPipelineIntegration:
    """End-to-end integration tests combining HallwayCompactor and HierarchicalLayoutSolver."""

    def test_end_to_end_pipeline_solves_and_restores_intermediate_rooms(self) -> None:
        """Pipeline solve with HallwayCompactor produces coordinates for all rooms."""
        rooms, edges = _make_bidirectional_corridor([1, 2, 3], direction_forward=0, direction_backward=2)
        world_graph = WorldGraph(
            rooms=rooms,
            exits=tuple(e for r in rooms.values() for e in r.exits),
            header=AreaHeader(filename="hall.are", name="Hall", builder="B", vnum_min=1, vnum_max=3),
        )

        solver = HierarchicalLayoutSolver(simplifier=HallwayCompactor())
        solved = solver.solve(world_graph)

        assert isinstance(solved, SolvedLayout)
        assert set(solved.coordinates.keys()) == {1, 2, 3}
        assert solved.diagnostics.status == SolverStatus.FEASIBLE

    def test_end_to_end_with_custom_solver_placements(self) -> None:
        """Verify intermediate room interpolation with explicit solved endpoint coordinates."""
        rooms, edges = _make_bidirectional_corridor([1, 2, 3], direction_forward=0, direction_backward=2)
        world_graph = WorldGraph(
            rooms=rooms,
            exits=tuple(e for r in rooms.values() for e in r.exits),
        )

        class CustomTreeSolver:
            def solve_tree(self, tree: SolveTree, options: LayoutOptions) -> TreeSolution:
                # Place room 1 at (0, 0, 0) and room 3 at (0, 2, 0)
                coords = {1: Coord3D(0, 0, 0), 3: Coord3D(0, 2, 0)}
                node_sol = NodeSolution(
                    node_id="root",
                    local_coordinates=coords,
                    bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 2, 0)),
                )
                return TreeSolution(node_solutions={"root": node_sol})

        solver = HierarchicalLayoutSolver(
            simplifier=HallwayCompactor(),
            tree_solver=CustomTreeSolver(),
        )
        solved = solver.solve(world_graph)

        assert solved.coordinates[1] == Coord3D(0, 0, 0)
        assert solved.coordinates[2] == Coord3D(0, 1, 0)
        assert solved.coordinates[3] == Coord3D(0, 2, 0)

    def test_assembler_missing_endpoint_safely_skipped(self) -> None:
        """If one endpoint coordinate is missing, the corridor expansion is skipped gracefully."""
        assembler = IdentityLayoutAssembler()
        tree = SolveTree(root=SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1})))
        # Only room 1 is solved, room 3 is missing
        sol = TreeSolution(node_solutions={"root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(0, 0, 0)})})
        corridor = CorridorSegment(start_vnum=1, end_vnum=3, direction=0, intermediate_vnums=(2,))
        simplified = SimplifiedGraph(
            graph=LayoutGraph(rooms={}, edges=()),
            corridors=(corridor,),
        )

        result = assembler.assemble(tree, sol, simplified, LayoutOptions())
        assert 1 in result.coordinates
        assert 2 not in result.coordinates
