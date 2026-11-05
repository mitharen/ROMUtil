"""Unit tests for child-child geometric clearance and separation across shared boundary ports."""

from __future__ import annotations

from collections import Counter
import dataclasses
from pathlib import Path

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import (
    ArticulationPointCut,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTreeNode,
    SplitPairCut,
)
from romutil.layout.milp import (
    ClusterFormulation,
    detect_cluster_collisions,
    formulate_cluster_model,
    inventory_cluster,
)
from romutil.layout.milp.clearance import (
    ClearanceConstraintBuilder,
    ClearanceViolation,
    VIOLATION_TIER_PRIORITY,
)
from romutil.layout.milp.inventory import ActiveEdge
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.models import Direction, Room
from romutil.parser import parse_file
from romutil.types import (
    Coord3D,
    LayoutOptions,
    LayoutSolverError,
    SolverStatus,
    WorldGraph,
)
from unittest.mock import MagicMock, patch

import pytest


def _make_graph(
    rooms: list[int],
    edges: list[tuple[int, int, Direction | int, int]],
    cut_edges: list[tuple[int, int, Direction | int, int]] | None = None,
) -> LayoutGraph:
    """Construct a LayoutGraph for testing."""
    room_defs = {
        v: Room(vnum=v, name=f"Room {v}", description="", exits=())
        for v in rooms
    }
    layout_edges: list[LayoutEdge] = [
        LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=False)
        for u, v, d, w in edges
    ]
    if cut_edges:
        for u, v, d, w in cut_edges:
            layout_edges.append(
                LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=True)
            )
    return LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))


class TestChildChildClearance:
    """Tests for pairwise child-child sub-box clearance across shared boundary ports and poles."""

    def test_child_child_separation_shared_boundary_port(self) -> None:
        """Child clusters sharing a boundary port have their interior sub-boxes separated."""
        # Child 1: Split pair cluster possessing interior room 10 and boundary port 1
        c1 = SolveTreeNode(
            node_id="c_spqr",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        # Child 2: Articulation cluster possessing interior room 20 and anchored at boundary port 1
        c2 = SolveTreeNode(
            node_id="c_artic",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c_spqr",
            local_coordinates={10: Coord3D(0, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 1, 1, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c_artic",
            local_coordinates={20: Coord3D(1, 0, 0)},
            port_coordinates={1: Coord3D(0, 0, 0)},
            sub_boxes=(BoundingBox3D(1, 1, 0, 0, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_spqr": sol1, "c_artic": sol2},
            graph=graph,
        )
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_child_child_separation()

        # Both sub-boxes must be registered for disjunctive separation
        separated = formulation.clearance_builder.separated_pairs
        assert ("child_child", ("c_artic", 0), ("c_spqr", 0)) in separated or (
            "child_child",
            ("c_spqr", 0),
            ("c_artic", 0),
        ) in separated
        assert hasattr(formulation.model, "child_child_br")

    def test_child_child_separation_shared_split_pair_poles(self) -> None:
        """Parallel split-pair child clusters sharing poles have their interior sub-boxes separated."""
        c1 = SolveTreeNode(
            node_id="c_branch_a",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        c2 = SolveTreeNode(
            node_id="c_branch_b",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        parent = SolveTreeNode(
            node_id="p_split",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c_branch_a",
            local_coordinates={10: Coord3D(0, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 1, 1, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c_branch_b",
            local_coordinates={20: Coord3D(1, 1, 0)},
            port_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(0, 2, 0)},
            sub_boxes=(BoundingBox3D(1, 1, 1, 1, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_branch_a": sol1, "c_branch_b": sol2},
            graph=graph,
        )
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_child_child_separation()

        separated = formulation.clearance_builder.separated_pairs
        assert ("child_child", ("c_branch_a", 0), ("c_branch_b", 0)) in separated or (
            "child_child",
            ("c_branch_b", 0),
            ("c_branch_a", 0),
        ) in separated
        assert hasattr(formulation.model, "child_child_br")

    def test_detect_cluster_collisions_shared_boundary_port_overlap(self) -> None:
        """Overlapping child sub-boxes sharing a boundary port are detected as collisions."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0), 2: Coord3D(0, 1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )
        formulation = formulate_cluster_model(inventory=inv)

        # Offsets place both child sub-boxes at identical world coordinates
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (0, 2, 0)},
            raw_child_offsets={"c1": (2, 2, 0), "c2": (2, 2, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        child_child_viols = [v for v in violations if v.category == "child_child"]
        assert len(child_child_viols) == 1
        assert child_child_viols[0].item_a == ("c1", 0)
        assert child_child_viols[0].item_b == ("c2", 0)

    def test_detect_cluster_collisions_shared_boundary_port_disjoint(self) -> None:
        """Non-overlapping child sub-boxes sharing a boundary port produce zero child-child violations."""
        c1 = SolveTreeNode(
            node_id="c1",
            node_type="leaf",
            rooms=frozenset({10}),
            metadata={"cut_contract": SplitPairCut(poles=frozenset({1, 2}))},
        )
        c2 = SolveTreeNode(
            node_id="c2",
            node_type="leaf",
            rooms=frozenset({20}),
            metadata={"cut_contract": ArticulationPointCut(anchor_vnum=1)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type="container",
            rooms=frozenset({1, 2}),
            children=(c1, c2),
        )

        sol1 = NodeSolution(
            node_id="c1",
            local_coordinates={10: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0), 2: Coord3D(0, 1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )
        sol2 = NodeSolution(
            node_id="c2",
            local_coordinates={20: Coord3D(0, 0, 0)},
            port_coordinates={1: Coord3D(0, -1, 0)},
            sub_boxes=(BoundingBox3D(0, 0, 0, 0, 0, 0),),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )
        formulation = formulate_cluster_model(inventory=inv)

        # Offsets place child sub-boxes disjointly
        extraction = SolvedClusterExtraction(
            raw_parent_coords={1: (0, 0, 0), 2: (0, 2, 0)},
            raw_child_offsets={"c1": (2, 2, 0), "c2": (10, 10, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        child_child_viols = [v for v in violations if v.category == "child_child"]
        assert len(child_child_viols) == 0

    def test_child_child_separation_empty_sub_boxes_ignored(self) -> None:
        """Child clusters without sub-boxes do not generate separation constraints."""
        c1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset())
        c2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset())
        parent = SolveTreeNode(
            node_id="p_empty",
            node_type="container",
            rooms=frozenset(),
            children=(c1, c2),
        )

        sol1 = NodeSolution("c1", {}, sub_boxes=())
        sol2 = NodeSolution("c2", {}, sub_boxes=())

        graph = _make_graph([], [])
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c1": sol1, "c2": sol2},
            graph=graph,
        )
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_child_child_separation()

        assert len(formulation.clearance_builder.separated_pairs) == 0
        assert not hasattr(formulation.model, "child_child_br")


class TestAreaClearanceIntegrity:
    """End-to-end area layout collision integrity tests."""

    def test_midgaard_zero_room_collisions(self) -> None:
        """Midgaard layout contains zero duplicate room coordinates across all clusters."""
        from romutil.layout.engine import HierarchicalLayoutSolver

        midgaard_path = Path("tests/fixtures/areas/midgaard.are")
        area = parse_file(midgaard_path)
        graph = WorldGraph.from_area_data(area)
        solver = HierarchicalLayoutSolver.production()
        sol = solver.solve(graph)

        coords = sol.coordinates
        counts = Counter(coords.values())
        duplicate_coords = {
            pos: [r for r, c in coords.items() if c == pos]
            for pos, count in counts.items()
            if count > 1
        }
        assert duplicate_coords == {}, f"Found overlapping rooms in Midgaard: {duplicate_coords}"

        # Explicitly verify key pairs across shared boundary ports / poles do not collide
        assert coords[3009] != coords[3004], "3009 (The Bakery) and 3004 (Cleric Guild Entrance) collide"
        assert coords[3028] != coords[3049], "3028 (Thieves Bar) and 3049 (Levee) collide"
        assert coords[3029] != coords[3203], "3029 (The Secret Yard) and 3203 (On the River) collide"

        # Verify Clan Road corridor does not penetrate or collide with the Cleric Inner Sanctum or Temple of Mota
        clan_road_rooms = {3258, 3266, 3315}
        cleric_sanctum_rooms = {3001, 3002, 3003}
        for r_clan in clan_road_rooms:
            for r_cleric in cleric_sanctum_rooms:
                assert coords[r_clan] != coords[r_cleric], f"{r_clan} and {r_cleric} collide"
        assert coords[3258].y != coords[3001].y, "Clan Road penetrates Cleric Inner Sanctum latitude"

    def test_sewer_corridor_avoids_room_7012(self) -> None:
        """Sewer pipe corridor between rooms 7011 and 7021 does not penetrate room 7012."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.layout.geometry import Segment3D
        from romutil.layout.milp.clearance import segment_room_penetration

        sewer_path = Path("tests/fixtures/areas/sewer.are")
        area = parse_file(sewer_path)
        graph = WorldGraph.from_area_data(area)
        solver = HierarchicalLayoutSolver.production()
        sol = solver.solve(graph)

        c7012 = sol.coordinates[7012]
        c7011 = sol.coordinates[7011]
        c7021 = sol.coordinates[7021]

        assert c7012 != c7011
        assert c7012 != c7021

        corridor_segment = Segment3D(c7011, c7021)
        assert not segment_room_penetration(corridor_segment, c7012), (
            f"Room 7012 at {c7012} penetrates corridor between 7011 at {c7011} and 7021 at {c7021}"
        )


    def test_thalos_zero_room_collisions(self) -> None:
        """Thalos layout contains zero duplicate room coordinates across all clusters."""
        from romutil.layout.engine import HierarchicalLayoutSolver

        thalos_path = Path("tests/fixtures/areas/thalos.are")
        area = parse_file(thalos_path)
        graph = WorldGraph.from_area_data(area)
        solver = HierarchicalLayoutSolver.production()
        sol = solver.solve(graph)

        coords = sol.coordinates
        counts = Counter(coords.values())
        duplicate_coords = {
            pos: [r for r, c in coords.items() if c == pos]
            for pos, count in counts.items()
            if count > 1
        }
        assert duplicate_coords == {}, f"Found overlapping rooms in Thalos: {duplicate_coords}"


@pytest.fixture(scope="module")
def midgaard_metropolitan_layout():
    """Module-scoped fixture computing Midgaard Metropolitan composite layout once."""
    from romutil.layout.engine import HierarchicalLayoutSolver
    from romutil.parser import merge_areas

    files = ["midgaard.are", "hood.are", "grave.are", "mobfact.are", "school.are"]
    areas = [parse_file(Path("tests/fixtures/areas") / f) for f in files]
    composite = merge_areas(areas)
    graph = WorldGraph.from_area_data(composite)
    solver = HierarchicalLayoutSolver.production()
    sol = solver.solve(graph, LayoutOptions(timeout_seconds=110))
    return composite, sol


class TestAreaClearanceIntegrity:
    """Integration clearance tests for authentic ROM 2.4 area files."""

    def test_midgaard_metropolitan_gangland_corridor_clearance(
        self, midgaard_metropolitan_layout
    ) -> None:
        """Midgaard Metropolitan elm street corridor does not intersect Gangland solomon street rooms."""
        composite, sol = midgaard_metropolitan_layout
        coords = sol.coordinates
        elm_3124 = coords[3124]
        concourse_3273 = coords[3273]
        assert elm_3124.y == concourse_3273.y
        elm_y = elm_3124.y

        gangland_solomon_rooms = [2166, 2167, 2168, 2169, 2170]
        for v in gangland_solomon_rooms:
            assert coords[v].y > elm_y, f"Room {v} at y={coords[v].y} intersects or is below Elm Street at y={elm_y}"

    def test_midgaard_metropolitan_zero_collisions_and_zero_corridor_crossings(
        self, midgaard_metropolitan_layout
    ) -> None:
        """Midgaard Metropolitan solves with 0 physical room collisions and 0 planar corridor crossings."""
        from romutil.layout.geometry import Segment3D, segments_intersect_3d

        composite, sol = midgaard_metropolitan_layout
        coords = sol.coordinates

        # 1. Zero physical collisions: no duplicate room coordinates
        counts = Counter(coords.values())
        duplicates = {
            pos: [r for r, c in coords.items() if c == pos]
            for pos, count in counts.items()
            if count > 1
        }
        assert duplicates == {}, f"Found overlapping rooms in Midgaard Metropolitan: {duplicates}"

        # 2. Zero corridor crossings: no non-incident exit segments cross in 3D
        # Exclude non-planar cross-area graduation teleport link (e.g. Mud School 3760 -> Midgaard Temple 3001)
        segments: list[tuple[tuple[int, int], Segment3D]] = []
        seen_edges: set[frozenset[int]] = set()
        for room in composite.rooms:
            r_vnum = room.vnum
            if r_vnum not in coords:
                continue
            p1 = coords[r_vnum]
            for exit_def in room.exits:
                to_vnum = exit_def.dst_vnum
                if to_vnum not in coords:
                    continue
                if (r_vnum, to_vnum) in {(3760, 3001), (3001, 3760)}:
                    continue
                edge_key = frozenset({r_vnum, to_vnum})
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                p2 = coords[to_vnum]
                segments.append(((r_vnum, to_vnum), Segment3D(p1, p2)))

        crossings: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(len(segments)):
            e1, seg1 = segments[i]
            for j in range(i + 1, len(segments)):
                e2, seg2 = segments[j]
                if e1[0] in e2 or e1[1] in e2:
                    continue
                if segments_intersect_3d(seg1, seg2):
                    crossings.append((e1, e2))

        assert crossings == [], f"Found corridor crossings in Midgaard Metropolitan: {crossings}"

        # 3. Specifically verify Temple entrance (3014 -> 3005) does not cross any other corridor
        temple_entrance_seg = Segment3D(coords[3014], coords[3005])
        temple_crossings = []
        for e, seg in segments:
            if 3014 in e or 3005 in e:
                continue
            if segments_intersect_3d(temple_entrance_seg, seg):
                temple_crossings.append(e)
        assert temple_crossings == [], f"Temple entrance (3014->3005) crosses corridors: {temple_crossings}"


class TestChildSubBoxCorridorClearance:
    """Unit tests for child keep-out sub-box corridor clearance and collision isolation."""

    def test_corridor_child_universal_obstacle_in_formulation(self) -> None:
        """Every child sub-box is formulated as an active obstacle against corridor interiors."""
        child = SolveTreeNode(node_id="c_branch", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_adapter", node_type="container", rooms=frozenset({100}), children=(child,))
        sol = NodeSolution(
            node_id="c_branch",
            local_coordinates={200: Coord3D(0, 5, 0)},
            port_coordinates={100: Coord3D(0, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 1, 4, 6, 0, 0),  # Sub-box 0: Enclosing interior room 200
            ),
        )

        graph = _make_graph([50, 100, 200], [(50, 100, Direction.north, 1)])
        inv = inventory_cluster(node=parent, child_solutions={"c_branch": sol}, graph=graph)
        inv = dataclasses.replace(inv, boundary_ports=frozenset({50}))
        ae = ActiveEdge(
            edge_id="int_50_100",
            src=50,
            dst=100,
            direction=(0, 1, 0),
            min_length=1,
            associated_rooms=frozenset({50, 100}),
            associated_children=frozenset(),
        )
        inv = dataclasses.replace(inv, active_edges=(ae,))
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_corridor_child_separation()

        # Every child sub-box is an active obstacle in the formulation with zero exceptions
        separated = formulation.clearance_builder.separated_pairs
        assert ("corridor_child", 0, ("c_branch", 0)) in separated

    def test_interface_corridor_stops_at_perimeter_no_collision(self) -> None:
        """Legitimate interface corridor connecting to child doorway stops at perimeter without collision."""
        child = SolveTreeNode(node_id="c_branch", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_adapter", node_type="container", rooms=frozenset({100}), children=(child,))
        sol = NodeSolution(
            node_id="c_branch",
            local_coordinates={200: Coord3D(0, 5, 0)},
            port_coordinates={100: Coord3D(0, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 1, 4, 6, 0, 0),  # Sub-box 0: Strictly interior room 200
            ),
        )

        graph = _make_graph([50, 100, 200], [(50, 100, Direction.north, 5)])
        inv = inventory_cluster(node=parent, child_solutions={"c_branch": sol}, graph=graph)
        inv = dataclasses.replace(inv, boundary_ports=frozenset({50}))
        ae = ActiveEdge(
            edge_id="int_50_100",
            src=50,
            dst=100,
            direction=(0, 1, 0),
            min_length=5,
            associated_rooms=frozenset({50, 100}),
            associated_children=frozenset(),
        )
        inv = dataclasses.replace(inv, active_edges=(ae,))
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_corridor_child_separation()

        # Placement where parent port 50 is at (0, -5, 0) and doorway 100 is at (0, 0, 0).
        # Corridor interior spans y in [-4, -1], which stops at perimeter y=0.
        # Child sub-box 0 spans y in [4, 6], so no collision occurs.
        extraction = SolvedClusterExtraction(
            raw_parent_coords={100: (0, 0, 0)},
            raw_child_offsets={"c_branch": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
            raw_port_coords={50: (0, -5, 0)},
        )
        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 0

    def test_unrelated_corridor_traversing_child_interior_detected_and_separated(self) -> None:
        """Unrelated corridor traversing across a child cluster's interior is detected and separated."""
        child = SolveTreeNode(node_id="c_house", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_root", node_type="container", rooms=frozenset({10, 20}), children=(child,))
        sol = NodeSolution(
            node_id="c_house",
            local_coordinates={200: Coord3D(5, 0, 0)},
            sub_boxes=(
                BoundingBox3D(4, 6, -1, 1, 0, 0),  # Sub-box 0: Encloses room 200
            ),
        )

        graph = _make_graph([10, 20, 200], [(10, 20, Direction.east, 10)])
        inv = inventory_cluster(node=parent, child_solutions={"c_house": sol}, graph=graph)
        ae = ActiveEdge(
            edge_id="ae_street",
            src=10,
            dst=20,
            direction=(1, 0, 0),
            min_length=10,
            associated_rooms=frozenset({10, 20}),
            associated_children=frozenset(),
        )
        inv = dataclasses.replace(inv, active_edges=(ae,))
        formulation = ClusterFormulation(inventory=inv)

        # Extraction where corridor runs from (0, 0, 0) to (10, 0, 0), plowing through [4, 6]
        extraction = SolvedClusterExtraction(
            raw_parent_coords={10: (0, 0, 0), 20: (10, 0, 0)},
            raw_child_offsets={"c_house": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 1
        assert cchild_violations[0].item_b == ("c_house", 0)

        # Cut injection injects corridor_child separation cut
        cuts = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts == 1
        assert hasattr(formulation.model, "corridor_child_br")
        assert ("corridor_child", 0, ("c_house", 0)) in formulation.clearance_builder.separated_pairs

    def test_concave_ushape_child_sub_boxes_protect_non_incident_wings(self) -> None:
        """Concave child sub-boxes strictly detect collision when corridor penetrates non-incident wing."""
        child = SolveTreeNode(node_id="c_ushape", node_type="leaf", rooms=frozenset({10, 20}))
        parent = SolveTreeNode(node_id="p_root", node_type="container", rooms=frozenset({1, 2}), children=(child,))
        sol = NodeSolution(
            node_id="c_ushape",
            local_coordinates={
                10: Coord3D(1, 1, 0),  # Wing 0 doorway at (1, 1, 0)
                20: Coord3D(5, 1, 0),  # Wing 1 interior room
            },
            sub_boxes=(
                BoundingBox3D(0, 1, 0, 2, 0, 0),  # Sub-box 0: Wing 0 up to doorway at x=1
                BoundingBox3D(4, 6, 0, 2, 0, 0),  # Sub-box 1: Wing 1 containing room 20
            ),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c_ushape": sol}, graph=graph)

        # Active edge incident to room 10 in wing 0, running East direction (1, 0, 0) towards parent room 2
        # Corridor interior spans x in [2, 7] at latitude y=1, penetrating wing 1 ([4, 6])
        inv = dataclasses.replace(
            inv,
            active_edges=(
                ActiveEdge(
                    edge_id="ae_piercing",
                    src=("c_ushape", 10),
                    dst=2,
                    direction=(1, 0, 0),
                    min_length=1,
                    associated_rooms=frozenset({10, 2}),
                    associated_children=frozenset({"c_ushape"}),
                ),
            ),
        )
        formulation = formulate_cluster_model(inventory=inv)

        # Extraction where ae_piercing runs from (1, 1, 0) to (8, 1, 0)
        # Corridor interior x in [2, 7] does NOT intersect sub-box 0 (x in [0, 1])
        # but directly penetrates wing 1 [4, 6]!
        extraction = SolvedClusterExtraction(
            raw_parent_coords={2: (8, 1, 0)},
            raw_child_offsets={"c_ushape": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 1
        assert cchild_violations[0].item_b == ("c_ushape", 1)  # Specifically wing 1

        cuts_injected = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_injected == 1
        assert hasattr(formulation.model, "corridor_child_br")
        assert ("corridor_child", 0, ("c_ushape", 1)) in formulation.clearance_builder.separated_pairs


class TestClearanceCutPriorityAndLimits:
    """Unit tests for ClearanceConstraintBuilder cut limits and category priority tiering."""

    def test_violation_tier_priority_mapping(self) -> None:
        """Category tier priority reflects structural hierarchy from macroscopic to corridor crossings."""
        assert VIOLATION_TIER_PRIORITY["parent_room"] == 0
        assert VIOLATION_TIER_PRIORITY["child_child"] == 1
        assert VIOLATION_TIER_PRIORITY["room_child"] == 1
        assert VIOLATION_TIER_PRIORITY["doorway_room"] == 2
        assert VIOLATION_TIER_PRIORITY["doorway_child"] == 2
        assert VIOLATION_TIER_PRIORITY["corridor_room"] == 3
        assert VIOLATION_TIER_PRIORITY["corridor_child"] == 3
        assert VIOLATION_TIER_PRIORITY["corridor_crossing"] == 4

    def test_priority_tiering_injects_highest_priority_violations_first(self) -> None:
        """inject_clearance_cuts prioritizes higher tiers (0-2) over lower tiers (3-4) within budget."""
        formulation = MagicMock()
        formulation.options = LayoutOptions(max_cuts_per_pass=3)
        builder = ClearanceConstraintBuilder(formulation)

        violations = [
            ClearanceViolation(category="corridor_crossing", item_a=1, item_b=2),
            ClearanceViolation(category="corridor_room", item_a=0, item_b=10),
            ClearanceViolation(category="doorway_room", item_a=(1, (1, 0, 0)), item_b=20),
            ClearanceViolation(category="room_child", item_a=30, item_b=("c1", 0)),
            ClearanceViolation(category="parent_room", item_a=100, item_b=101),
        ]

        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation") as mock_sep:
            injected_count = builder.inject_clearance_cuts(violations)
            assert injected_count == 3
            # Tier 0 (parent_room), Tier 1 (room_child), and Tier 2 (doorway_room) should be injected
            assert ("parent_room", 100, 101) in builder.separated_pairs
            assert ("room_child", 30, ("c1", 0)) in builder.separated_pairs
            assert ("doorway_room", (1, (1, 0, 0)), 20) in builder.separated_pairs
            # Lower tier violations (corridor_room, corridor_crossing) should NOT be injected in this pass
            assert ("corridor_room", 0, 10) not in builder.separated_pairs
            assert ("corridor_crossing", 1, 2) not in builder.separated_pairs
            assert mock_sep.call_count == 3

    def test_max_cuts_per_pass_budget_limit_and_sequential_batches(self) -> None:
        """max_cuts_per_pass truncates lower priority violations and permits subsequent batch injection."""
        formulation = MagicMock()
        formulation.options = LayoutOptions(max_cuts_per_pass=2)
        builder = ClearanceConstraintBuilder(formulation)

        violations = [
            ClearanceViolation(category="parent_room", item_a=1, item_b=2),
            ClearanceViolation(category="parent_room", item_a=3, item_b=4),
            ClearanceViolation(category="parent_room", item_a=5, item_b=6),
        ]

        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation"):
            # Pass 1: budget of 2 cuts from formulation options
            cuts1 = builder.inject_clearance_cuts(violations)
            assert cuts1 == 2
            assert len(builder.separated_pairs) == 2
            assert ("parent_room", 1, 2) in builder.separated_pairs
            assert ("parent_room", 3, 4) in builder.separated_pairs

            # Pass 2: remaining unseparated violation injected with explicit override limit
            cuts2 = builder.inject_clearance_cuts(violations, max_cuts_per_pass=1)
            assert cuts2 == 1
            assert len(builder.separated_pairs) == 3
            assert ("parent_room", 5, 6) in builder.separated_pairs

            # Pass 3: all candidate violations are already separated
            cuts3 = builder.inject_clearance_cuts(violations)
            assert cuts3 == 0

    def test_inject_clearance_cuts_empty_and_zero_budget(self) -> None:
        """inject_clearance_cuts returns zero when violations is empty or budget <= 0."""
        formulation = MagicMock()
        builder = ClearanceConstraintBuilder(formulation)
        assert builder.inject_clearance_cuts([]) == 0

        v = [ClearanceViolation(category="parent_room", item_a=1, item_b=2)]
        assert builder.inject_clearance_cuts(v, max_cuts_per_pass=0) == 0
        assert builder.inject_clearance_cuts(v, max_cuts_per_pass=-1) == 0
        assert len(builder.separated_pairs) == 0

    def test_inject_clearance_cuts_deduplicates_in_same_batch(self) -> None:
        """Duplicate violations in the same input batch count once against budget."""
        formulation = MagicMock()
        builder = ClearanceConstraintBuilder(formulation)
        violations = [
            ClearanceViolation(category="parent_room", item_a=1, item_b=2),
            ClearanceViolation(category="parent_room", item_a=2, item_b=1),
            ClearanceViolation(category="child_child", item_a=("c1", 0), item_b=("c2", 0)),
            ClearanceViolation(category="child_child", item_a=("c2", 0), item_b=("c1", 0)),
        ]
        with patch("romutil.layout.milp.clearance.add_disjunctive_box_separation"):
            cuts = builder.inject_clearance_cuts(violations, max_cuts_per_pass=10)
            assert cuts == 2
            assert len(builder.separated_pairs) == 2
