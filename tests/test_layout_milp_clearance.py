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
    get_incident_child_sub_box_indices,
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

    def test_midgaard_metropolitan_gangland_corridor_clearance(self) -> None:
        """Midgaard Metropolitan elm street corridor does not intersect Gangland solomon street rooms."""
        from romutil.layout.engine import HierarchicalLayoutSolver
        from romutil.types import LayoutOptions

        from romutil.parser import merge_areas

        files = ["midgaard.are", "hood.are", "grave.are", "mobfact.are", "school.are"]
        areas = [parse_file(Path("tests/fixtures/areas") / f) for f in files]
        composite = merge_areas(areas)
        graph = WorldGraph.from_area_data(composite)
        solver = HierarchicalLayoutSolver.production()
        sol = solver.solve(graph, LayoutOptions(timeout_seconds=15))

        coords = sol.coordinates
        elm_3124 = coords[3124]
        concourse_3273 = coords[3273]
        assert elm_3124.y == concourse_3273.y
        elm_y = elm_3124.y

        gangland_solomon_rooms = [2166, 2167, 2168, 2169, 2170]
        for v in gangland_solomon_rooms:
            assert coords[v].y > elm_y, f"Room {v} at y={coords[v].y} intersects or is below Elm Street at y={elm_y}"


class TestPortScopedSubBoxCorridorClearance:
    """Unit tests for port-scoped child sub-box corridor clearance exemptions."""

    def test_get_incident_child_sub_box_indices_strictly_enclosed(self) -> None:
        """Active edges incident to specific ports strictly enclose and exempt only that sub-box."""
        sol = NodeSolution(
            node_id="c_l",
            local_coordinates={
                10: Coord3D(1, 1, 0),
                20: Coord3D(6, 1, 0),
            },
            port_coordinates={
                100: Coord3D(1, 1, 0),
                200: Coord3D(6, 1, 0),
            },
            sub_boxes=(
                BoundingBox3D(0, 2, 0, 2, 0, 0),  # Sub-box 0 encloses room 10 / port 100
                BoundingBox3D(5, 7, 0, 2, 0, 0),  # Sub-box 1 encloses room 20 / port 200
            ),
        )

        # Edge incident to room 10 on c_l
        ae1 = ActiveEdge(
            edge_id="ae_port10",
            src=("c_l", 10),
            dst=1,
            direction=(0, -1, 0),
            min_length=2,
            associated_rooms=frozenset({10, 1}),
            associated_children=frozenset({"c_l"}),
        )
        exempt1 = get_incident_child_sub_box_indices("c_l", sol, sol.sub_boxes, ae1)
        assert exempt1 == {0}

        # Edge incident to port 200 on c_l
        ae2 = ActiveEdge(
            edge_id="ae_port200",
            src=2,
            dst=("c_l", 200),
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({2, 200}),
            associated_children=frozenset({"c_l"}),
        )
        exempt2 = get_incident_child_sub_box_indices("c_l", sol, sol.sub_boxes, ae2)
        assert exempt2 == {1}

        # Edge not incident to c_l
        ae_other = ActiveEdge(
            edge_id="ae_ext",
            src=1,
            dst=2,
            direction=(1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 2}),
            associated_children=frozenset(),
        )
        exempt_other = get_incident_child_sub_box_indices("c_l", sol, sol.sub_boxes, ae_other)
        assert exempt_other == set()

    def test_get_incident_child_sub_box_indices_carved_out_adjacent_port(self) -> None:
        """Boundary ports carved out of keep-out sub-boxes fall back to adjacent sub-box."""
        sol = NodeSolution(
            node_id="c_carved",
            local_coordinates={10: Coord3D(2, 2, 0)},
            port_coordinates={
                1: Coord3D(0, 2, 0),  # Chebyshev distance 1 from box 0 ([1, 3] x [1, 3])
            },
            sub_boxes=(
                BoundingBox3D(1, 3, 1, 3, 0, 0),  # Sub-box 0 adjacent to port 1
                BoundingBox3D(8, 10, 1, 3, 0, 0),  # Sub-box 1 distant
            ),
        )

        ae = ActiveEdge(
            edge_id="ae_carved",
            src=("c_carved", 1),
            dst=99,
            direction=(-1, 0, 0),
            min_length=2,
            associated_rooms=frozenset({1, 99}),
            associated_children=frozenset({"c_carved"}),
        )
        exempt = get_incident_child_sub_box_indices("c_carved", sol, sol.sub_boxes, ae)
        assert exempt == {0}

    def test_get_incident_child_sub_box_indices_multi_port(self) -> None:
        """Active edges touching multiple boundary ports exempt all incident sub-boxes."""
        sol = NodeSolution(
            node_id="c_multi",
            local_coordinates={10: Coord3D(1, 1, 0), 20: Coord3D(6, 1, 0)},
            sub_boxes=(
                BoundingBox3D(0, 2, 0, 2, 0, 0),
                BoundingBox3D(5, 7, 0, 2, 0, 0),
            ),
        )

        # Intra-cluster corridor spanning between room 10 and room 20
        ae = ActiveEdge(
            edge_id="ae_internal",
            src=("c_multi", 10),
            dst=("c_multi", 20),
            direction=(1, 0, 0),
            min_length=5,
            associated_rooms=frozenset({10, 20}),
            associated_children=frozenset({"c_multi"}),
        )
        exempt = get_incident_child_sub_box_indices("c_multi", sol, sol.sub_boxes, ae)
        assert exempt == {0, 1}

    def test_corridor_child_collision_detects_penetration_of_non_incident_wing(self) -> None:
        """Corridor entering one sub-box of a child cluster detects collision when penetrating another wing."""
        # U-shaped child cluster with wing 0 and wing 1 at the same latitude
        child = SolveTreeNode(node_id="c_ushape", node_type="leaf", rooms=frozenset({10, 20}))
        parent = SolveTreeNode(node_id="p_root", node_type="container", rooms=frozenset({1, 2}), children=(child,))
        sol = NodeSolution(
            node_id="c_ushape",
            local_coordinates={
                10: Coord3D(1, 1, 0),  # Wing 0 port
                20: Coord3D(5, 1, 0),  # Wing 1 interior room
            },
            sub_boxes=(
                BoundingBox3D(0, 2, 0, 2, 0, 0),  # Sub-box 0: Wing 0 containing port 10
                BoundingBox3D(4, 6, 0, 2, 0, 0),  # Sub-box 1: Wing 1 containing room 20
            ),
        )

        graph = _make_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c_ushape": sol}, graph=graph)

        # Active edge incident to room 10 in wing 0, running East direction (1, 0, 0)
        # Slices across x in [2, 7] at latitude y=1, directly penetrating wing 1 ([4, 6])
        inv = dataclasses.replace(
            inv,
            active_edges=(
                ActiveEdge(
                    edge_id="ae_piercing",
                    src=("c_ushape", 10),
                    dst=2,
                    direction=(1, 0, 0),
                    min_length=7,
                    associated_rooms=frozenset({10, 2}),
                    associated_children=frozenset({"c_ushape"}),
                ),
            ),
        )
        formulation = formulate_cluster_model(inventory=inv)

        # Extraction where ae_piercing runs from (1, 1, 0) to (8, 1, 0)
        # Child offset is (0, 0, 0), so wing 1 spans [4, 6] x [0, 2] x [0, 0]
        # Corridor interior x in [2, 7] at y=1 penetrates wing 1 [4, 6]!
        extraction = SolvedClusterExtraction(
            raw_parent_coords={2: (8, 1, 0)},
            raw_child_offsets={"c_ushape": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )

        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 1
        assert cchild_violations[0].item_b == ("c_ushape", 1)  # Specifically sub-box 1 (wing 1)

        # Verify cut injection applies corridor_child separation
        cuts_injected = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_injected == 1
        assert hasattr(formulation.model, "corridor_child_br")

    def test_corridor_child_separation_non_associated_boundary_port(self) -> None:
        """Active edge touching a boundary port of a non-associated child separates interior from other sub-boxes."""
        # Child cluster possessing port 100 (at entrance) and interior room 200 in sub-box 1
        child = SolveTreeNode(node_id="c_gang", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_root", node_type="container", rooms=frozenset({100, 300}), children=(child,))
        sol = NodeSolution(
            node_id="c_gang",
            local_coordinates={200: Coord3D(5, 0, 0)},
            port_coordinates={100: Coord3D(0, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 1, 0, 1, 0, 0),  # Sub-box 0: Encloses port 100
                BoundingBox3D(4, 6, 0, 1, 0, 0),  # Sub-box 1: Encloses room 200
            ),
        )

        graph = _make_graph([100, 200, 300], [(100, 300, Direction.east, 10)])
        inv = inventory_cluster(node=parent, child_solutions={"c_gang": sol}, graph=graph)
        # Active corridor from 100 to 300 (associated_children is empty, non-associated)
        ae = ActiveEdge(
            edge_id="ae_street",
            src=100,
            dst=300,
            direction=(1, 0, 0),
            min_length=10,
            associated_rooms=frozenset({100, 300}),
            associated_children=frozenset(),
        )
        inv = dataclasses.replace(inv, active_edges=(ae,))
        formulation = ClusterFormulation(inventory=inv)
        formulation.clearance_builder.add_corridor_child_separation()

        # Sub-box 0 (incident to port 100) is exempt; sub-box 1 must be separated
        separated = formulation.clearance_builder.separated_pairs
        assert ("corridor_child", 0, ("c_gang", 1)) in separated
        assert ("corridor_child", 0, ("c_gang", 0)) not in separated
        assert hasattr(formulation.model, "corridor_child_br")

        # Test collision detection:
        # Placement where child sub-box 1 (at x in [4, 6]) penetrates corridor interior (x in [1, 9])
        extraction_colliding = SolvedClusterExtraction(
            raw_parent_coords={100: (0, 0, 0), 300: (10, 0, 0)},
            raw_child_offsets={"c_gang": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations = detect_cluster_collisions(formulation, extraction_colliding)
        cchild_viols = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_viols) == 1
        assert cchild_viols[0].item_b == ("c_gang", 1)

        # Placement where child is translated clear of corridor (y = 5)
        extraction_clear = SolvedClusterExtraction(
            raw_parent_coords={100: (0, 0, 0), 300: (10, 0, 0)},
            raw_child_offsets={"c_gang": (0, 5, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations_clear = detect_cluster_collisions(formulation, extraction_clear)
        assert len([v for v in violations_clear if v.category == "corridor_child"]) == 0

    def test_boundary_interface_edge_exempt_from_child_separation(self) -> None:
        """Active edge directly connecting external boundary port to child attachment port is exempt."""
        child = SolveTreeNode(node_id="c_branch", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(node_id="p_adapter", node_type="container", rooms=frozenset({100}), children=(child,))
        sol = NodeSolution(
            node_id="c_branch",
            local_coordinates={200: Coord3D(0, 5, 0)},
            port_coordinates={100: Coord3D(0, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 1, 0, 1, 0, 0),  # Sub-box 0: Port
                BoundingBox3D(0, 1, 4, 6, 0, 0),  # Sub-box 1: Room 200
            ),
        )

        graph = _make_graph([50, 100, 200], [(50, 100, Direction.north, 1)])
        inv = inventory_cluster(node=parent, child_solutions={"c_branch": sol}, graph=graph)
        # 50 is an external boundary port connecting to child attachment port 100
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

        # Both sub-boxes of c_branch should be exempt because int_50_100 is an external boundary interface edge
        separated = formulation.clearance_builder.separated_pairs
        assert len([p for p in separated if p[0] == "corridor_child"]) == 0

        # Collision detection also skips boundary interface edge against child
        extraction = SolvedClusterExtraction(
            raw_parent_coords={100: (0, 0, 0)},
            raw_child_offsets={"c_branch": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
            raw_port_coords={50: (0, -10, 0)},
        )
        violations = detect_cluster_collisions(formulation, extraction)
        assert len([v for v in violations if v.category == "corridor_child"]) == 0


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
