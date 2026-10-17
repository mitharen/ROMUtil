"""Unit tests for child-child geometric clearance and separation across shared boundary ports."""

from __future__ import annotations

from collections import Counter
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
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.models import Direction, Room
from romutil.parser import parse_file
from romutil.types import Coord3D, LayoutSolverError, SolverStatus, WorldGraph


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
