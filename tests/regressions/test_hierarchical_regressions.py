"""Hierarchical layout and composite area defect regression test suite.

Verifies multi-area composite solves and child keep-out sub-box clearance invariants:
- Midgaard Metropolitan: 330-room composite solve, Mud School elevation separation (z > 0),
  zero corridor crossings, Gangland corridor clearance, and Clan Road routing.
- Concave U-shape child keep-out sub-boxes: strict protection of non-incident wings
  without bypasses, and clean perimeter interface corridor termination.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import pytest

from romutil.layout import (
    HierarchicalLayoutSolver,
    LayoutEdge,
    LayoutGraph,
    Segment3D,
    SolveTreeNode,
    segments_intersect_3d,
)
from romutil.layout.geometry import BoundingBox3D
from romutil.layout.milp.clearance import detect_cluster_collisions
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.layout.milp.cluster import formulate_cluster_model
from romutil.layout.milp.inventory import ActiveEdge, NodeSolution, inventory_cluster
from romutil.models import Direction, Exit, Room
from romutil.parser import merge_areas, parse_file
from romutil.types import Coord3D, LayoutOptions, SolvedLayout, SolverStatus, WorldGraph
from tests.regressions.invariants import (
    assert_corridor_non_penetration,
    assert_zero_coordinate_collisions,
)

pytestmark = [pytest.mark.regression]

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "areas"


def _make_layout_graph(
    rooms: list[int],
    edges: list[tuple[int, int, Direction | int, int]],
) -> LayoutGraph:
    room_defs = {
        v: Room(vnum=v, name=f"Room {v}", description="", exits=())
        for v in rooms
    }
    layout_edges = [
        LayoutEdge(src=u, dst=v, direction=int(d), weight=w, cut=False)
        for u, v, d, w in edges
    ]
    return LayoutGraph(rooms=room_defs, edges=tuple(layout_edges))


@pytest.fixture(scope="module")
def midgaard_metropolitan_composite_solution():
    """Module-scoped fixture computing Midgaard Metropolitan composite layout once."""
    files = ["midgaard.are", "hood.are", "grave.are", "mobfact.are", "school.are"]
    areas = [parse_file(FIXTURES_DIR / f) for f in files]
    composite = merge_areas(areas)
    graph = WorldGraph.from_area_data(composite)
    solver = HierarchicalLayoutSolver.production()
    layout = solver.solve(graph, LayoutOptions(timeout_seconds=110))
    return composite, graph, layout


class TestMidgaardMetropolitanCompositeRegression:
    """Regression suite verifying 330-room Midgaard Metropolitan composite layout."""

    def test_midgaard_metropolitan_elevation_and_collision_invariants(
        self, midgaard_metropolitan_composite_solution
    ) -> None:
        """Composite solves 330 rooms with 0 collisions and strict Mud School elevation separation."""
        composite, graph, layout = midgaard_metropolitan_composite_solution

        assert len(graph.rooms) == 330
        assert layout.diagnostics.status in (
            SolverStatus.FEASIBLE,
            SolverStatus.OPTIMAL,
            SolverStatus.TIMEOUT,
        )
        assert len(layout.coordinates) == 330

        # Invariant 1: Zero coordinate collisions across all 330 composite rooms
        assert_zero_coordinate_collisions(layout)

        # Invariant 2: Mud School (rooms 3700-3760) vertical elevation separation (z > 0)
        coords = layout.coordinates
        school_rooms = [r.vnum for r in composite.rooms if 3700 <= r.vnum <= 3799]
        assert len(school_rooms) == 59
        for v in school_rooms:
            assert coords[v].z > 0, (
                f"Mud School room {v} at z={coords[v].z}; expected elevation separation z > 0"
            )

        # Invariant 3: Elm Street corridor clearance against Gangland Solomon Street
        elm_3124 = coords[3124]
        concourse_3273 = coords[3273]
        assert elm_3124.y == concourse_3273.y
        elm_y = elm_3124.y

        gangland_solomon_rooms = [2166, 2167, 2168, 2169, 2170]
        for v in gangland_solomon_rooms:
            assert coords[v].y > elm_y, (
                f"Gangland room {v} at y={coords[v].y} intersects or is below Elm Street at y={elm_y}"
            )

        # Invariant 4: Clan Road routes strictly north of Temple Square
        assert coords[3315].y > coords[3005].y, (
            f"Clan Road at y={coords[3315].y} must route strictly north of Temple Square at y={coords[3005].y}"
        )

    def test_midgaard_metropolitan_zero_corridor_crossings(
        self, midgaard_metropolitan_composite_solution
    ) -> None:
        """Composite layout exhibits zero 3D crossings among non-incident corridor segments."""
        composite, _graph, layout = midgaard_metropolitan_composite_solution
        coords = layout.coordinates

        # Gather all non-teleport corridor segments
        segments: list[tuple[tuple[int, int], Segment3D]] = []
        seen_edges: set[frozenset[int]] = set()

        for room in composite.rooms:
            u = room.vnum
            if u not in coords:
                continue
            pu = coords[u]
            for exit_def in room.exits:
                v = exit_def.dst_vnum
                if v not in coords:
                    continue
                # Exclude non-planar cross-area graduation teleport link
                if (u, v) in {(3760, 3001), (3001, 3760)}:
                    continue
                edge_key = frozenset({u, v})
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                pv = coords[v]
                segments.append(((u, v), Segment3D(pu, pv)))

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

        # Specifically verify Temple entrance (3014 -> 3005) does not cross any other corridor
        temple_entrance_seg = Segment3D(coords[3014], coords[3005])
        temple_crossings = [
            e for e, seg in segments
            if 3014 not in e and 3005 not in e and segments_intersect_3d(temple_entrance_seg, seg)
        ]
        assert temple_crossings == [], f"Temple entrance (3014->3005) crosses corridors: {temple_crossings}"


class TestConcaveSubBoxClearanceRegression:
    """Regression suite verifying concave child keep-out sub-box protection and non-penetration."""

    def test_concave_ushape_sub_boxes_protect_non_incident_wings(self) -> None:
        """Concave child keep-out sub-boxes detect penetration when corridor pierces non-incident wing."""
        # Build concave child cluster:
        # Wing 0: x in [0, 1], doorway at (1, 1, 0)
        # Wing 1: x in [4, 6], interior room 20 at (5, 1, 0)
        child = SolveTreeNode(node_id="c_ushape", node_type="leaf", rooms=frozenset({10, 20}))
        parent = SolveTreeNode(
            node_id="p_root", node_type="container", rooms=frozenset({1, 2}), children=(child,)
        )
        sol = NodeSolution(
            node_id="c_ushape",
            local_coordinates={
                10: Coord3D(1, 1, 0),
                20: Coord3D(5, 1, 0),
            },
            sub_boxes=(
                BoundingBox3D(0, 1, 0, 2, 0, 0),  # Wing 0 up to doorway at x=1
                BoundingBox3D(4, 6, 0, 2, 0, 0),  # Wing 1 containing room 20
            ),
        )

        # Graph with corridor running East from doorway room 10 to parent room 2
        rooms_map = {
            1: Room(vnum=1, name="Parent 1", description="desc", exits=()),
            2: Room(vnum=2, name="Parent 2", description="desc", exits=()),
            10: Room(vnum=10, name="Child 10", description="desc", exits=(Exit(direction=Direction.east, dst_vnum=2),)),
            20: Room(vnum=20, name="Child 20", description="desc", exits=()),
        }
        graph = WorldGraph(rooms=rooms_map, exits=())

        # Test invariant helper: corridor from (1, 1, 0) to (8, 1, 0) penetrates wing 1 [4, 6]
        piercing_layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(8, 1, 0),
                10: Coord3D(1, 1, 0),
                20: Coord3D(5, 1, 0),
            }
        )
        wing_boxes = [BoundingBox3D(0, 1, 0, 2, 0, 0), BoundingBox3D(4, 6, 0, 2, 0, 0)]
        with pytest.raises(AssertionError) as exc_info:
            assert_corridor_non_penetration(piercing_layout, graph, keepout_boxes=wing_boxes)
        assert "intersects keep-out box" in str(exc_info.value)

        # Also verify MILP cluster formulation detects this penetration
        layout_graph = _make_layout_graph([1, 2, 10, 20], [])
        inv = inventory_cluster(node=parent, child_solutions={"c_ushape": sol}, graph=layout_graph)
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

        extraction = SolvedClusterExtraction(
            raw_parent_coords={2: (8, 1, 0)},
            raw_child_offsets={"c_ushape": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
        )
        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 1
        assert cchild_violations[0].item_b == ("c_ushape", 1)  # Wing 1 keep-out box

        cuts_injected = formulation.clearance_builder.inject_clearance_cuts(violations)
        assert cuts_injected == 1
        assert hasattr(formulation.model, "corridor_child_br")
        assert ("corridor_child", 0, ("c_ushape", 1)) in formulation.clearance_builder.separated_pairs

    def test_child_corridor_interface_doorway_perimeter_clearance(self) -> None:
        """Legitimate interface corridor terminating at doorway perimeter does not collide with child sub-boxes."""
        child = SolveTreeNode(node_id="c_branch", node_type="leaf", rooms=frozenset({200}))
        parent = SolveTreeNode(
            node_id="p_adapter", node_type="container", rooms=frozenset({100}), children=(child,)
        )
        sol = NodeSolution(
            node_id="c_branch",
            local_coordinates={200: Coord3D(0, 5, 0)},
            port_coordinates={100: Coord3D(0, 0, 0)},
            sub_boxes=(
                BoundingBox3D(0, 1, 4, 6, 0, 0),  # Strictly interior room 200
            ),
        )

        layout_graph = _make_layout_graph([50, 100, 200], [(50, 100, Direction.north, 5)])
        inv = inventory_cluster(node=parent, child_solutions={"c_branch": sol}, graph=layout_graph)
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
        formulation = formulate_cluster_model(inventory=inv)
        formulation.clearance_builder.add_corridor_child_separation()

        extraction = SolvedClusterExtraction(
            raw_parent_coords={100: (0, 0, 0)},
            raw_child_offsets={"c_branch": (0, 0, 0)},
            solver_status=SolverStatus.OPTIMAL,
            raw_port_coords={50: (0, -5, 0)},
        )
        violations = detect_cluster_collisions(formulation, extraction)
        cchild_violations = [v for v in violations if v.category == "corridor_child"]
        assert len(cchild_violations) == 0
