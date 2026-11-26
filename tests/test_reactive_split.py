"""Unit and integration tests for reactive directional cut-set splitting."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from romutil.layout import LayoutGraph
from romutil.layout.assembly import HierarchicalLayoutAssembler
from romutil.layout.hierarchy.pipeline import PipelineTreeBuilder
from romutil.layout.ir import (
    ArticulationPointCut,
    BoundingBox3D,
    BridgeCut,
    CutType,
    LayoutEdge,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    SplitPairCut,
    TreeNodeType,
)
from romutil.layout.milp.clearance import ClearanceConstraintBuilder, ClearanceViolation
from romutil.layout.milp.cluster import (
    UnresolvableCollisionError,
    _extract_colliding_entities,
    _raise_collision_error,
    solve_cluster_with_cutting_planes,
)
from romutil.layout.milp.formulation import formulate_cluster_model
from romutil.layout.milp.inventory import (
    ActiveEdge,
    ChildChildAlignment,
    ClusterInventory,
    GatewayCutHandler,
    get_child_contract_interface_rooms,
    inventory_cluster,
)
from romutil.layout.milp.reactive_split import (
    _as_coord3d,
    _child_covered_exit_axes,
    _safe_displacement,
    compute_split_lp_signature,
    compute_total_edge_length,
    detect_planar_transverse_shear,
    diagnose_transverse_shear,
    diagnose_unresolvable_collision,
    generate_coordinate_threshold_partitions,
    get_child_interface_rooms,
    is_splittable_child,
    solve_3d_deformation_diagnostic,
    split_child_along_direction,
)
from romutil.layout.milp.solver import LayoutInfeasibleError
from romutil.layout.milp.tree import HierarchicalTreeSolver
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import Direction
from romutil.types import Coord3D, LayoutOptions, LayoutSolverError, Room, SolverStatus


def _room(vnum: int) -> Room:
    return Room(vnum=vnum, name=f"Room {vnum}", description="Test room.")


def _edge(src: int, dst: int, direction: Direction, weight: int = 1, cut: bool = False) -> LayoutEdge:
    return LayoutEdge(
        src=src,
        dst=dst,
        direction=direction,
        weight=weight,
        cut=cut,
        metadata={},
    )


def _bidir_edges(pairs: list[tuple[int, int, Direction, int]]) -> tuple[LayoutEdge, ...]:
    edges: list[LayoutEdge] = []
    for u, v, d, w in pairs:
        edges.append(_edge(u, v, d, weight=w))
        edges.append(_edge(v, u, d.invert(), weight=w))
    return tuple(edges)


def _node_sol(
    node_id: str,
    local_coords: dict[int, Coord3D],
    port_coords: dict[int, Coord3D] | None = None,
    all_coords: dict[int, Coord3D] | None = None,
    sub_boxes: tuple[BoundingBox3D, ...] = (),
) -> NodeSolution:
    all_map = dict(local_coords)
    if all_coords:
        all_map.update(all_coords)
    ports = dict(port_coords or {})
    return NodeSolution(
        node_id=node_id,
        local_coordinates=local_coords,
        sub_boxes=sub_boxes,
        metadata={
            "all_room_coordinates": all_map,
            "port_coordinates": ports,
            "child_offsets": {},
            "child_sub_boxes": {},
            "status": SolverStatus.OPTIMAL,
        },
        port_coordinates=ports,
    )


def _split_pair_meta(p0: int, p1: int, extra: dict | None = None) -> dict:
    meta = {
        "cut_type": CutType.SPLIT_PAIR,
        "poles": (p0, p1),
        "cut_contract": SplitPairCut(poles=(p0, p1)),
    }
    if extra:
        meta.update(extra)
    return meta


def _bridge_meta(u: int, v: int, direction: Direction, weight: int = 1) -> dict:
    return {
        "cut_type": CutType.BRIDGE,
        "anchor_vnum": u,
        "bridge_edge": (u, v),
        "direction": int(direction),
        "weight": weight,
        "cut_contract": BridgeCut(
            bridge_edge=(u, v),
            direction=int(direction),
            weight=weight,
            anchor_vnum=u,
        ),
    }


def _ap_meta(anchor: int, extra: dict | None = None) -> dict:
    meta = {
        "cut_type": CutType.ARTICULATION_POINT,
        "anchor_vnum": anchor,
        "cut_contract": ArticulationPointCut(anchor_vnum=anchor),
    }
    if extra:
        meta.update(extra)
    return meta


class TestSplittableAndInterfaces:
    """Tests for splittable child classification and interface room discovery."""

    def test_is_splittable_child_classification(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11, 12, 20)},
            edges=_bidir_edges(
                [
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.east, 1),
                    (11, 2, Direction.east, 1),
                    (1, 20, Direction.north, 1),
                    (2, 12, Direction.south, 1),
                ]
            ),
        )
        split_child = SolveTreeNode(
            node_id="c_split",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11}),
            children=(),
            metadata=_split_pair_meta(1, 2),
        )
        single_room_split = SolveTreeNode(
            node_id="c_single",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({12}),
            children=(),
            metadata=_split_pair_meta(1, 2),
        )
        one_cut_child = SolveTreeNode(
            node_id="c_bridge",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({20, 21}),
            children=(),
            metadata=_bridge_meta(1, 20, Direction.north),
        )
        nested_shell = SolveTreeNode(
            node_id="c_shell",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(split_child,),
            metadata=_split_pair_meta(1, 2),
        )
        reactive_piece = SolveTreeNode(
            node_id="c_piece",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({10, 11}),
            children=(),
            metadata={
                **_ap_meta(1),
                "reactive_split_origin": "c_split",
                "boundary_ports": (1, 2),
            },
        )

        sols = {
            "c_split": _node_sol("c_split", {10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)}, {1: Coord3D(-1, 0, 0), 2: Coord3D(2, 0, 0)}),
            "c_single": _node_sol("c_single", {12: Coord3D(0, 0, 0)}, {1: Coord3D(-1, 0, 0), 2: Coord3D(1, 0, 0)}),
            "c_bridge": _node_sol("c_bridge", {20: Coord3D(0, 0, 0), 21: Coord3D(0, 1, 0)}, {1: Coord3D(0, -1, 0)}),
            "c_shell": _node_sol("c_shell", {}, {1: Coord3D(-1, 0, 0), 2: Coord3D(2, 0, 0)}, {10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)}),
            "c_piece": _node_sol("c_piece", {10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)}, {1: Coord3D(-1, 0, 0), 2: Coord3D(2, 0, 0)}),
        }

        parent = SolveTreeNode(
            node_id="parent",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2}),
            children=(split_child, single_room_split, one_cut_child, nested_shell, reactive_piece),
            metadata={},
        )
        inv = ClusterInventory(
            node=parent,
            rooms=(1, 2),
            children=tuple(parent.children),
            child_solutions=sols,
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("e1", 1, ("c_split", 10), (1, 0, 0), 1, frozenset({1}), frozenset({"c_split"})),
                ActiveEdge("e2", ("c_split", 11), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"c_split"})),
                ActiveEdge("e3", 1, ("c_shell", 10), (1, 0, 0), 1, frozenset({1}), frozenset({"c_shell"})),
                ActiveEdge("e4", ("c_shell", 11), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"c_shell"})),
                ActiveEdge("e5", 1, ("c_piece", 10), (1, 0, 0), 1, frozenset({1}), frozenset({"c_piece"})),
                ActiveEdge("e6", ("c_piece", 11), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"c_piece"})),
                ActiveEdge("e7", 1, ("c_single", 12), (0, -1, 0), 1, frozenset({1}), frozenset({"c_single"})),
                ActiveEdge("e8", ("c_single", 12), 2, (0, 1, 0), 1, frozenset({2}), frozenset({"c_single"})),
                ActiveEdge("e9", 1, ("c_bridge", 20), (0, 1, 0), 1, frozenset({1}), frozenset({"c_bridge"})),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        assert is_splittable_child(split_child, inv, sols) is True
        assert is_splittable_child(single_room_split, inv, sols) is False
        assert is_splittable_child(one_cut_child, inv, sols) is False
        assert is_splittable_child(nested_shell, inv, sols) is True
        assert is_splittable_child(reactive_piece, inv, sols) is True
        iface, coords = get_child_interface_rooms(split_child, inv, sols)
        assert set(iface) == {10, 11}
        assert set(coords) == {10, 11}


class TestDirectionalSplitter:
    """Tests for Pass 1 1-edge cut, sc_group atomicity, and Pass 2 Edmonds-Karp directed min-cut."""

    def test_invalid_inputs_return_none(self) -> None:
        graph = LayoutGraph(rooms={1: _room(1), 2: _room(2)}, edges=())
        child = SolveTreeNode("c", TreeNodeType.SERIES_LOOP, frozenset({1, 2}), (), _split_pair_meta(10, 20))
        assert split_child_along_direction(child, {1}, {2}, (1, 1, 0), {10, 20}, graph, "r0") is None
        assert split_child_along_direction(child, set(), {2}, (1, 0, 0), {10, 20}, graph, "r0") is None
        assert split_child_along_direction(child, {1}, {1, 2}, (1, 0, 0), {10, 20}, graph, "r0") is None

        single = SolveTreeNode("s", TreeNodeType.SERIES_LOOP, frozenset({1}), (), _split_pair_meta(10, 20))
        assert split_child_along_direction(single, {1}, {2}, (1, 0, 0), {10, 20}, graph, "r0") is None

    def test_pass1_balanced_bridge_cut_respects_atomic_sc_groups(self) -> None:
        rooms = {v: _room(v) for v in (1, 2, 10, 11, 12, 13, 30, 31)}
        graph = LayoutGraph(
            rooms=rooms,
            edges=_bidir_edges(
                [
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.east, 1),
                    (11, 12, Direction.east, 1),
                    (12, 13, Direction.east, 1),
                    (13, 2, Direction.east, 1),
                    (11, 30, Direction.south, 1),
                    (30, 31, Direction.east, 1),
                    (31, 12, Direction.north, 1),
                ]
            ),
        )
        sc = SolveTreeNode(
            node_id="sc_wing",
            node_type=TreeNodeType.BLOCK,
            rooms=frozenset({30, 31}),
            children=(),
            metadata=_ap_meta(11),
        )
        child = SolveTreeNode(
            node_id="c_chain",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 11, 12, 13}),
            children=(sc,),
            metadata=_split_pair_meta(1, 2),
        )
        res = split_child_along_direction(child, {10}, {13}, (1, 0, 0), {1, 2}, graph, "rs0")
        assert res is not None
        piece_s, piece_t, unpacked = res
        assert unpacked == []
        assert 10 in piece_s.rooms and 13 in piece_t.rooms
        s_subs = {c.node_id for c in piece_s.children}
        t_subs = {c.node_id for c in piece_t.children}
        assert len(s_subs ^ t_subs) == 1 and "sc_wing" in (s_subs | t_subs)
        assert piece_s.metadata.get("severed_bridge_edges") == piece_t.metadata.get("severed_bridge_edges")

    def test_pass2_parallel_p_node_directed_min_cut_unpacks_nested_shells(self) -> None:
        rooms = {v: _room(v) for v in (1, 2, 10, 11, 20, 21, 22, 30)}
        graph = LayoutGraph(
            rooms=rooms,
            edges=_bidir_edges(
                [
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.east, 1),
                    (11, 2, Direction.east, 1),
                    (1, 20, Direction.north, 1),
                    (20, 21, Direction.east, 1),
                    (21, 22, Direction.east, 1),
                    (22, 2, Direction.south, 1),
                    (21, 30, Direction.north, 1),
                ]
            ),
        )
        sub_leaf = SolveTreeNode("leaf", TreeNodeType.BLOCK, frozenset({30}), (), _bridge_meta(21, 30, Direction.north))
        branch_a = SolveTreeNode("branch_a", TreeNodeType.SERIES_LOOP, frozenset({10, 11}), (), _split_pair_meta(1, 2))
        branch_b = SolveTreeNode("branch_b", TreeNodeType.SERIES_LOOP, frozenset({20, 21, 22}), (sub_leaf,), _split_pair_meta(1, 2))
        p_node = SolveTreeNode(
            node_id="p_cont",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(branch_a, branch_b),
            metadata=_split_pair_meta(1, 2),
        )
        res = split_child_along_direction(p_node, {10, 20}, {11, 22}, (1, 0, 0), {1, 2}, graph, "rs0")
        assert res is not None
        piece_s, piece_t, unpacked = res
        assert {u.node_id for u in unpacked} == {"branch_a", "branch_b"}
        assert {10, 20}.issubset(piece_s.rooms)
        assert {11, 22}.issubset(piece_t.rooms)
        severed = set(piece_s.metadata["severed_bridge_edges"])
        assert (10, 11) in severed
        assert any(e in severed for e in ((20, 21), (21, 22)))


class TestDiagnosticsAndHybridTriggers:
    """Tests for collision/shear diagnostics, inventory port hooks, and hybrid tree triggers."""

    def test_3d_diagnostic_detects_vertical_split_axis(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 10, 11, 12)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.north, 3),
                    (3, 2, Direction.east, 3),
                    (1, 10, Direction.east, 1),
                    (10, 12, Direction.east, 1),
                    (12, 11, Direction.north, 1),
                    (11, 2, Direction.east, 1),
                ]
            ),
        )
        child = SolveTreeNode(
            node_id="c_rigid",
            node_type=TreeNodeType.SERIES_LOOP,
            rooms=frozenset({10, 12, 11}),
            children=(),
            metadata=_split_pair_meta(1, 2),
        )
        c_sol = _node_sol(
            "c_rigid",
            {10: Coord3D(0, 0, 0), 12: Coord3D(1, 0, 0), 11: Coord3D(1, 1, 0)},
            {1: Coord3D(-1, 0, 0), 2: Coord3D(2, 1, 0)},
        )
        parent = SolveTreeNode(
            node_id="p_root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2, 3}),
            children=(child,),
            metadata={},
        )
        inv = inventory_cluster(node=parent, child_solutions={"c_rigid": c_sol}, graph=graph, options=LayoutOptions())
        requests = solve_3d_deformation_diagnostic(inv, [child])
        assert len(requests) == 1
        cid, s_r, t_r, u_dir = requests[0]
        assert cid == "c_rigid"
        assert s_r == {10} and t_r == {11}
        assert u_dir == (0, 1, 0)

    def test_3d_diagnostic_allows_vertical_slack_for_non_splittable_bridges(self) -> None:
        # 1-cut vertical bridge (11 -> 20 UP) with dx=3 would make a strict LP infeasible
        # unless vertical transverse slack is allowed so the LP stays feasible for splittable c_sp.
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11, 20, 30, 31)},
            edges=_bidir_edges(
                [
                    (1, 10, Direction.west, 3),
                    (10, 11, Direction.north, 3),
                    (11, 20, Direction.up, 1),
                    (20, 2, Direction.down, 1),
                    (1, 30, Direction.east, 1),
                    (30, 31, Direction.north, 1),
                    (31, 2, Direction.east, 1),
                ]
            ),
        )
        b_west = SolveTreeNode("b_west", TreeNodeType.BLOCK, frozenset({10, 11}), (), _bridge_meta(1, 10, Direction.west, 3))
        b_up = SolveTreeNode("b_up", TreeNodeType.BLOCK, frozenset({20}), (), _bridge_meta(2, 20, Direction.up, 1))
        c_sp = SolveTreeNode("c_sp", TreeNodeType.SERIES_LOOP, frozenset({30, 31}), (), _split_pair_meta(1, 2))
        sols = {
            "b_west": _node_sol("b_west", {10: Coord3D(0, 0, 0), 11: Coord3D(0, 3, 0)}, {1: Coord3D(3, 0, 0)}),
            "b_up": _node_sol("b_up", {20: Coord3D(0, 0, 0)}, {2: Coord3D(0, 0, -1)}),
            "c_sp": _node_sol("c_sp", {30: Coord3D(0, 0, 0), 31: Coord3D(0, 1, 0)}, {1: Coord3D(-1, 0, 0), 2: Coord3D(1, 1, 0)}),
        }
        parent = SolveTreeNode("p", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2}), (b_west, b_up, c_sp), {})
        inv = inventory_cluster(node=parent, child_solutions=sols, graph=graph, options=LayoutOptions())
        reqs = solve_3d_deformation_diagnostic(inv, [c_sp], graph=graph)
        assert reqs and reqs[0][0] == "c_sp" and reqs[0][3] == (0, 1, 0)

    def test_collision_diagnosis_ranks_pole_unclamping_and_cycle_weights(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11, 12, 13)},
            edges=_bidir_edges(
                [
                    (10, 1, Direction.east, 1),
                    (1, 11, Direction.east, 1),
                    (11, 12, Direction.east, 1),
                    (12, 2, Direction.east, 1),
                    (2, 13, Direction.north, 1),
                ]
            ),
        )
        big = SolveTreeNode("big", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12, 13}), (), _split_pair_meta(1, 2))
        parent = SolveTreeNode("p", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2}), (big,), {})
        big_sol = _node_sol(
            "big",
            {10: Coord3D(0, 0, 0), 11: Coord3D(2, 0, 0), 12: Coord3D(3, 0, 0), 13: Coord3D(5, 1, 0)},
            {1: Coord3D(1, 0, 0), 2: Coord3D(5, 0, 0)},
        )
        inv = ClusterInventory(
            node=parent,
            rooms=(1, 2),
            children=(big,),
            child_solutions={"big": big_sol},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("ae1", ("big", 10), 1, (1, 0, 0), 1, frozenset({1}), frozenset({"big"})),
                ActiveEdge("ae2", 1, ("big", 11), (1, 0, 0), 1, frozenset({1}), frozenset({"big"})),
                ActiveEdge("ae3", ("big", 12), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"big"})),
                ActiveEdge("ae4", ("big", 13), 2, (0, -1, 0), 1, frozenset({2}), frozenset({"big"})),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        exc = UnresolvableCollisionError(
            "blocked",
            node_id="p",
            collisions=(ClearanceViolation("child_child", ("big", 0), ("other", 0), "overlap", 1),),
            colliding_entities=(),
            pruned_reasons=(
                {"u": "Contradicted along axis 'z': requires a - b >= 1 (cycle weight 5 > 0)."},
                {"a": "Contradicted along axis 'y': requires a - b >= 1 (cycle weight 2 > 0)."},
            ),
            inventory=inv,
        )
        cands = diagnose_unresolvable_collision(exc, (big,), inv, {"big": big_sol}, graph)
        assert cands
        x_cands = [c for c in cands if c[3] in ((1, 0, 0), (-1, 0, 0))]
        assert x_cands[0][1] == {10} and x_cands[0][2] == {11, 12, 13}

    def test_shear_detection_edge_length_and_split_sibling_checks(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11)},
            edges=_bidir_edges([(1, 10, Direction.east, 1), (10, 11, Direction.north, 1), (11, 2, Direction.east, 1)]),
        )
        child = SolveTreeNode("c", TreeNodeType.SERIES_LOOP, frozenset({10, 11}), (), _split_pair_meta(1, 2))
        c_sol = _node_sol("c", {10: Coord3D(0, 0, 0), 11: Coord3D(0, 1, 0)}, {1: Coord3D(-1, 0, 0), 2: Coord3D(1, 1, 0)})
        parent = SolveTreeNode("p", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2}), (child,), {})
        inv = ClusterInventory(
            node=parent,
            rooms=(1, 2),
            children=(child,),
            child_solutions={"c": c_sol},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("sp_1_2", 1, 2, (1, 0, 0), 1, frozenset({1, 2}), frozenset({"c"})),
                ActiveEdge("e_shear", 1, ("c", 10), (1, 0, 0), 1, frozenset({1}), frozenset({"c"}), allow_transverse_slack=True),
                ActiveEdge("e_ok", ("c", 11), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"c"})),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        p_sol = NodeSolution(
            node_id="p",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(3, 2, 0)},
            metadata={"child_offsets": {"c": Coord3D(2, 1, 0)}, "all_room_coordinates": {}, "port_coordinates": {}},
            port_coordinates={},
        )
        sheared = detect_planar_transverse_shear(p_sol, inv, (child,), {"c": c_sol})
        assert len(sheared) == 1
        assert sheared[0][0].edge_id == "e_shear"
        assert sheared[0][1:] == (1, 1)
        assert compute_total_edge_length(p_sol, inv, {"c": c_sol}) == 4
        diag = diagnose_transverse_shear(p_sol, inv, (child,), {"c": c_sol})
        assert any(req[0] == "c" for req in diag)

        # Verify split-sibling edge detection even when one sibling has 1 room
        s_one = SolveTreeNode("c__s", TreeNodeType.RIGID_CLUSTER, frozenset({10}), (), {"reactive_split_origin": "c"})
        s_two = SolveTreeNode("c__t", TreeNodeType.RIGID_CLUSTER, frozenset({11, 12}), (), {"reactive_split_origin": "c", "boundary_ports": (1, 2)})
        sol_one = _node_sol("c__s", {10: Coord3D(0, 0, 0)})
        sol_two = _node_sol("c__t", {11: Coord3D(0, 0, 0), 12: Coord3D(1, 0, 0)})
        inv_sib = ClusterInventory(
            node=parent,
            rooms=(1, 2),
            children=(s_one, s_two),
            child_solutions={"c__s": sol_one, "c__t": sol_two},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("ccb_1", ("c__s", 10), ("c__t", 11), (1, 0, 0), 1, frozenset(), frozenset({"c__s", "c__t"}), allow_transverse_slack=True),
                ActiveEdge("e_p1", 1, ("c__t", 11), (1, 0, 0), 1, frozenset({1}), frozenset({"c__t"})),
                ActiveEdge("e_p2", ("c__t", 12), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"c__t"})),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        p_sib = NodeSolution(
            node_id="p",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(3, 2, 1)},
            metadata={"child_offsets": {"c__s": Coord3D(0, 0, 0), "c__t": Coord3D(1, 2, 1)}, "all_room_coordinates": {}, "port_coordinates": {}},
            port_coordinates={},
        )
        sib_shear = detect_planar_transverse_shear(p_sib, inv_sib, (s_one, s_two), {"c__s": sol_one, "c__t": sol_two})
        assert {(ae.edge_id, ax, v) for ae, ax, v in sib_shear if ae.edge_id == "ccb_1"} == {
            ("ccb_1", 1, 2),
            ("ccb_1", 2, 1),
        }

    def test_empty_split_pair_container_gateway_preservation_and_severed_edges(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11)},
            edges=_bidir_edges([(1, 10, Direction.east, 1), (10, 11, Direction.east, 1), (11, 2, Direction.east, 1)]),
        )
        piece_s = SolveTreeNode(
            "c__rs0_S",
            TreeNodeType.RIGID_CLUSTER,
            frozenset({10}),
            (),
            {**_bridge_meta(1, 10, Direction.east), "severed_bridge_edges": ((10, 11),), "reactive_split_origin": "c"},
        )
        piece_t = SolveTreeNode(
            "c__rs0_T",
            TreeNodeType.RIGID_CLUSTER,
            frozenset({11}),
            (),
            {**_bridge_meta(2, 11, Direction.west), "severed_bridge_edges": ((10, 11),), "reactive_split_origin": "c"},
        )
        container = SolveTreeNode(
            node_id="cont",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(piece_s, piece_t),
            metadata=_split_pair_meta(1, 2),
        )
        sols = {
            "c__rs0_S": _node_sol("c__rs0_S", {10: Coord3D(0, 0, 0)}, {1: Coord3D(-1, 0, 0)}),
            "c__rs0_T": _node_sol("c__rs0_T", {11: Coord3D(0, 0, 0)}, {2: Coord3D(1, 0, 0)}),
        }
        inv = inventory_cluster(node=container, child_solutions=sols, graph=graph, options=LayoutOptions())
        e_ids = {ae.edge_id for ae in inv.active_edges}
        assert "gw_c__rs0_S_1_10" in e_ids
        assert "gw_c__rs0_T_2_11" in e_ids
        assert "ccb_c__rs0_S_c__rs0_T_10_11" in e_ids

        # Also exercise boundary_ports on contract & metadata + severed gateway edges in both directions
        port_contract = SimpleNamespace(poles=(1,), boundary_ports=(99,))
        port_node = SimpleNamespace(
            node_id="pn",
            cut_contract=port_contract,
            metadata={"boundary_ports": (98,)},
        )
        assert {98, 99}.issubset(get_child_contract_interface_rooms(port_node))  # type: ignore[arg-type]

        parent_sev = SolveTreeNode(
            "p_sev",
            TreeNodeType.RIGID_CLUSTER,
            frozenset({1, 2}),
            (
                SolveTreeNode("ps", TreeNodeType.RIGID_CLUSTER, frozenset({10}), (), {"severed_bridge_edges": ((1, 10), (11, 2))}),
                SolveTreeNode("pt", TreeNodeType.RIGID_CLUSTER, frozenset({11}), (), {"severed_bridge_edges": ((1, 10), (11, 2))}),
            ),
            {},
        )
        gw = GatewayCutHandler(
            node=parent_sev,
            child_solutions={"ps": _node_sol("ps", {10: Coord3D(0, 0, 0)}), "pt": _node_sol("pt", {11: Coord3D(0, 0, 0)})},
            graph=graph,
            all_local_rooms=frozenset({1, 2}),
        )
        sev_edges, sev_anchors, sev_aligned = gw._process_severed_gateway_edges()
        assert len(sev_edges) == 2
        assert sev_anchors["ps"] == {1} and sev_anchors["pt"] == {2}
        assert sev_aligned == {"ps", "pt"}

    def test_end_to_end_reactive_split_and_resynthesis_across_nested_p_node(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21, 22)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.west, 1),
                    (3, 4, Direction.north, 4),
                    (4, 8, Direction.east, 5),
                    (8, 2, Direction.south, 2),
                    (1, 5, Direction.south, 1),
                    (5, 6, Direction.east, 5),
                    (6, 7, Direction.north, 3),
                    (7, 2, Direction.west, 1),
                    (1, 10, Direction.north, 1),
                    (10, 11, Direction.east, 1),
                    (11, 2, Direction.east, 1),
                    (1, 20, Direction.east, 1),
                    (20, 21, Direction.east, 1),
                    (21, 22, Direction.north, 1),
                    (22, 2, Direction.north, 1),
                ]
            ),
        )
        branch_a = SolveTreeNode("br_a", TreeNodeType.SERIES_LOOP, frozenset({10, 11}), (), _split_pair_meta(1, 2))
        branch_b = SolveTreeNode("br_b", TreeNodeType.SERIES_LOOP, frozenset({20, 21, 22}), (), _split_pair_meta(1, 2))
        nested_p = SolveTreeNode(
            node_id="nested_p",
            node_type=TreeNodeType.SPLIT_PAIR_CONTAINER,
            rooms=frozenset(),
            children=(branch_a, branch_b),
            metadata=_split_pair_meta(1, 2),
        )
        root = SolveTreeNode(
            node_id="root",
            node_type=TreeNodeType.RIGID_CLUSTER,
            rooms=frozenset({1, 2, 3, 4, 5, 6, 7, 8}),
            children=(nested_p,),
            metadata={},
        )
        tree = SolveTree(root=root, components=(root,), graph=graph)
        opts = LayoutOptions(timeout_seconds=15)
        tree_sol = HierarchicalTreeSolver().solve_tree(tree, opts)
        assert set(tree_sol.node_solutions) == {"root", "nested_p", "br_a", "br_b"}
        simplified = SimplifiedGraph(graph=graph, corridors=(), severed_chords=())
        assembled = HierarchicalLayoutAssembler().assemble(tree, tree_sol, simplified, opts)
        assert set(assembled.coordinates) == set(graph.rooms)
        assert len(set(assembled.coordinates.values())) == len(graph.rooms)

    def test_multi_round_split_produces_interior_middle_piece(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11, 12, 13, 14, 15)},
            edges=_bidir_edges(
                [
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.east, 1),
                    (11, 12, Direction.east, 1),
                    (12, 13, Direction.east, 1),
                    (13, 14, Direction.east, 1),
                    (14, 15, Direction.east, 1),
                    (15, 2, Direction.east, 1),
                ]
            ),
        )
        child = SolveTreeNode(
            "chain", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12, 13, 14, 15}), (), _split_pair_meta(1, 2)
        )
        r1 = split_child_along_direction(child, {10}, {15}, (1, 0, 0), {1, 2}, graph, "r0")
        assert r1 is not None
        piece_s, _piece_t, _ = r1
        assert set(piece_s.rooms) == {10, 11, 12}
        r2 = split_child_along_direction(piece_s, {10}, {12}, (1, 0, 0), {1, 2}, graph, "r1")
        assert r2 is not None
        left_p, mid_p, _ = r2
        assert isinstance(mid_p.cut_contract, BridgeCut)
        assert mid_p.cut_contract.anchor_vnum in left_p.rooms or mid_p.cut_contract.anchor_vnum == 13
        assert len(mid_p.metadata.get("severed_bridge_edges", ())) >= 2

    def test_empty_container_secondary_alignment_relaxed_in_diagnostic(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11, 20, 21)},
            edges=_bidir_edges(
                [
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.north, 1),
                    (11, 2, Direction.east, 1),
                    (1, 20, Direction.north, 1),
                    (20, 21, Direction.east, 2),
                    (21, 2, Direction.east, 1),
                ]
            ),
        )
        c1 = SolveTreeNode("c1", TreeNodeType.SERIES_LOOP, frozenset({10, 11}), (), _split_pair_meta(1, 2))
        c2 = SolveTreeNode("c2", TreeNodeType.SERIES_LOOP, frozenset({20, 21}), (), _split_pair_meta(1, 2))
        cont = SolveTreeNode("cont", TreeNodeType.SPLIT_PAIR_CONTAINER, frozenset(), (c1, c2), {})
        s1 = _node_sol("c1", {10: Coord3D(0, 0, 0), 11: Coord3D(0, 1, 0)}, {1: Coord3D(-1, 0, 0), 2: Coord3D(1, 1, 0)})
        s2 = _node_sol("c2", {20: Coord3D(0, 0, 0), 21: Coord3D(2, 0, 0)}, {1: Coord3D(-2, 0, 0), 2: Coord3D(1, 0, 0)})
        inv = ClusterInventory(
            node=cont,
            rooms=(),
            children=(c1, c2),
            child_solutions={"c1": s1, "c2": s2},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("e1", ("c1", 10), ("c2", 20), (1, 0, 0), 1, frozenset(), frozenset({"c1", "c2"})),
                ActiveEdge("e2", ("c1", 11), ("c2", 21), (1, 0, 0), 1, frozenset(), frozenset({"c1", "c2"})),
            ),
            child_alignments=(
                ChildChildAlignment("c1", "c2", Coord3D(-1, 0, 0), Coord3D(-2, 0, 0)),
                ChildChildAlignment("c1", "c2", Coord3D(1, 1, 0), Coord3D(1, 0, 0)),
            ),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        reqs = solve_3d_deformation_diagnostic(inv, [c1, c2])
        assert any(r[0] in ("c1", "c2") for r in reqs)

    def test_trigger_b_preserves_valid_sheared_layout_when_no_split_possible(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 4, 10)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.north, 1),
                    (3, 2, Direction.east, 2),
                    (1, 4, Direction.east, 2),
                    (4, 2, Direction.north, 1),
                    (1, 10, Direction.east, 1),
                    (10, 2, Direction.east, 1),
                ]
            ),
        )
        s_single = SolveTreeNode("s_single", TreeNodeType.SERIES_LOOP, frozenset({10}), (), _split_pair_meta(1, 2))
        root = SolveTreeNode("root", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2, 3, 4}), (s_single,), {})
        tree = SolveTree(root=root, components=(root,), graph=graph)
        sol = HierarchicalTreeSolver().solve_tree(tree, LayoutOptions(timeout_seconds=10))
        assert "root" in sol.node_solutions and "s_single" in sol.node_solutions

    def test_diagnose_transverse_shear_direct_inference_on_vertical_and_dy_edges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11)},
            edges=_bidir_edges([(1, 10, Direction.north, 1), (10, 11, Direction.east, 1), (11, 2, Direction.north, 1)]),
        )
        c = SolveTreeNode("c", TreeNodeType.SERIES_LOOP, frozenset({10, 11}), (), _split_pair_meta(1, 2))
        c_sol = _node_sol("c", {10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0)}, {1: Coord3D(0, -1, 0), 2: Coord3D(1, 1, 0)})
        parent = SolveTreeNode("p", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2}), (c,), {})
        inv = ClusterInventory(
            node=parent,
            rooms=(1, 2),
            children=(c,),
            child_solutions={"c": c_sol},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("e_n1", 1, ("c", 10), (0, 1, 0), 1, frozenset({1}), frozenset({"c"}), allow_transverse_slack=True),
                ActiveEdge("e_n2", ("c", 11), 2, (0, 1, 0), 1, frozenset({2}), frozenset({"c"}), allow_transverse_slack=True),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        p_sol = NodeSolution(
            node_id="p",
            local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(2, 2, 1)},
            metadata={"child_offsets": {"c": Coord3D(0, 1, 0)}, "all_room_coordinates": {}, "port_coordinates": {}},
            port_coordinates={},
        )
        sheared = detect_planar_transverse_shear(p_sol, inv, (c,), {"c": c_sol})
        assert {(ae.edge_id, ax) for ae, ax, _ in sheared} == {("e_n2", 0), ("e_n2", 2)}
        monkeypatch.setattr("romutil.layout.milp.reactive_split.solve_3d_deformation_diagnostic", lambda **_kw: [])
        cands = diagnose_transverse_shear(p_sol, inv, (c,), {"c": c_sol})
        assert any(cid == "c" and s == {10} and t == {11} for cid, s, t, _ in cands)

    def test_helper_edge_cases_and_z_axis_diagnostic(self) -> None:
        assert _safe_displacement(999) is None
        assert _as_coord3d((1, 2, 3)) == Coord3D(1, 2, 3)

        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 4, 10, 11, 12)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.east, 3),
                    (3, 2, Direction.north, 2),
                    (3, 4, Direction.up, 1),
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.east, 1),
                    (11, 12, Direction.up, 1),
                    (12, 2, Direction.east, 1),
                ]
            ),
        )
        child = SolveTreeNode("cz", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12}), (), _split_pair_meta(1, 2))
        c_sol = _node_sol(
            "cz",
            {10: Coord3D(0, 0, 0), 11: Coord3D(1, 0, 0), 12: Coord3D(1, 0, 1)},
            {1: Coord3D(-1, 0, 0), 2: Coord3D(2, 0, 1)},
        )
        parent = SolveTreeNode("pz", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2, 3, 4}), (child,), {})
        inv = ClusterInventory(
            node=parent,
            rooms=(1, 2, 3, 4),
            children=(child,),
            child_solutions={"cz": c_sol},
            boundary_ports=frozenset(),
            active_edges=(
                ActiveEdge("sp_port_1", 1, 3, (1, 0, 0), 3, frozenset({1, 3}), frozenset()),
                ActiveEdge("sp_span_cz", 1, 2, (1, 0, 0), 1, frozenset({1, 2}), frozenset({"cz"}), allow_transverse_slack=True),
                ActiveEdge("int_1_3", 1, 3, (1, 0, 0), 3, frozenset({1, 3}), frozenset()),
                ActiveEdge("int_3_2", 3, 2, (0, 1, 0), 2, frozenset({2, 3}), frozenset()),
                ActiveEdge("int_3_4", 3, 4, (0, 0, 1), 1, frozenset({3, 4}), frozenset()),
                ActiveEdge("gw_1_10", 1, ("cz", 10), (1, 0, 0), 1, frozenset({1}), frozenset({"cz"})),
                ActiveEdge("gw_12_2", ("cz", 12), 2, (1, 0, 0), 1, frozenset({2}), frozenset({"cz"})),
            ),
            child_alignments=(),
            gateway_anchors={},
            aligned_children=frozenset(),
            equated_children=frozenset(),
            graph=graph,
        )
        assert solve_3d_deformation_diagnostic(inv, []) == []
        reqs = solve_3d_deformation_diagnostic(inv, [child])
        assert reqs and reqs[0][0] == "cz" and reqs[0][3] in ((0, 1, 0), (0, 0, 1))

        any_ax, both_ax = _child_covered_exit_axes(child, None, graph)
        assert any_ax == {0} and both_ax == {0}
        assert get_child_interface_rooms(child, inv, {}) == ([], {})
        assert diagnose_unresolvable_collision(UnresolvableCollisionError("x", node_id="pz"), (child,), None, {"cz": c_sol}, graph) == []
        assert compute_split_lp_signature([], {10}, {11}, (1, 0, 0)) is None
        assert generate_coordinate_threshold_partitions(None, {10}, {11}, (1, 0, 0)) == []

    def test_cluster_unresolvable_collision_error_and_clearance_reasons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 1. ClearanceConstraintBuilder records unsatisfiable_collision_reasons read-only
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2)},
            edges=_bidir_edges([(1, 2, Direction.east, 1)]),
        )
        root = SolveTreeNode("r", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2}), (), {})
        inv = inventory_cluster(root, {}, graph, LayoutOptions())
        form = formulate_cluster_model(inv, LayoutOptions())
        cb = ClearanceConstraintBuilder(form)
        cb.spatial_dag.filter_candidate_directions = lambda *_a, **_kw: ([], {"r": "blocked (cycle weight 2 > 0)"})  # type: ignore[method-assign]
        cuts = cb.inject_clearance_cuts([ClearanceViolation("parent_room", 1, 2, "overlap", 1)])
        assert cuts == 0
        assert len(cb.unsatisfiable_collisions) == 1
        assert cb.unsatisfiable_collision_reasons == [{"r": "blocked (cycle weight 2 > 0)"}]

        # 2. _extract_colliding_entities extracts rooms, children, and corridor children
        active = (
            ActiveEdge("e0", 1, ("c1", 10), (1, 0, 0), 1, frozenset({1}), frozenset({"c1"})),
        )
        collisions = (
            ClearanceViolation("corridor_room", 0, SimpleNamespace(vnum=2), "cross", 1),
            ClearanceViolation("child_child", SimpleNamespace(child_id="c2"), ("c3", 30), "overlap", 1),
            ClearanceViolation("corridor_crossing", SimpleNamespace(edge_index=0), ("edge", 0), "cross", 1),
        )
        ents = _extract_colliding_entities(collisions, active)
        assert ("corridor", 0) in ents
        assert ("child", "c1") in ents
        assert ("room", 2) in ents
        assert ("child", "c2") in ents
        assert ("child", "c3") in ents

        # 3. _raise_collision_error raises UnresolvableCollisionError when unsatisfiable_collisions exist
        mock_form = SimpleNamespace(clearance_builder=cb, inventory=inv)
        with pytest.raises(UnresolvableCollisionError) as exc_info:
            _raise_collision_error(mock_form, "r", "deadlock")  # type: ignore[arg-type]
        assert exc_info.value.node_id == "r"
        assert exc_info.value.pruned_reasons == ({"r": "blocked (cycle weight 2 > 0)"},)

        # 4. solve_cluster_with_cutting_planes raises UnresolvableCollisionError on LayoutInfeasibleError with unsat
        monkeypatch.setattr(
            "romutil.layout.milp.cluster.solve_and_extract",
            lambda **_kw: (_ for _ in ()).throw(LayoutInfeasibleError("infeasible")),
        )
        form.clearance_builder.unsatisfiable_collisions.append(ClearanceViolation("parent_room", 1, 2, "overlap", 1))
        form.clearance_builder.unsatisfiable_collision_reasons.append({"r": "blocked"})
        with pytest.raises(UnresolvableCollisionError):
            solve_cluster_with_cutting_planes(formulation=form, node=root)

    def test_trigger_a_unresolvable_collision_reactive_split_and_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 10, 11, 12)},
            edges=_bidir_edges(
                [
                    (1, 2, Direction.east, 4),
                    (1, 10, Direction.north, 1),
                    (10, 11, Direction.east, 2),
                    (11, 12, Direction.east, 2),
                    (12, 2, Direction.south, 1),
                ]
            ),
        )
        c = SolveTreeNode("c", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12}), (), _split_pair_meta(1, 2))
        root = SolveTreeNode("root", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2}), (c,), {})
        tree = SolveTree(root=root, components=(root,), graph=graph)

        import romutil.layout.milp.tree as tree_mod

        orig_solve_cluster = tree_mod.solve_cluster
        calls: list[tuple[str, ...]] = []

        def _wrapped_solve(*args: object, **kwargs: object) -> NodeSolution:
            node = kwargs.get("node") or args[0]
            if isinstance(node, SolveTreeNode) and node.node_id == "root":
                cids = tuple(ch.node_id for ch in node.children)
                calls.append(cids)
                if cids == ("c",):
                    inv = inventory_cluster(node, kwargs.get("child_solutions", {}), graph, LayoutOptions())  # type: ignore[arg-type]
                    raise UnresolvableCollisionError(
                        "deadlock",
                        node_id="root",
                        collisions=(ClearanceViolation("child_room", ("c", 0), 1, "overlap", 1),),
                        colliding_entities=(("child", "c"), ("room", 1)),
                        pruned_reasons=({"r": "Contradicted along axis 'x': requires a - b >= 1 (cycle weight 1 > 0)."},),
                        inventory=inv,
                    )
            return orig_solve_cluster(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(tree_mod, "solve_cluster", _wrapped_solve)
        sol = HierarchicalTreeSolver().solve_tree(tree, LayoutOptions(timeout_seconds=10))
        assert set(sol.node_solutions) == {"root", "c"}
        assert calls[0] == ("c",) and len(calls) >= 2 and any("__rs" in cid for cid in calls[1])

        # When UnresolvableCollisionError has no splittable candidate and no best_valid_state, re-raise
        def _always_fail(*args: object, **kwargs: object) -> NodeSolution:
            node = kwargs.get("node") or args[0]
            if isinstance(node, SolveTreeNode) and node.node_id == "root":
                raise UnresolvableCollisionError("unrecoverable", node_id="root", colliding_entities=())
            return orig_solve_cluster(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(tree_mod, "solve_cluster", _always_fail)
        with pytest.raises(UnresolvableCollisionError):
            HierarchicalTreeSolver().solve_tree(tree, LayoutOptions(timeout_seconds=10))

    def test_diagnostic_gates_impossible_directions_and_routes_to_cuttable_sibling(self) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 4, 10, 11, 12, 20, 21)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.north, 3),
                    (3, 2, Direction.east, 4),
                    (1, 10, Direction.east, 1),
                    (10, 12, Direction.south, 1),
                    (12, 11, Direction.south, 1),
                    (11, 4, Direction.east, 1),
                    (4, 20, Direction.east, 1),
                    (20, 21, Direction.north, 1),
                    (21, 2, Direction.east, 1),
                ]
            ),
        )
        c_locked = SolveTreeNode(
            "c_locked", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12}), (), _split_pair_meta(1, 4)
        )
        c_cuttable = SolveTreeNode(
            "c_cuttable", TreeNodeType.SERIES_LOOP, frozenset({20, 21}), (), _split_pair_meta(4, 2)
        )
        sol_locked = _node_sol(
            "c_locked",
            {10: Coord3D(0, 2, 0), 12: Coord3D(0, 1, 0), 11: Coord3D(0, 0, 0)},
            {1: Coord3D(-1, 2, 0), 4: Coord3D(1, 0, 0)},
        )
        sol_cut = _node_sol(
            "c_cuttable",
            {20: Coord3D(0, 0, 0), 21: Coord3D(0, 1, 0)},
            {4: Coord3D(-1, 0, 0), 2: Coord3D(1, 1, 0)},
        )
        parent = SolveTreeNode(
            "parent", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2, 3, 4}), (c_locked, c_cuttable), {}
        )
        inv = inventory_cluster(
            node=parent,
            child_solutions={"c_locked": sol_locked, "c_cuttable": sol_cut},
            graph=graph,
            options=LayoutOptions(),
        )
        reqs = solve_3d_deformation_diagnostic(inv, [c_locked, c_cuttable], graph=graph)
        assert reqs
        assert not any(cid == "c_locked" and u_dir != (0, 1, 0) for cid, _s, _t, u_dir in reqs)
        assert not any(cid == "c_locked" and 10 in s and 11 in t for cid, s, t, _u in reqs)
        assert any(
            cid == "c_cuttable" and s == {20} and t == {21} and u_dir == (0, 1, 0)
            for cid, s, t, u_dir in reqs
        )

    def test_failed_splits_retry_routes_diagnostic_to_alternative_sibling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 4, 10, 11, 12, 20, 21)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.north, 3),
                    (3, 2, Direction.east, 5),
                    (1, 10, Direction.east, 1),
                    (10, 12, Direction.north, 1),
                    (12, 11, Direction.east, 1),
                    (11, 4, Direction.east, 1),
                    (4, 20, Direction.east, 1),
                    (20, 21, Direction.north, 1),
                    (21, 2, Direction.east, 1),
                ]
            ),
        )
        c_first = SolveTreeNode(
            "c_first", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12}), (), _split_pair_meta(1, 4)
        )
        c_second = SolveTreeNode(
            "c_second", TreeNodeType.SERIES_LOOP, frozenset({20, 21}), (), _split_pair_meta(4, 2)
        )
        root = SolveTreeNode(
            "root", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2, 3, 4}), (c_first, c_second), {}
        )
        tree = SolveTree(root=root, components=(root,), graph=graph)

        import romutil.layout.milp.reactive_split as rs_mod
        import romutil.layout.milp.tree as tree_mod

        orig_split = tree_mod.split_child_along_direction
        orig_diag = rs_mod.solve_3d_deformation_diagnostic
        seen_failed_snapshots: list[set[tuple[str, int, int]]] = []
        split_targets: list[str] = []

        def _wrapped_diag(*args: object, **kwargs: object) -> list:
            fs = kwargs.get("failed_splits")
            seen_failed_snapshots.append(set(fs) if isinstance(fs, set) else set())
            return orig_diag(*args, **kwargs)  # type: ignore[arg-type]

        def _wrapped_split(child: SolveTreeNode, *args: object, **kwargs: object):
            split_targets.append(child.node_id)
            if child.node_id == "c_first":
                return None
            return orig_split(child, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(rs_mod, "solve_3d_deformation_diagnostic", _wrapped_diag)
        monkeypatch.setattr(tree_mod, "split_child_along_direction", _wrapped_split)
        sol = HierarchicalTreeSolver().solve_tree(tree, LayoutOptions(timeout_seconds=15))
        assert set(sol.node_solutions) == {"root", "c_first", "c_second"}
        assert any(("c_first", 1, 1) in snap for snap in seen_failed_snapshots)
        assert "c_second" in split_targets

    def test_trigger_b_fallback_restores_best_valid_state_on_post_split_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph = LayoutGraph(
            rooms={v: _room(v) for v in (1, 2, 3, 10, 11, 12)},
            edges=_bidir_edges(
                [
                    (1, 3, Direction.north, 3),
                    (3, 2, Direction.east, 4),
                    (1, 10, Direction.east, 1),
                    (10, 11, Direction.north, 1),
                    (11, 12, Direction.east, 1),
                    (12, 2, Direction.east, 1),
                ]
            ),
        )
        c = SolveTreeNode("c", TreeNodeType.SERIES_LOOP, frozenset({10, 11, 12}), (), _split_pair_meta(1, 2))
        root = SolveTreeNode("root", TreeNodeType.RIGID_CLUSTER, frozenset({1, 2, 3}), (c,), {})
        tree = SolveTree(root=root, components=(root,), graph=graph)

        import romutil.layout.milp.tree as tree_mod

        orig_solve_cluster = tree_mod.solve_cluster
        observed_children: list[tuple[str, ...]] = []

        def _wrapped_solve(*args: object, **kwargs: object) -> NodeSolution:
            node = kwargs.get("node") or args[0]
            if isinstance(node, SolveTreeNode) and node.node_id == "root":
                child_ids = tuple(ch.node_id for ch in node.children)
                observed_children.append(child_ids)
                if any("__rs" in cid for cid in child_ids):
                    raise LayoutSolverError("forced post-split failure")
            return orig_solve_cluster(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(tree_mod, "solve_cluster", _wrapped_solve)
        sol = HierarchicalTreeSolver().solve_tree(tree, LayoutOptions(timeout_seconds=15))
        assert set(sol.node_solutions) == {"root", "c"}
        assert observed_children == [("c",), ("c__rs0_S", "c__rs0_T")]

    @pytest.mark.regression
    def test_catacombs_2057_nested_p_node_stretch_hybrid_reactive_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify Stage 3 keeps nested P-node intact and Stage 4 Trigger B resolves stretch cleanly."""
        vnums = [
            2001, 2002, 2003, 2029, 2030, 2031, 2033, 2034, 2036, 2039, 2040,
            2042, 2045, 2047, 2048, 2050, 2056, 2057, 2059, 2060, 2061, 2062,
        ]
        rooms = {v: _room(v) for v in vnums}
        pairs = [
            (2030, 2001, Direction.up, 1),
            (2001, 2002, Direction.south, 6),
            (2002, 2003, Direction.east, 5),
            (2003, 2062, Direction.down, 1),
            (2030, 2031, Direction.east, 1),
            (2031, 2034, Direction.south, 1),
            (2034, 2050, Direction.south, 2),
            (2034, 2036, Direction.east, 2),
            (2036, 2045, Direction.south, 3),
            (2045, 2059, Direction.south, 1),
            (2050, 2059, Direction.east, 2),
            (2036, 2039, Direction.east, 2),
            (2039, 2047, Direction.south, 1),
            (2047, 2062, Direction.south, 3),
            (2059, 2060, Direction.east, 1),
            (2060, 2061, Direction.south, 1),
            (2061, 2062, Direction.east, 1),
            # Nested spqr_2 (poles 2030, 2050) -> spqr_3 (poles 2048, 2050) -> spqr_4 [2056, 2057]
            (2030, 2029, Direction.west, 1),
            (2029, 2033, Direction.west, 1),
            (2033, 2042, Direction.south, 1),
            (2040, 2042, Direction.east, 2),
            (2040, 2048, Direction.south, 1),
            (2048, 2050, Direction.east, 2),
            (2048, 2056, Direction.south, 1),
            (2056, 2057, Direction.east, 1),
            (2057, 2050, Direction.north, 1),
        ]
        graph = LayoutGraph(rooms=rooms, edges=_bidir_edges(pairs))
        simplified = SimplifiedGraph(graph=graph, corridors=(), severed_chords=())
        tree = PipelineTreeBuilder().build_tree(simplified)

        def _find_node(node: SolveTreeNode, target_rooms: set[int]) -> SolveTreeNode | None:
            if set(node.rooms) == target_rooms:
                return node
            for ch in node.children:
                found = _find_node(ch, target_rooms)
                if found is not None:
                    return found
            return None

        spqr_2 = _find_node(tree.root, {2029, 2033, 2040, 2042, 2048})
        spqr_4 = _find_node(tree.root, {2056, 2057})
        assert spqr_2 is not None and spqr_4 is not None
        assert isinstance(spqr_2.cut_contract, SplitPairCut)
        assert isinstance(spqr_4.cut_contract, SplitPairCut)

        import romutil.layout.milp.tree as tree_mod

        orig_split = tree_mod.split_child_along_direction
        recorded_splits: list[tuple[str, tuple[int, int, int], tuple[str, ...]]] = []

        def _spy_split(*args: object, **kwargs: object):
            res = orig_split(*args, **kwargs)  # type: ignore[arg-type]
            child = kwargs.get("child") or args[0]
            direction = kwargs.get("direction") or args[3]
            if res is not None and isinstance(child, SolveTreeNode):
                recorded_splits.append(
                    (child.node_id, direction, tuple(u.node_id for u in res[2]))  # type: ignore[arg-type]
                )
            return res

        monkeypatch.setattr(tree_mod, "split_child_along_direction", _spy_split)

        opts = LayoutOptions(timeout_seconds=15)
        tree_sol = HierarchicalTreeSolver().solve_tree(tree, opts)
        assembled = HierarchicalLayoutAssembler().assemble(tree, tree_sol, simplified, opts)

        assert assembled.diagnostics.status == SolverStatus.OPTIMAL
        assert len(set(assembled.coordinates.values())) == len(rooms)
        assert any(spqr_4.node_id in unpacked for _cid, _d, unpacked in recorded_splits)

        for u, v, d, w in pairs:
            cu, cv = assembled.coordinates[u], assembled.coordinates[v]
            dx, dy, dz = direction_to_displacement(d)
            vx, vy, vz = cv.x - cu.x, cv.y - cu.y, cv.z - cu.z
            dist = vx * dx + vy * dy + vz * dz
            assert dist >= w
            assert (vx, vy, vz) == (dist * dx, dist * dy, dist * dz)

        c2057, c2050 = assembled.coordinates[2057], assembled.coordinates[2050]
        c2029, c2030 = assembled.coordinates[2029], assembled.coordinates[2030]
        c2048 = assembled.coordinates[2048]
        assert (c2050.x - c2057.x, c2050.y - c2057.y, c2050.z - c2057.z) == (0, 1, 0)
        assert (c2030.x - c2029.x, c2030.y - c2029.y, c2030.z - c2029.z) == (1, 0, 0)
        assert (c2050.x - c2048.x, c2050.y - c2048.y, c2050.z - c2048.z) == (5, 0, 0)
