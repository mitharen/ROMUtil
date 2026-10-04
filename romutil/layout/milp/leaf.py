"""MILP formulation and solver for irreducible leaf clusters (R-nodes and S-nodes)."""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import pyomo.environ as pyo
from pyomo.opt import SolverStatus as PyomoSolverStatus, TerminationCondition

from romutil.layout.geometry.primitives import BoundingBox3D
from romutil.layout.ir import (
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.protocols import TreeSolver
from romutil.layout.simplification.warps import direction_to_displacement
from romutil.models import RoomDef
from romutil.types import Coord3D, LayoutOptions


def _find_free_pos(
    preferred: tuple[int, int, int],
    occupied: set[tuple[int, int, int]],
    step: tuple[int, int, int] = (1, 0, 0),
) -> tuple[int, int, int]:
    """Find the nearest unoccupied integer coordinate starting from preferred."""
    if preferred not in occupied:
        return preferred

    cx, cy, cz = preferred
    dx, dy, dz = step if step != (0, 0, 0) else (1, 0, 0)
    dist = 1
    while True:
        candidate = (cx + dx * dist, cy + dy * dist, cz + dz * dist)
        if candidate not in occupied:
            return candidate
        dist += 1


def _solve_leaf_fallback(
    node: SolveTreeNode,
    graph: LayoutGraph,
    active_edges: Sequence[tuple[LayoutEdge, tuple[int, int, int]]] | None = None,
) -> NodeSolution:
    """Deterministic fallback solver using BFS propagation along exit directions."""
    rooms = sorted(node.rooms)
    if not rooms:
        return NodeSolution(node_id=node.node_id, local_coordinates={}, bounding_box=None)
    if len(rooms) == 1:
        r = rooms[0]
        return NodeSolution(
            node_id=node.node_id,
            local_coordinates={r: Coord3D(0, 0, 0)},
            bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)),
        )

    if active_edges is None:
        room_set = set(rooms)
        parsed_edges: list[tuple[LayoutEdge, tuple[int, int, int]]] = []
        for e in graph.edges:
            if not e.cut and e.src in room_set and e.dst in room_set:
                try:
                    disp = direction_to_displacement(e.direction)
                    parsed_edges.append((e, disp))
                except ValueError:
                    continue
        active_edges = parsed_edges

    adj: dict[int, list[tuple[int, tuple[int, int, int], int, bool]]] = {u: [] for u in rooms}
    for e, disp in active_edges:
        w = max(1, e.weight)
        adj[e.src].append((e.dst, disp, w, False))
        adj[e.dst].append((e.src, disp, w, True))

    coords: dict[int, tuple[int, int, int]] = {}
    occupied: set[tuple[int, int, int]] = set()

    for r in rooms:
        if r in coords:
            continue
        if not coords:
            start_pos = (0, 0, 0)
        else:
            max_x = max(p[0] for p in occupied)
            start_pos = (max_x + 2, 0, 0)

        pos = _find_free_pos(start_pos, occupied)
        coords[r] = pos
        occupied.add(pos)

        queue: deque[int] = deque([r])
        while queue:
            curr = queue.popleft()
            cx, cy, cz = coords[curr]
            for nbr, (dx, dy, dz), w, is_rev in adj[curr]:
                if nbr not in coords:
                    step_x, step_y, step_z = (-dx, -dy, -dz) if is_rev else (dx, dy, dz)
                    target = (cx + step_x * w, cy + step_y * w, cz + step_z * w)
                    pos = _find_free_pos(target, occupied, (step_x, step_y, step_z))
                    coords[nbr] = pos
                    occupied.add(pos)
                    queue.append(nbr)

    min_x = min(x for x, y, z in coords.values())
    min_y = min(y for x, y, z in coords.values())
    min_z = min(z for x, y, z in coords.values())

    local_coords = {
        u: Coord3D(x - min_x, y - min_y, z - min_z)
        for u, (x, y, z) in coords.items()
    }
    bbox = BoundingBox3D.from_coordinates(local_coords.values())
    min_corner = Coord3D(bbox.min_x, bbox.min_y, bbox.min_z)
    max_corner = Coord3D(bbox.max_x, bbox.max_y, bbox.max_z)
    return NodeSolution(
        node_id=node.node_id,
        local_coordinates=local_coords,
        bounding_box=(min_corner, max_corner),
    )


def solve_leaf_cluster(
    node: SolveTreeNode,
    graph: LayoutGraph,
    options: LayoutOptions | None = None,
    solver_name: str = "cbc",
) -> NodeSolution:
    """Solve 3D integer coordinates for all rooms in a leaf cluster via MILP.

    Args:
        node: Leaf SolveTreeNode representing the cluster to lay out.
        graph: Underlying LayoutGraph with room definitions and exit edges.
        options: Optional layout configuration parameters (e.g. timeout).
        solver_name: Name of the Pyomo MILP solver (defaults to 'cbc').

    Returns:
        NodeSolution containing solved non-overlapping 3D coordinates and bounding box.
    """
    rooms = sorted(node.rooms)
    if not rooms:
        return NodeSolution(node_id=node.node_id, local_coordinates={}, bounding_box=None)

    if len(rooms) == 1:
        r = rooms[0]
        return NodeSolution(
            node_id=node.node_id,
            local_coordinates={r: Coord3D(0, 0, 0)},
            bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)),
        )

    room_set = set(rooms)
    active_edges: list[tuple[LayoutEdge, tuple[int, int, int]]] = []
    for edge in graph.edges:
        if not edge.cut and edge.src in room_set and edge.dst in room_set:
            try:
                disp = direction_to_displacement(edge.direction)
                active_edges.append((edge, disp))
            except ValueError:
                continue

    try:
        solver = pyo.SolverFactory(solver_name)
        if not solver.available():
            return _solve_leaf_fallback(node, graph, active_edges)
    except Exception:
        return _solve_leaf_fallback(node, graph, active_edges)

    try:
        model = pyo.ConcreteModel()
        anchor = rooms[0]
        m_coord = max(1000, 100 * len(rooms))
        m_big = 2 * m_coord

        model.x = pyo.Var(rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y = pyo.Var(rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z = pyo.Var(rooms, domain=pyo.Integers, bounds=(-m_coord, m_coord))

        model.x[anchor].fix(0)
        model.y[anchor].fix(0)
        model.z[anchor].fix(0)

        # Exit vector edge constraints: (xv - xu, yv - yu, zv - zu) = Le * d
        edge_indices = list(range(len(active_edges)))
        model.L = pyo.Var(
            edge_indices,
            domain=pyo.Integers,
            bounds=lambda m, i: (max(1, active_edges[i][0].weight), m_coord),
        )

        model.exit_x = pyo.ConstraintList()
        model.exit_y = pyo.ConstraintList()
        model.exit_z = pyo.ConstraintList()

        for i, (edge, (dx, dy, dz)) in enumerate(active_edges):
            model.exit_x.add(model.x[edge.dst] - model.x[edge.src] == model.L[i] * dx)
            model.exit_y.add(model.y[edge.dst] - model.y[edge.src] == model.L[i] * dy)
            model.exit_z.add(model.z[edge.dst] - model.z[edge.src] == model.L[i] * dz)

        # Hard non-overlap separation for all distinct pairs u < v
        pairs = [
            (rooms[i], rooms[j])
            for i in range(len(rooms))
            for j in range(i + 1, len(rooms))
        ]

        model.b_l = pyo.Var(pairs, domain=pyo.Binary)
        model.b_r = pyo.Var(pairs, domain=pyo.Binary)
        model.b_b = pyo.Var(pairs, domain=pyo.Binary)
        model.b_a = pyo.Var(pairs, domain=pyo.Binary)
        model.b_d = pyo.Var(pairs, domain=pyo.Binary)
        model.b_u = pyo.Var(pairs, domain=pyo.Binary)

        model.sep_sum = pyo.ConstraintList()
        model.sep_r = pyo.ConstraintList()
        model.sep_l = pyo.ConstraintList()
        model.sep_a = pyo.ConstraintList()
        model.sep_b = pyo.ConstraintList()
        model.sep_u = pyo.ConstraintList()
        model.sep_d = pyo.ConstraintList()

        for u, v in pairs:
            model.sep_sum.add(
                model.b_l[u, v]
                + model.b_r[u, v]
                + model.b_b[u, v]
                + model.b_a[u, v]
                + model.b_d[u, v]
                + model.b_u[u, v]
                >= 1
            )
            # Right: xu - xv >= 1
            model.sep_r.add(model.x[u] - model.x[v] >= 1 - 2 * m_big * (1 - model.b_r[u, v]))
            # Left: xv - xu >= 1
            model.sep_l.add(model.x[v] - model.x[u] >= 1 - 2 * m_big * (1 - model.b_l[u, v]))
            # Ahead/North: yu - yv >= 1
            model.sep_a.add(model.y[u] - model.y[v] >= 1 - 2 * m_big * (1 - model.b_a[u, v]))
            # Behind/South: yv - yu >= 1
            model.sep_b.add(model.y[v] - model.y[u] >= 1 - 2 * m_big * (1 - model.b_b[u, v]))
            # Up: zu - zv >= 1
            model.sep_u.add(model.z[u] - model.z[v] >= 1 - 2 * m_big * (1 - model.b_u[u, v]))
            # Down: zv - zu >= 1
            model.sep_d.add(model.z[v] - model.z[u] >= 1 - 2 * m_big * (1 - model.b_d[u, v]))

        # Bounding box variables and constraints
        model.x_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.x_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.y_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z_min = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))
        model.z_max = pyo.Var(domain=pyo.Integers, bounds=(-m_coord, m_coord))

        model.bbox_bounds = pyo.ConstraintList()
        for u in rooms:
            model.bbox_bounds.add(model.x_max >= model.x[u])
            model.bbox_bounds.add(model.x_min <= model.x[u])
            model.bbox_bounds.add(model.y_max >= model.y[u])
            model.bbox_bounds.add(model.y_min <= model.y[u])
            model.bbox_bounds.add(model.z_max >= model.z[u])
            model.bbox_bounds.add(model.z_min <= model.z[u])

        stretch_expr = sum(
            10 * (model.L[i] - max(1, active_edges[i][0].weight))
            for i in edge_indices
        )
        span_expr = (
            (model.x_max - model.x_min)
            + (model.y_max - model.y_min)
            + (model.z_max - model.z_min)
        )
        model.obj = pyo.Objective(expr=stretch_expr + span_expr, sense=pyo.minimize)

        if options and options.timeout_seconds:
            if solver_name == "cbc":
                solver.options["seconds"] = options.timeout_seconds
            elif solver_name == "glpk":
                solver.options["tmlim"] = options.timeout_seconds

        results = solver.solve(model, tee=False)

        valid_status = (
            results.solver.status == PyomoSolverStatus.ok
            and results.solver.termination_condition
            in (
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit,
            )
        )
        if not valid_status:
            return _solve_leaf_fallback(node, graph, active_edges)

        raw_coords: dict[int, tuple[int, int, int]] = {}
        for u in rooms:
            vx = pyo.value(model.x[u])
            vy = pyo.value(model.y[u])
            vz = pyo.value(model.z[u])
            if vx is None or vy is None or vz is None:
                return _solve_leaf_fallback(node, graph, active_edges)
            raw_coords[u] = (int(round(vx)), int(round(vy)), int(round(vz)))

        # Normalization: shift min_x, min_y, min_z to 0
        min_x = min(x for x, y, z in raw_coords.values())
        min_y = min(y for x, y, z in raw_coords.values())
        min_z = min(z for x, y, z in raw_coords.values())

        local_coords = {
            u: Coord3D(x - min_x, y - min_y, z - min_z)
            for u, (x, y, z) in raw_coords.items()
        }
        bbox = BoundingBox3D.from_coordinates(local_coords.values())
        min_corner = Coord3D(bbox.min_x, bbox.min_y, bbox.min_z)
        max_corner = Coord3D(bbox.max_x, bbox.max_y, bbox.max_z)
        return NodeSolution(
            node_id=node.node_id,
            local_coordinates=local_coords,
            bounding_box=(min_corner, max_corner),
        )
    except Exception:
        return _solve_leaf_fallback(node, graph, active_edges)


class LeafTreeSolver(TreeSolver):
    """Hierarchical tree solver that solves leaf clusters via MILP and aggregates non-leaf boxes."""

    def __init__(
        self,
        graph: LayoutGraph | None = None,
        solver_name: str = "cbc",
    ) -> None:
        self.graph = graph
        self.solver_name = solver_name

    def solve_tree(
        self,
        tree: SolveTree,
        options: LayoutOptions | None = None,
    ) -> TreeSolution:
        """Solve local coordinates for leaf nodes and enclose non-leaf bounding boxes."""
        layout_graph = self.graph if self.graph is not None else tree.graph
        if layout_graph is None:
            all_rooms: set[int] = set()

            def _collect_rooms(n: SolveTreeNode) -> None:
                all_rooms.update(n.rooms)
                for ch in n.children:
                    _collect_rooms(ch)

            _collect_rooms(tree.root)
            for comp in tree.components:
                _collect_rooms(comp)

            layout_graph = LayoutGraph(
                rooms={v: RoomDef(vnum=v, name=f"Room {v}", description="", exits=()) for v in all_rooms},
                edges=(),
            )

        solutions: dict[str, NodeSolution] = {}

        def _solve_node(node: SolveTreeNode) -> NodeSolution:
            if node.node_id in solutions:
                return solutions[node.node_id]

            if not node.children:
                sol = solve_leaf_cluster(
                    node,
                    layout_graph,
                    options,
                    solver_name=self.solver_name,
                )
                solutions[node.node_id] = sol
                return sol

            child_solutions: list[NodeSolution] = [_solve_node(child) for child in node.children]

            child_corners: list[Coord3D] = []
            child_coords: dict[int, Coord3D] = {}
            for c_sol in child_solutions:
                child_coords.update(c_sol.local_coordinates)
                if c_sol.bounding_box is not None:
                    child_corners.extend(c_sol.bounding_box)

            if child_corners:
                enclosing_box = BoundingBox3D.from_coordinates(child_corners)
                bbox: tuple[Coord3D, Coord3D] | None = (
                    Coord3D(enclosing_box.min_x, enclosing_box.min_y, enclosing_box.min_z),
                    Coord3D(enclosing_box.max_x, enclosing_box.max_y, enclosing_box.max_z),
                )
            else:
                bbox = None

            node_local_coords = {
                v: child_coords.get(v, Coord3D(0, 0, 0))
                for v in node.rooms
            }

            sol = NodeSolution(
                node_id=node.node_id,
                local_coordinates=node_local_coords,
                bounding_box=bbox,
            )
            solutions[node.node_id] = sol
            return sol

        _solve_node(tree.root)
        for comp in tree.components:
            if comp.node_id not in solutions:
                _solve_node(comp)

        return TreeSolution(node_solutions=solutions)
