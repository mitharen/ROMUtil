"""Non-Euclidean warp cycle detection and minimal chord severing."""

from __future__ import annotations

import collections
from dataclasses import replace
from fractions import Fraction
from typing import Sequence

import networkx as nx

from romutil.layout.ir import (
    CorridorSegment,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
)
from romutil.layout.protocols import GraphSimplifier
from romutil.models import Direction
from romutil.types import LayoutOptions

_DIRECTION_OFFSETS: dict[Direction, tuple[int, int, int]] = {
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


def direction_to_displacement(direction: int | Direction) -> tuple[int, int, int]:
    """Convert an exit direction or integer into a 3D displacement vector (dx, dy, dz).

    Args:
        direction: Direction enum member or integer representation.

    Returns:
        3D unit integer displacement tuple (dx, dy, dz).

    Raises:
        ValueError: If direction is not a recognized Direction value.
    """
    if isinstance(direction, Direction):
        d = direction
    else:
        d = Direction(direction)
    return _DIRECTION_OFFSETS[d]


def is_euclidean_satisfiable(cycle_vectors: Sequence[tuple[int, int, int]]) -> bool:
    """Determine if a cycle of direction vectors is Euclidean satisfiable.

    A cycle is Euclidean satisfiable if there exist positive integer edge lengths
    L_e >= 1 such that sum_{e in C} L_e * d_e = (0, 0, 0).

    Elastic loops that can mathematically close (even with unequal lengths)
    are Euclidean-satisfiable and must not be cut.

    Args:
        cycle_vectors: Sequence of 3D displacement vectors along the cycle.

    Returns:
        True if the cycle can close with positive edge lengths >= 1, False otherwise.
    """
    if not cycle_vectors:
        return False

    # Check if sum is already zero with uniform edge lengths L_e = 1
    sx = sum(v[0] for v in cycle_vectors)
    sy = sum(v[1] for v in cycle_vectors)
    sz = sum(v[2] for v in cycle_vectors)
    if sx == 0 and sy == 0 and sz == 0:
        return True

    # Sign check: along each axis with non-zero motion, there must be at least one
    # positive step and at least one negative step.
    for d in range(3):
        vals = [v[d] for v in cycle_vectors if v[d] != 0]
        if vals:
            if all(x > 0 for x in vals) or all(x < 0 for x in vals):
                return False

    # If all vectors are strictly axis-aligned (at most one non-zero component each),
    # the coordinate equations decouple and the existence of positive and negative
    # steps along every active axis is necessary and sufficient.
    is_axis_aligned = all(
        sum(1 for d in range(3) if v[d] != 0) <= 1 for v in cycle_vectors
    )
    if is_axis_aligned:
        return True

    # General feasibility check for coupled/diagonal cycles via Phase-I Simplex
    # Let L_e = 1 + s_e where s_e >= 0.
    # The system is sum s_e * d_e = -sum d_e =: b with s_e >= 0.
    k = len(cycle_vectors)
    b = [-sx, -sy, -sz]

    rows_a: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for d in range(3):
        row = [cycle_vectors[i][d] for i in range(k)]
        target = b[d]
        if all(x == 0 for x in row):
            continue
        if target < 0:
            row = [-x for x in row]
            target = -target
        rows_a.append([Fraction(x) for x in row])
        rhs.append(Fraction(target))

    m = len(rows_a)

    # Build Phase-I tableau: m constraint rows + 1 objective row
    # Columns: s_0..s_{k-1}, a_0..a_{m-1}, RHS
    tableau: list[list[Fraction]] = []
    for r in range(m):
        row_cells = (
            list(rows_a[r])
            + [Fraction(1) if i == r else Fraction(0) for i in range(m)]
            + [rhs[r]]
        )
        tableau.append(row_cells)

    # Objective row: minimize W = sum a_r.
    # In terms of non-basics: W + sum_j (sum_r A_{r,j}) s_j = sum b_r
    obj_row: list[Fraction] = (
        [sum((rows_a[r][j] for r in range(m)), start=Fraction(0)) for j in range(k)]
        + [Fraction(0) for _ in range(m)]
        + [sum(rhs, start=Fraction(0))]
    )
    tableau.append(obj_row)

    while True:
        if tableau[m][-1] == 0:
            return True

        # Bland's rule: smallest index j with reduced cost > 0
        entering_col = -1
        for j in range(k):
            if tableau[m][j] > 0:
                entering_col = j
                break

        if entering_col == -1:
            return tableau[m][-1] == 0

        # Ratio test with Bland's rule for tie-breaking
        leaving_row = -1
        min_ratio: Fraction | None = None
        for r in range(m):
            val = tableau[r][entering_col]
            if val > 0:
                ratio = tableau[r][-1] / val
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    leaving_row = r

        # Pivot on (leaving_row, entering_col)
        pivot_val = tableau[leaving_row][entering_col]
        for c in range(len(tableau[0])):
            tableau[leaving_row][c] /= pivot_val
        for r in range(m + 1):
            if r != leaving_row:
                factor = tableau[r][entering_col]
                if factor != 0:
                    for c in range(len(tableau[0])):
                        tableau[r][c] -= factor * tableau[leaving_row][c]


class WarpSeverer:
    """Graph simplifier detecting non-Euclidean warp cycles and severing minimal chords."""

    def simplify(
        self,
        graph: LayoutGraph,
        options: LayoutOptions | None = None,
    ) -> SimplifiedGraph:
        """Simplify a LayoutGraph by detecting and cutting non-Euclidean warp cycles."""
        if not graph.rooms or not graph.edges:
            return SimplifiedGraph(graph=graph)

        active_edges: list[LayoutEdge] = [e for e in graph.edges if not e.cut]
        severed_chords: list[LayoutEdge] = []

        # Check for non-Euclidean self-loops
        for edge in list(active_edges):
            if edge.src == edge.dst:
                try:
                    disp = direction_to_displacement(edge.direction)
                    satisfiable = is_euclidean_satisfiable([disp])
                except ValueError:
                    satisfiable = False
                if not satisfiable:
                    severed_edge = replace(edge, cut=True)
                    severed_chords.append(severed_edge)
                    active_edges.remove(edge)

        while True:
            # Construct undirected view of remaining active edges
            g = nx.Graph()
            for vnum in graph.rooms:
                g.add_node(vnum)

            undirected_edge_map: dict[tuple[int, int], list[LayoutEdge]] = (
                collections.defaultdict(list)
            )
            for edge in active_edges:
                if edge.src != edge.dst:
                    g.add_edge(edge.src, edge.dst)
                    pair = (min(edge.src, edge.dst), max(edge.src, edge.dst))
                    undirected_edge_map[pair].append(edge)

            cycles = nx.cycle_basis(g)
            if not cycles:
                break

            # Deterministic ordering of cycles by length and vertex IDs
            cycles.sort(key=lambda c: (len(c), sorted(c)))

            found_warp = False
            for cycle in cycles:
                p = len(cycle)
                cycle_vectors: list[tuple[int, int, int]] = []
                cycle_steps_valid = True

                for i in range(p):
                    u = cycle[i]
                    v = cycle[(i + 1) % p]
                    pair = (min(u, v), max(u, v))
                    edges_between = undirected_edge_map[pair]

                    fwd_edges = [e for e in edges_between if e.src == u and e.dst == v]
                    edge = fwd_edges[0] if fwd_edges else edges_between[0]
                    reverse = not bool(fwd_edges)

                    try:
                        vec = direction_to_displacement(edge.direction)
                        if reverse:
                            vec = (-vec[0], -vec[1], -vec[2])
                    except ValueError:
                        cycle_steps_valid = False
                        break
                    cycle_vectors.append(vec)

                if not cycle_steps_valid:
                    continue

                if not is_euclidean_satisfiable(cycle_vectors):
                    found_warp = True

                    # Select minimal chord in the cycle to sever
                    candidate_pairs = [
                        (
                            min(cycle[i], cycle[(i + 1) % p]),
                            max(cycle[i], cycle[(i + 1) % p]),
                        )
                        for i in range(p)
                    ]

                    def chord_score(
                        pair: tuple[int, int],
                    ) -> tuple[int, int, int, tuple[int, int]]:
                        u_node, v_node = pair
                        edges = undirected_edge_map.get(pair, [])
                        min_weight = min((e.weight for e in edges), default=1)
                        deg_sum = g.degree(u_node) + g.degree(v_node)
                        max_deg = max(g.degree(u_node), g.degree(v_node))
                        return (min_weight, -deg_sum, -max_deg, pair)

                    best_pair = min(candidate_pairs, key=chord_score)
                    edges_to_sever = list(undirected_edge_map[best_pair])

                    for edge in edges_to_sever:
                        severed_edge = replace(edge, cut=True)
                        severed_chords.append(severed_edge)
                        active_edges.remove(edge)

                    # Recompute cycle basis with the severed chord removed
                    break

            if not found_warp:
                break

        simplified_graph = LayoutGraph(
            rooms=dict(graph.rooms),
            edges=tuple(active_edges),
        )
        return SimplifiedGraph(
            graph=simplified_graph,
            severed_chords=tuple(severed_chords),
        )


class ChainedSimplifier:
    """Sequentially executes multiple graph simplifiers, accumulating corridors and severed chords."""

    def __init__(self, simplifiers: Sequence[GraphSimplifier]) -> None:
        self.simplifiers = tuple(simplifiers)

    def simplify(
        self,
        graph: LayoutGraph,
        options: LayoutOptions | None = None,
    ) -> SimplifiedGraph:
        """Run all configured simplifiers sequentially on the layout graph."""
        current_graph = graph
        all_corridors: list[CorridorSegment] = []
        all_severed: list[LayoutEdge] = []

        opts = options if options is not None else LayoutOptions()
        for simplifier in self.simplifiers:
            simplified = simplifier.simplify(current_graph, opts)
            current_graph = simplified.graph
            all_corridors.extend(simplified.corridors)
            all_severed.extend(simplified.severed_chords)

        return SimplifiedGraph(
            graph=current_graph,
            corridors=tuple(all_corridors),
            severed_chords=tuple(all_severed),
        )
