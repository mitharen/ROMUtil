"""Layer-neutral incremental 1D positional constraint DAG.

Tracks pairwise lower bounds ``coord_k(v) - coord_k(u) >= w`` per axis with
longest-path propagation and positive-cycle (contradiction) detection. It has
no dependency on MILP or SolveTree types, so both layers may seed it.
"""

from __future__ import annotations

import collections

AXIS_X: int = 0
AXIS_Y: int = 1
AXIS_Z: int = 2
AXIS_NAMES: tuple[str, str, str] = ("x", "y", "z")

NodeRef = int | str


class PositionalConstraintDAG:
    """Incremental 1D positional constraint DAG tracking relative coordinate bounds across 3 axes.

    For each axis k in {X, Y, Z}, maintains a system of difference constraints
    where a directed edge u -(w)-> v represents the relative lower bound:
        coord_k(v) - coord_k(u) >= w

    Provides microsecond cycle and contradiction detection to determine if a candidate
    separation cut can be satisfied without introducing a positive cycle (integer infeasibility).
    """

    def __init__(self) -> None:
        self._nodes: dict[int, set[NodeRef]] = {AXIS_X: set(), AXIS_Y: set(), AXIS_Z: set()}
        self._dist: dict[int, dict[NodeRef, dict[NodeRef, int]]] = {
            AXIS_X: collections.defaultdict(dict),
            AXIS_Y: collections.defaultdict(dict),
            AXIS_Z: collections.defaultdict(dict),
        }
        self._edges: dict[int, dict[NodeRef, dict[NodeRef, int]]] = {
            AXIS_X: collections.defaultdict(dict),
            AXIS_Y: collections.defaultdict(dict),
            AXIS_Z: collections.defaultdict(dict),
        }

    def _ensure_node(self, axis: int, u: NodeRef) -> None:
        """Register a node on the specified axis if not already present."""
        if u not in self._nodes[axis]:
            self._nodes[axis].add(u)
            self._dist[axis][u][u] = 0

    def get_lower_bound(self, axis: int, u: NodeRef, v: NodeRef) -> int | None:
        """Retrieve the maximum lower bound on coord_axis(v) - coord_axis(u).

        Returns 0 if u == v, the path weight if a directed path exists, or None if no path exists.
        """
        if u == v:
            return 0
        if u not in self._nodes[axis] or v not in self._nodes[axis]:
            return None
        return self._dist[axis][u].get(v)

    def can_add_edge(self, axis: int, u: NodeRef, v: NodeRef, weight: int) -> bool:
        """Determine if adding the edge u -(weight)-> v would introduce a positive cycle.

        Enforcing coord_axis(v) - coord_axis(u) >= weight creates a contradiction if the existing
        graph already establishes that coord_axis(u) - coord_axis(v) >= w_vu with w_vu + weight > 0.
        """
        if u == v:
            return weight <= 0

        if u not in self._nodes[axis] or v not in self._nodes[axis]:
            return True

        w_vu = self.get_lower_bound(axis, v, u)
        if w_vu is not None and w_vu + weight > 0:
            return False

        return True

    def add_edge(self, axis: int, u: NodeRef, v: NodeRef, weight: int) -> bool:
        """Add a directed constraint edge u -(weight)-> v on the specified axis.

        Returns True if the edge was successfully integrated without contradiction,
        or False if rejected due to positive cycle formation.
        """
        if not self.can_add_edge(axis, u, v, weight):
            return False

        self._ensure_node(axis, u)
        self._ensure_node(axis, v)

        if u == v:
            return True

        self._edges[axis][u][v] = max(self._edges[axis][u].get(v, weight), weight)

        current_d = self._dist[axis][u].get(v)
        if current_d is not None and current_d >= weight:
            return True

        self._dist[axis][u][v] = weight

        nodes_to_u = [
            (i, d_iu)
            for i, d_map in self._dist[axis].items()
            if (d_iu := d_map.get(u)) is not None
        ]
        nodes_from_v = [
            (j, d_vj)
            for j, d_vj in self._dist[axis][v].items()
        ]

        for i, d_iu in nodes_to_u:
            dist_i = self._dist[axis][i]
            for j, d_vj in nodes_from_v:
                candidate = d_iu + weight + d_vj
                old_val = dist_i.get(j)
                if old_val is None or candidate > old_val:
                    dist_i[j] = candidate

        return True

    def add_equality(self, axis: int, u: NodeRef, v: NodeRef, offset: int) -> bool:
        """Add an exact coordinate displacement equality coord_axis(v) - coord_axis(u) == offset."""
        if not self.can_add_edge(axis, u, v, offset):
            return False
        if not self.can_add_edge(axis, v, u, -offset):
            return False
        added_uv = self.add_edge(axis, u, v, offset)
        added_vu = self.add_edge(axis, v, u, -offset)
        return added_uv and added_vu
