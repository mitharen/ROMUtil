"""Degree-2 collinear hallway compaction and coordinate expansion."""

from __future__ import annotations

import collections
from typing import Mapping

import networkx as nx

from romutil.layout.ir import (
    CorridorSegment,
    LayoutEdge,
    LayoutGraph,
    SimplifiedGraph,
)
from romutil.models import Direction, compute_room_footprints
from romutil.types import Coord3D, LayoutOptions


def _distribute_axis_positions(
    step_lengths: tuple[int, ...],
    delta: int,
) -> list[int]:
    """Return strictly monotonic cumulative offsets along an axis summing to delta."""
    n_steps = len(step_lengths)
    total_min = sum(step_lengths)
    if total_min <= 0:
        return [round(i * delta / n_steps) for i in range(1, n_steps)]
    sign = 1 if delta >= 0 else -1
    abs_delta = abs(delta)
    if abs_delta >= total_min:
        slack = abs_delta - total_min
        base_extra, rem = divmod(slack, n_steps)
        cum = 0
        positions: list[int] = []
        for idx in range(n_steps - 1):
            step = step_lengths[idx] + base_extra + (1 if idx < rem else 0)
            cum += step
            positions.append(sign * cum)
        return positions
    cum = 0
    positions = []
    for idx in range(n_steps - 1):
        cum += step_lengths[idx]
        positions.append(round(cum * delta / total_min))
    return positions


def expand_corridor_coordinates(
    corridor: CorridorSegment,
    start_coord: Coord3D,
    end_coord: Coord3D,
) -> Mapping[int, Coord3D]:
    """Calculate interpolated 3D integer coordinates for intermediate corridor rooms.

    Args:
        corridor: CorridorSegment containing intermediate room vnums.
        start_coord: Solved 3D coordinate of corridor.start_vnum.
        end_coord: Solved 3D coordinate of corridor.end_vnum.

    Returns:
        Mapping from intermediate room vnum to interpolated Coord3D.
    """
    if not corridor.intermediate_vnums:
        return {}

    dx = end_coord.x - start_coord.x
    dy = end_coord.y - start_coord.y
    dz = end_coord.z - start_coord.z

    n_intermediate = len(corridor.intermediate_vnums)
    step_lengths = corridor.step_lengths
    if len(step_lengths) == n_intermediate + 1 and sum(step_lengths) > 0:
        total_min = sum(step_lengths)
        xs = (
            _distribute_axis_positions(step_lengths, dx)
            if abs(dx) >= abs(dy) and abs(dx) >= abs(dz) and abs(dx) >= total_min
            else [
                round(sum(step_lengths[:i]) * dx / total_min)
                for i in range(1, n_intermediate + 1)
            ]
        )
        ys = (
            _distribute_axis_positions(step_lengths, dy)
            if abs(dy) > abs(dx) and abs(dy) >= abs(dz) and abs(dy) >= total_min
            else [
                round(sum(step_lengths[:i]) * dy / total_min)
                for i in range(1, n_intermediate + 1)
            ]
        )
        zs = (
            _distribute_axis_positions(step_lengths, dz)
            if abs(dz) > abs(dx) and abs(dz) > abs(dy) and abs(dz) >= total_min
            else [
                round(sum(step_lengths[:i]) * dz / total_min)
                for i in range(1, n_intermediate + 1)
            ]
        )
        return {
            vnum: Coord3D(
                x=int(start_coord.x + xs[idx]),
                y=int(start_coord.y + ys[idx]),
                z=int(start_coord.z + zs[idx]),
            )
            for idx, vnum in enumerate(corridor.intermediate_vnums)
        }

    total_steps = n_intermediate + 1
    expanded: dict[int, Coord3D] = {}
    for i, vnum in enumerate(corridor.intermediate_vnums, start=1):
        x = start_coord.x + round(i * dx / total_steps)
        y = start_coord.y + round(i * dy / total_steps)
        z = start_coord.z + round(i * dz / total_steps)
        expanded[vnum] = Coord3D(x=int(x), y=int(y), z=int(z))

    return expanded


class HallwayCompactor:
    """Graph simplifier collapsing degree-2 collinear hallways into macro-edges."""

    def simplify(
        self,
        graph: LayoutGraph,
        options: LayoutOptions | None = None,
    ) -> SimplifiedGraph:
        """Simplify a LayoutGraph by compacting maximal collinear degree-2 chains."""
        if not graph.rooms:
            return SimplifiedGraph(graph=graph)

        edge_map: dict[tuple[int, int], list[LayoutEdge]] = collections.defaultdict(list)
        for edge in graph.edges:
            edge_map[(edge.src, edge.dst)].append(edge)

        # Build undirected view of the layout graph
        g = nx.Graph()
        for vnum in graph.rooms:
            g.add_node(vnum)
        for edge in graph.edges:
            g.add_edge(edge.src, edge.dst)

        # Identify collinear degree-2 rooms
        collinear_candidates: set[int] = set()
        for v in graph.rooms:
            if g.degree(v) != 2:
                continue

            neighbors = list(g.neighbors(v))
            if len(neighbors) != 2:
                continue
            u, w = neighbors[0], neighbors[1]
            if u not in graph.rooms or w not in graph.rooms:
                continue

            edges_v_u = edge_map.get((v, u), [])
            edges_u_v = edge_map.get((u, v), [])
            edges_v_w = edge_map.get((v, w), [])
            edges_w_v = edge_map.get((w, v), [])

            if (
                len(edges_v_u) != 1
                or len(edges_u_v) != 1
                or len(edges_v_w) != 1
                or len(edges_w_v) != 1
            ):
                continue

            try:
                d_vu = Direction(edges_v_u[0].direction)
                d_uv = Direction(edges_u_v[0].direction)
                d_vw = Direction(edges_v_w[0].direction)
                d_wv = Direction(edges_w_v[0].direction)
            except ValueError:
                continue

            # Must be reciprocal between u <-> v and v <-> w
            if d_uv != d_vu.invert() or d_wv != d_vw.invert():
                continue

            # Must be collinear and opposite at v:
            # exit to u and exit to w must be opposite
            if d_vu.invert() != d_vw:
                continue

            collinear_candidates.add(v)

        if not collinear_candidates:
            return SimplifiedGraph(graph=graph)

        terrain_scales = options.terrain_scales if options is not None else None
        _coord_unit, _room_scales, room_extents = compute_room_footprints(
            graph.rooms,
            terrain_scales=terrain_scales,
        )

        # Find maximal chains in the induced subgraph of collinear candidates
        sub_g = g.subgraph(collinear_candidates)
        corridors: list[CorridorSegment] = []
        macro_edges: list[LayoutEdge] = []
        all_intermediate: set[int] = set()

        for comp in nx.connected_components(sub_g):
            end_candidates = [n for n in comp if sub_g.degree(n) <= 1]
            if not end_candidates:
                # Cycle with no non-collinear boundary
                continue

            if len(comp) == 1:
                node = next(iter(comp))
                chain_nodes = [node]
                neighbors = list(g.neighbors(node))
                u, w = neighbors[0], neighbors[1]
            else:
                p1, p2 = end_candidates[0], end_candidates[1]
                chain_nodes = list(nx.shortest_path(sub_g, p1, p2))
                u = next(n for n in g.neighbors(p1) if n not in comp)
                w = next(n for n in g.neighbors(p2) if n not in comp)

            first_node = chain_nodes[0]
            edge_u_first = edge_map[(u, first_node)][0]
            d = edge_u_first.direction

            # Establish canonical orientation
            if u < w or (u == w and d <= int(Direction(d).invert())):
                start_vnum = u
                end_vnum = w
                intermediate = chain_nodes
                direction = d
            else:
                start_vnum = w
                end_vnum = u
                intermediate = list(reversed(chain_nodes))
                direction = int(Direction(d).invert())

            # Calculate minimum step lengths and macro-edge weight along the chain
            chain_sequence = [start_vnum, *intermediate, end_vnum]
            is_vertical = Direction(direction) in (Direction.up, Direction.down)
            step_lengths_list: list[int] = []
            for a, b in zip(chain_sequence, chain_sequence[1:]):
                raw_w = edge_map[(a, b)][0].weight
                if is_vertical:
                    step_lengths_list.append(max(1, raw_w))
                else:
                    hi_a = room_extents.get(a, (0, 0))[1]
                    lo_b = room_extents.get(b, (0, 0))[0]
                    step_lengths_list.append(max(raw_w, hi_a + lo_b + 1))

            step_lengths = tuple(step_lengths_list)
            total_weight = sum(step_lengths)
            if intermediate:
                transverse_extent = (
                    max(room_extents.get(v, (0, 0))[0] for v in intermediate),
                    max(room_extents.get(v, (0, 0))[1] for v in intermediate),
                )
            else:
                transverse_extent = (0, 0)

            corridor = CorridorSegment(
                start_vnum=start_vnum,
                end_vnum=end_vnum,
                direction=direction,
                intermediate_vnums=tuple(intermediate),
                step_lengths=step_lengths,
                transverse_extent=transverse_extent,
            )
            corridors.append(corridor)
            macro_edges.append(
                LayoutEdge(
                    src=start_vnum,
                    dst=end_vnum,
                    direction=direction,
                    weight=total_weight,
                    metadata={"corridor": corridor, "is_corridor": True, "is_compacted": True},
                )
            )
            all_intermediate.update(intermediate)

        # Sort corridors deterministically
        corridors.sort(key=lambda c: (c.start_vnum, c.end_vnum))

        # Build simplified graph
        compacted_rooms = {
            vnum: room
            for vnum, room in graph.rooms.items()
            if vnum not in all_intermediate
        }
        retained_edges = [
            edge
            for edge in graph.edges
            if edge.src not in all_intermediate and edge.dst not in all_intermediate
        ]
        compacted_edges = retained_edges + macro_edges

        compacted_graph = LayoutGraph(
            rooms=compacted_rooms,
            edges=tuple(compacted_edges),
        )
        return SimplifiedGraph(
            graph=compacted_graph,
            corridors=tuple(corridors),
        )
