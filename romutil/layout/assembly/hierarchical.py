"""Top-down hierarchical coordinate assembly and corridor decompression."""

from __future__ import annotations

from typing import Any

from romutil.layout.ir import (
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.protocols import LayoutAssembler
from romutil.layout.simplification.hallways import expand_corridor_coordinates
from romutil.types import (
    Coord3D,
    LayoutDiagnostics,
    LayoutOptions,
    SolvedLayout,
    SolverStatus,
)


def _has_rooms_in_node(node: SolveTreeNode) -> bool:
    """Return True if node or any of its descendants contain rooms."""
    if node.rooms:
        return True
    return any(_has_rooms_in_node(child) for child in node.children)


def _tree_has_rooms(tree: SolveTree, simplified: SimplifiedGraph) -> bool:
    """Return True if tree or simplified graph contains any rooms."""
    if tree.graph and tree.graph.rooms:
        return True
    if simplified.graph and simplified.graph.rooms:
        return True
    if tree.root and _has_rooms_in_node(tree.root):
        return True
    return any(_has_rooms_in_node(comp) for comp in tree.components)


def _solution_has_rooms(solution: TreeSolution) -> bool:
    """Return True if any node solution contains local room coordinates."""
    return any(bool(s.local_coordinates) for s in solution.node_solutions.values())


def _aggregate_status(solution: TreeSolution) -> SolverStatus:
    """Aggregate solver statuses across all node solutions."""
    if not solution.node_solutions:
        return SolverStatus.FEASIBLE

    statuses: list[SolverStatus] = []
    for node_sol in solution.node_solutions.values():
        raw_status = node_sol.metadata.get("status")
        if raw_status is None:
            statuses.append(SolverStatus.FEASIBLE)
        elif isinstance(raw_status, SolverStatus):
            statuses.append(raw_status)
        elif isinstance(raw_status, str):
            try:
                statuses.append(SolverStatus(raw_status.lower()))
            except ValueError:
                statuses.append(SolverStatus.FEASIBLE)
        else:
            statuses.append(SolverStatus.FEASIBLE)

    if any(s == SolverStatus.TIMEOUT for s in statuses):
        return SolverStatus.TIMEOUT
    if any(s == SolverStatus.INFEASIBLE for s in statuses):
        return SolverStatus.INFEASIBLE
    if all(s == SolverStatus.OPTIMAL for s in statuses):
        return SolverStatus.OPTIMAL
    return SolverStatus.FEASIBLE


def _parse_offset(raw_offset: Any) -> Coord3D:
    """Convert raw offset value to Coord3D."""
    if isinstance(raw_offset, Coord3D):
        return raw_offset
    if isinstance(raw_offset, (tuple, list)) and len(raw_offset) == 3:
        return Coord3D(int(raw_offset[0]), int(raw_offset[1]), int(raw_offset[2]))
    return Coord3D(0, 0, 0)


def _collect_subtree_local_coords(
    node: SolveTreeNode,
    solution: TreeSolution,
    origin: Coord3D = Coord3D(0, 0, 0),
) -> dict[int, Coord3D]:
    """Collect local room coordinates across a node's subtree relative to origin."""
    coords: dict[int, Coord3D] = {}
    node_sol = solution.node_solutions.get(node.node_id)
    child_offsets_meta = node_sol.metadata.get("child_offsets", {}) if node_sol else {}

    if node_sol is not None:
        for u in node.rooms:
            if u in node_sol.local_coordinates:
                loc = node_sol.local_coordinates[u]
                coords[u] = Coord3D(origin.x + loc.x, origin.y + loc.y, origin.z + loc.z)
        for u, loc in node_sol.local_coordinates.items():
            if u not in coords:
                coords[u] = Coord3D(origin.x + loc.x, origin.y + loc.y, origin.z + loc.z)

    for child in node.children:
        c_offset = _parse_offset(child_offsets_meta.get(child.node_id, Coord3D(0, 0, 0)))
        child_origin = Coord3D(origin.x + c_offset.x, origin.y + c_offset.y, origin.z + c_offset.z)
        child_coords = _collect_subtree_local_coords(child, solution, child_origin)
        coords.update(child_coords)

    return coords


class HierarchicalLayoutAssembler(LayoutAssembler):
    """Stage 5 assembler accumulating coordinates top-down from SolveTree with corridor decompression."""

    def assemble(
        self,
        tree: SolveTree,
        solution: TreeSolution,
        simplified: SimplifiedGraph,
        options: LayoutOptions | None = None,
    ) -> SolvedLayout:
        """Assemble a global SolvedLayout from tree solutions and topology."""
        opts = options if options is not None else LayoutOptions()

        if not _tree_has_rooms(tree, simplified) or not _solution_has_rooms(solution):
            return SolvedLayout(
                coordinates={},
                diagnostics=LayoutDiagnostics(
                    status=SolverStatus.FEASIBLE,
                    duration_seconds=0.0,
                ),
            )

        visited_nodes: set[str] = set()
        global_coords: dict[int, Coord3D] = {}

        def _traverse_node(node: SolveTreeNode, origin: Coord3D) -> None:
            visited_nodes.add(node.node_id)
            node_sol = solution.node_solutions.get(node.node_id)
            child_offsets_meta = node_sol.metadata.get("child_offsets", {}) if node_sol else {}

            if node_sol is not None:
                for u in node.rooms:
                    if u in node_sol.local_coordinates:
                        loc = node_sol.local_coordinates[u]
                        global_coords[u] = Coord3D(origin.x + loc.x, origin.y + loc.y, origin.z + loc.z)
                for u, loc in node_sol.local_coordinates.items():
                    if u not in global_coords:
                        global_coords[u] = Coord3D(origin.x + loc.x, origin.y + loc.y, origin.z + loc.z)

            for child in node.children:
                c_offset = _parse_offset(child_offsets_meta.get(child.node_id, Coord3D(0, 0, 0)))
                child_origin = Coord3D(
                    origin.x + c_offset.x,
                    origin.y + c_offset.y,
                    origin.z + c_offset.z,
                )
                _traverse_node(child, child_origin)

        # 1. Traverse root node starting at origin (0, 0, 0)
        _traverse_node(tree.root, Coord3D(0, 0, 0))

        # 2. Check tree.components: traverse unvisited components and offset along X-axis
        padding = opts.component_padding
        for comp in tree.components:
            if comp.node_id not in visited_nodes:
                node_sol = solution.node_solutions.get(comp.node_id)
                if node_sol and node_sol.bounding_box is not None:
                    min_x = node_sol.bounding_box[0].x
                else:
                    local_coords = _collect_subtree_local_coords(comp, solution, Coord3D(0, 0, 0))
                    min_x = min((c.x for c in local_coords.values()), default=0)

                if global_coords:
                    prev_max_x = max(c.x for c in global_coords.values())
                    comp_origin_x = prev_max_x + padding - min_x
                else:
                    comp_origin_x = -min_x

                _traverse_node(comp, Coord3D(comp_origin_x, 0, 0))

        # 3. Check any unvisited node in solution.node_solutions
        for node_id, node_sol in solution.node_solutions.items():
            if node_id not in visited_nodes:
                visited_nodes.add(node_id)
                if not node_sol.local_coordinates:
                    continue
                if node_sol.bounding_box is not None:
                    min_x = node_sol.bounding_box[0].x
                else:
                    min_x = min(c.x for c in node_sol.local_coordinates.values())

                if global_coords:
                    prev_max_x = max(c.x for c in global_coords.values())
                    offset_x = prev_max_x + padding - min_x
                else:
                    offset_x = -min_x

                for u, loc in node_sol.local_coordinates.items():
                    global_coords[u] = Coord3D(loc.x + offset_x, loc.y, loc.z)

        # 4. Hallway expansion
        for corridor in simplified.corridors:
            if corridor.start_vnum in global_coords and corridor.end_vnum in global_coords:
                expanded = expand_corridor_coordinates(
                    corridor,
                    global_coords[corridor.start_vnum],
                    global_coords[corridor.end_vnum],
                )
                global_coords.update(expanded)

        # 5. Coordinate Normalization
        normalized_coords: dict[int, Coord3D] = {}
        if global_coords:
            min_x = min(c.x for c in global_coords.values())
            min_y = min(c.y for c in global_coords.values())
            min_z = min(c.z for c in global_coords.values())
            normalized_coords = {
                vnum: Coord3D(c.x - min_x, c.y - min_y, c.z - min_z)
                for vnum, c in global_coords.items()
            }

        # 6. Status aggregation
        status = _aggregate_status(solution)

        return SolvedLayout(
            coordinates=normalized_coords,
            diagnostics=LayoutDiagnostics(
                status=status,
                duration_seconds=0.0,
            ),
        )
