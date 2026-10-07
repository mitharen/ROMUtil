"""Unit and integration tests for Stage 5 HierarchicalLayoutAssembler."""

from __future__ import annotations

import pytest

from romutil.layout.assembly import HierarchicalLayoutAssembler
from romutil.layout.engine import HierarchicalLayoutSolver
from romutil.layout.hierarchy import SpqrTreeBuilder
from romutil.layout.ir import (
    CorridorSegment,
    LayoutEdge,
    LayoutGraph,
    NodeSolution,
    SimplifiedGraph,
    SolveTree,
    SolveTreeNode,
    TreeSolution,
)
from romutil.layout.milp import HierarchicalTreeSolver
from romutil.layout.protocols import LayoutAssembler
from romutil.layout.simplification import HallwayCompactor
from romutil.models import Direction, Exit, Room
from romutil.types import (
    Coord3D,
    LayoutOptions,
    SolvedLayout,
    SolverStatus,
    WorldGraph,
)


def _make_dummy_room(vnum: int) -> Room:
    return Room(vnum=vnum, name=f"Room {vnum}", description="", exits=())


class TestHierarchicalLayoutAssemblerProtocol:
    """Verify HierarchicalLayoutAssembler conforms to LayoutAssembler protocol."""

    def test_implements_protocol(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        assert isinstance(assembler, LayoutAssembler)


class TestEmptyTreeAndSolutions:
    """Tests for empty input handling in HierarchicalLayoutAssembler."""

    def test_empty_tree_and_solution(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset())
        tree = SolveTree(root=root, components=(root,), graph=None)
        solution = TreeSolution(node_solutions={})
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={}, edges=()))

        result = assembler.assemble(tree, solution, simplified)
        assert isinstance(result, SolvedLayout)
        assert result.coordinates == {}
        assert result.diagnostics.status == SolverStatus.FEASIBLE
        assert result.diagnostics.duration_seconds == 0.0

    def test_tree_has_rooms_but_solution_empty(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1, 2}))
        tree = SolveTree(root=root, components=(root,))
        solution = TreeSolution(node_solutions={})
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={1: _make_dummy_room(1)}, edges=()))

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {}
        assert result.diagnostics.status == SolverStatus.FEASIBLE

    def test_tree_has_rooms_but_node_solution_empty_coordinates(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(root,))
        solution = TreeSolution(
            node_solutions={"root": NodeSolution(node_id="root", local_coordinates={})}
        )
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={1: _make_dummy_room(1)}, edges=()))

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {}
        assert result.diagnostics.status == SolverStatus.FEASIBLE

    def test_solution_has_rooms_but_tree_has_no_rooms(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset())
        tree = SolveTree(root=root, components=(), graph=None)
        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(0, 0, 0)})
            }
        )
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={}, edges=()))

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {}
        assert result.diagnostics.status == SolverStatus.FEASIBLE

    def test_tree_has_rooms_via_components(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset())
        comp = SolveTreeNode(node_id="comp", node_type="leaf", rooms=frozenset({10}))
        tree = SolveTree(root=root, components=(comp,), graph=None)
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={}, edges=()))
        solution = TreeSolution(
            node_solutions={
                "comp": NodeSolution(node_id="comp", local_coordinates={10: Coord3D(5, 5, 5)})
            }
        )
        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {10: Coord3D(0, 0, 0)}

    def test_tree_has_rooms_via_simplified_graph(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset())
        tree = SolveTree(root=root, components=(), graph=None)
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={1: _make_dummy_room(1)}, edges=()))
        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(2, 2, 2)})
            }
        )
        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {1: Coord3D(0, 0, 0)}

    def test_tree_has_rooms_via_root_only(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(), graph=None)
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={}, edges=()))
        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(2, 2, 2)})
            }
        )
        result = assembler.assemble(tree, solution, simplified, options=None)
        assert result.coordinates == {1: Coord3D(0, 0, 0)}


class TestSingleLeafCluster:
    """Verify coordinate normalization on a single leaf cluster."""

    def test_single_leaf_normalization(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        leaf = SolveTreeNode(node_id="leaf1", node_type="leaf", rooms=frozenset({10, 20}))
        tree = SolveTree(root=leaf, components=(leaf,))
        solution = TreeSolution(
            node_solutions={
                "leaf1": NodeSolution(
                    node_id="leaf1",
                    local_coordinates={
                        10: Coord3D(100, 200, 300),
                        20: Coord3D(101, 200, 300),
                    },
                    bounding_box=(Coord3D(100, 200, 300), Coord3D(101, 200, 300)),
                )
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={10: _make_dummy_room(10), 20: _make_dummy_room(20)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates[10] == Coord3D(0, 0, 0)
        assert result.coordinates[20] == Coord3D(1, 0, 0)
        assert result.diagnostics.status == SolverStatus.FEASIBLE


class TestParentChildCoordinateAccumulation:
    """Verify affine coordinate accumulation O_child = O_parent + child_offset."""

    def test_parent_and_child_cluster(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        child = SolveTreeNode(node_id="child", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({1}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,))

        solution = TreeSolution(
            node_solutions={
                "parent": NodeSolution(
                    node_id="parent",
                    local_coordinates={1: Coord3D(0, 0, 0)},
                    bounding_box=(Coord3D(0, 0, 0), Coord3D(10, 5, 0)),
                    metadata={"child_offsets": {"child": Coord3D(5, 2, 1)}},
                ),
                "child": NodeSolution(
                    node_id="child",
                    local_coordinates={2: Coord3D(1, 1, 0)},
                    bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 1, 0)),
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified)
        # Parent room 1 at local (0, 0, 0) -> global (0, 0, 0)
        # Child origin = (5, 2, 1). Child room 2 at local (1, 1, 0) -> global (6, 3, 1)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(6, 3, 1)

    def test_child_offset_as_tuple(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        child = SolveTreeNode(node_id="child", node_type="leaf", rooms=frozenset({2}))
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({1}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,))

        solution = TreeSolution(
            node_solutions={
                "parent": NodeSolution(
                    node_id="parent",
                    local_coordinates={1: Coord3D(0, 0, 0)},
                    metadata={"child_offsets": {"child": (4, 3, 2)}},
                ),
                "child": NodeSolution(
                    node_id="child",
                    local_coordinates={2: Coord3D(0, 0, 0)},
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(4, 3, 2)


class TestMultiTierNestedTree:
    """Verify deep multi-tier hierarchical tree traversal and affine offsets."""

    def test_four_tier_nested_solve_tree(self) -> None:
        # root -> parent -> child container -> leaf
        leaf = SolveTreeNode(node_id="leaf", node_type="leaf", rooms=frozenset({4}))
        child_cont = SolveTreeNode(
            node_id="child_cont",
            node_type="block",
            rooms=frozenset({3}),
            children=(leaf,),
        )
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({2}),
            children=(child_cont,),
        )
        root = SolveTreeNode(
            node_id="root",
            node_type="root",
            rooms=frozenset({1}),
            children=(parent,),
        )
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(
                    node_id="root",
                    local_coordinates={1: Coord3D(0, 0, 0)},
                    metadata={"child_offsets": {"parent": Coord3D(10, 0, 0)}},
                ),
                "parent": NodeSolution(
                    node_id="parent",
                    local_coordinates={2: Coord3D(0, 5, 0)},
                    metadata={"child_offsets": {"child_cont": Coord3D(0, 10, 0)}},
                ),
                "child_cont": NodeSolution(
                    node_id="child_cont",
                    local_coordinates={3: Coord3D(0, 0, 2)},
                    metadata={"child_offsets": {"leaf": Coord3D(0, 0, 5)}},
                ),
                "leaf": NodeSolution(
                    node_id="leaf",
                    local_coordinates={4: Coord3D(1, 1, 1)},
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={i: _make_dummy_room(i) for i in range(1, 5)},
                edges=(),
            )
        )

        assembler = HierarchicalLayoutAssembler()
        result = assembler.assemble(tree, solution, simplified)

        # 1: root origin (0, 0, 0) + local (0, 0, 0) = (0, 0, 0)
        # 2: parent origin (10, 0, 0) + local (0, 5, 0) = (10, 5, 0)
        # 3: child_cont origin (10, 10, 0) + local (0, 0, 2) = (10, 10, 2)
        # 4: leaf origin (10, 10, 5) + local (1, 1, 1) = (11, 11, 6)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(10, 5, 0)
        assert result.coordinates[3] == Coord3D(10, 10, 2)
        assert result.coordinates[4] == Coord3D(11, 11, 6)


class TestArticulationPointRoomSharing:
    """Verify consistent coordinate for cut-vertex rooms shared across parent and child."""

    def test_shared_articulation_point(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        # Room 2 is shared between parent and child
        child = SolveTreeNode(node_id="child", node_type="leaf", rooms=frozenset({2, 3}))
        parent = SolveTreeNode(
            node_id="parent",
            node_type="block",
            rooms=frozenset({1, 2}),
            children=(child,),
        )
        tree = SolveTree(root=parent, components=(parent,))

        # In parent, room 2 is placed at (3, 0, 0).
        # Child's offset in parent is (3, 0, 0).
        # In child, room 2 is placed at (0, 0, 0) and room 3 is placed at (0, 2, 0).
        solution = TreeSolution(
            node_solutions={
                "parent": NodeSolution(
                    node_id="parent",
                    local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(3, 0, 0)},
                    metadata={"child_offsets": {"child": Coord3D(3, 0, 0)}},
                ),
                "child": NodeSolution(
                    node_id="child",
                    local_coordinates={2: Coord3D(0, 0, 0), 3: Coord3D(0, 2, 0)},
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={i: _make_dummy_room(i) for i in (1, 2, 3)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(3, 0, 0)
        assert result.coordinates[3] == Coord3D(3, 2, 0)


class TestHallwayExpansion:
    """Verify corridor intermediate rooms are correctly decompressed and interpolated."""

    def test_corridor_expansion_single_intermediate(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1, 3}))
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(
                    node_id="root",
                    local_coordinates={1: Coord3D(0, 0, 0), 3: Coord3D(2, 0, 0)},
                )
            }
        )
        corridor = CorridorSegment(start_vnum=1, end_vnum=3, direction=0, intermediate_vnums=(2,))
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2), 3: _make_dummy_room(3)},
                edges=(),
            ),
            corridors=(corridor,),
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(1, 0, 0)
        assert result.coordinates[3] == Coord3D(2, 0, 0)

    def test_corridor_expansion_multiple_intermediates(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({10, 40}))
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(
                    node_id="root",
                    local_coordinates={10: Coord3D(0, 0, 0), 40: Coord3D(0, 3, 0)},
                )
            }
        )
        corridor = CorridorSegment(
            start_vnum=10,
            end_vnum=40,
            direction=Direction.north,
            intermediate_vnums=(20, 30),
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={v: _make_dummy_room(v) for v in (10, 20, 30, 40)},
                edges=(),
            ),
            corridors=(corridor,),
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates[10] == Coord3D(0, 0, 0)
        assert result.coordinates[20] == Coord3D(0, 1, 0)
        assert result.coordinates[30] == Coord3D(0, 2, 0)
        assert result.coordinates[40] == Coord3D(0, 3, 0)

    def test_corridor_skipped_if_endpoints_missing(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(
                    node_id="root",
                    local_coordinates={1: Coord3D(0, 0, 0)},
                )
            }
        )
        # Endpoint 99 is missing from global coords
        corridor = CorridorSegment(start_vnum=1, end_vnum=99, direction=0, intermediate_vnums=(2,))
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2), 99: _make_dummy_room(99)},
                edges=(),
            ),
            corridors=(corridor,),
        )

        result = assembler.assemble(tree, solution, simplified)
        assert 1 in result.coordinates
        assert 2 not in result.coordinates


class TestMultiComponentLayout:
    """Verify disconnected component spacing along X-axis respects component_padding."""

    def test_two_components_default_padding(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        comp1 = SolveTreeNode(node_id="comp1", node_type="leaf", rooms=frozenset({1, 2}))
        comp2 = SolveTreeNode(node_id="comp2", node_type="leaf", rooms=frozenset({3, 4}))
        tree = SolveTree(root=comp1, components=(comp1, comp2))

        # comp1: room 1 at (0, 0, 0), room 2 at (1, 0, 0). Max X = 1
        # comp2: room 3 at (0, 0, 0), room 4 at (1, 0, 0). Min X = 0
        solution = TreeSolution(
            node_solutions={
                "comp1": NodeSolution(
                    node_id="comp1",
                    local_coordinates={1: Coord3D(0, 0, 0), 2: Coord3D(1, 0, 0)},
                    bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 0, 0)),
                ),
                "comp2": NodeSolution(
                    node_id="comp2",
                    local_coordinates={3: Coord3D(0, 0, 0), 4: Coord3D(1, 0, 0)},
                    bounding_box=(Coord3D(0, 0, 0), Coord3D(1, 0, 0)),
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={i: _make_dummy_room(i) for i in range(1, 5)},
                edges=(),
            )
        )

        options = LayoutOptions(component_padding=2)
        result = assembler.assemble(tree, solution, simplified, options)

        # comp1 is at x in [0, 1]
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(1, 0, 0)
        # comp2 offset = max_x (1) + padding (2) = 3
        assert result.coordinates[3] == Coord3D(3, 0, 0)
        assert result.coordinates[4] == Coord3D(4, 0, 0)

    def test_custom_padding_value(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        comp1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({1}))
        comp2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({2}))
        tree = SolveTree(root=comp1, components=(comp1, comp2))

        solution = TreeSolution(
            node_solutions={
                "c1": NodeSolution(node_id="c1", local_coordinates={1: Coord3D(0, 0, 0)}),
                "c2": NodeSolution(node_id="c2", local_coordinates={2: Coord3D(0, 0, 0)}),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2)},
                edges=(),
            )
        )

        options = LayoutOptions(component_padding=5)
        result = assembler.assemble(tree, solution, simplified, options)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(5, 0, 0)

    def test_unvisited_component_with_nested_children_no_bbox(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        comp1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({1}))
        c2_leaf = SolveTreeNode(node_id="c2_leaf", node_type="leaf", rooms=frozenset({3}))
        comp2 = SolveTreeNode(
            node_id="c2",
            node_type="block",
            rooms=frozenset({2}),
            children=(c2_leaf,),
        )
        tree = SolveTree(root=comp1, components=(comp1, comp2))

        solution = TreeSolution(
            node_solutions={
                "c1": NodeSolution(node_id="c1", local_coordinates={1: Coord3D(0, 0, 0)}),
                "c2": NodeSolution(
                    node_id="c2",
                    local_coordinates={2: Coord3D(0, 0, 0)},
                    metadata={"child_offsets": {"c2_leaf": Coord3D(2, 0, 0)}},
                    bounding_box=None,
                ),
                "c2_leaf": NodeSolution(
                    node_id="c2_leaf",
                    local_coordinates={3: Coord3D(0, 0, 0)},
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={i: _make_dummy_room(i) for i in (1, 2, 3)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified, LayoutOptions(component_padding=3))
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        # comp2 room 2 at offset 0 + 3 = 3
        assert result.coordinates[2] == Coord3D(3, 0, 0)
        # child room 3 at offset 3 + 2 = 5
        assert result.coordinates[3] == Coord3D(5, 0, 0)

    def test_dummy_root_with_multiple_components(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        dummy_root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset())
        comp1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({1}))
        comp2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({2}))
        tree = SolveTree(root=dummy_root, components=(comp1, comp2))

        solution = TreeSolution(
            node_solutions={
                "c1": NodeSolution(node_id="c1", local_coordinates={1: Coord3D(0, 0, 0)}),
                "c2": NodeSolution(node_id="c2", local_coordinates={2: Coord3D(0, 0, 0)}),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified, LayoutOptions(component_padding=2))
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(2, 0, 0)


class TestUnvisitedNodesInSolution:
    """Verify nodes in TreeSolution not attached to tree nodes are safely offset."""

    def test_unattached_solution_node(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(0, 0, 0)}),
                "unattached": NodeSolution(
                    node_id="unattached",
                    local_coordinates={2: Coord3D(0, 0, 0)},
                    bounding_box=(Coord3D(0, 0, 0), Coord3D(0, 0, 0)),
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified, LayoutOptions(component_padding=4))
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(4, 0, 0)

    def test_unattached_solution_node_without_bounding_box(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(0, 0, 0)}),
                "unattached_nobox": NodeSolution(
                    node_id="unattached_nobox",
                    local_coordinates={2: Coord3D(1, 0, 0)},
                    bounding_box=None,
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified, LayoutOptions(component_padding=2))
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(2, 0, 0)

    def test_unattached_solution_node_empty_coords_skipped(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="leaf", rooms=frozenset({1}))
        tree = SolveTree(root=root, components=(root,))

        solution = TreeSolution(
            node_solutions={
                "root": NodeSolution(node_id="root", local_coordinates={1: Coord3D(0, 0, 0)}),
                "unattached_empty": NodeSolution(
                    node_id="unattached_empty",
                    local_coordinates={},
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {1: Coord3D(0, 0, 0)}

    def test_unattached_solution_node_when_global_coords_empty(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        root = SolveTreeNode(node_id="root", node_type="root", rooms=frozenset())
        tree = SolveTree(root=root, components=(), graph=None)
        # simplified graph has rooms so _tree_has_rooms is True
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms={1: _make_dummy_room(1)}, edges=()))
        solution = TreeSolution(
            node_solutions={
                "unattached": NodeSolution(
                    node_id="unattached",
                    local_coordinates={1: Coord3D(5, 5, 5)},
                    bounding_box=None,
                )
            }
        )
        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates == {1: Coord3D(0, 0, 0)}

    def test_collect_subtree_extra_rooms_not_in_node_rooms(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        comp1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({1}))
        # comp2 node.rooms only has 2, but local_coordinates has 2 and 3
        comp2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({2}))
        tree = SolveTree(root=comp1, components=(comp1, comp2))
        solution = TreeSolution(
            node_solutions={
                "c1": NodeSolution(node_id="c1", local_coordinates={1: Coord3D(0, 0, 0)}),
                "c2": NodeSolution(
                    node_id="c2",
                    local_coordinates={2: Coord3D(0, 0, 0), 3: Coord3D(1, 0, 0)},
                    bounding_box=None,
                ),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={1: _make_dummy_room(1), 2: _make_dummy_room(2), 3: _make_dummy_room(3)},
                edges=(),
            )
        )
        result = assembler.assemble(tree, solution, simplified, LayoutOptions(component_padding=2))
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(2, 0, 0)
        assert result.coordinates[3] == Coord3D(3, 0, 0)


class TestStatusAggregation:
    """Verify status aggregation across node solutions."""

    def _make_fixture(self, statuses: dict[str, str | SolverStatus | None]) -> tuple[SolveTree, TreeSolution, SimplifiedGraph]:
        nodes: list[SolveTreeNode] = []
        node_sols: dict[str, NodeSolution] = {}
        rooms: dict[int, Room] = {}

        for idx, (node_id, st) in enumerate(statuses.items(), start=1):
            nodes.append(SolveTreeNode(node_id=node_id, node_type="leaf", rooms=frozenset({idx})))
            metadata = {"status": st} if st is not None else {}
            node_sols[node_id] = NodeSolution(
                node_id=node_id,
                local_coordinates={idx: Coord3D(0, 0, 0)},
                metadata=metadata,
            )
            rooms[idx] = _make_dummy_room(idx)

        root = nodes[0]
        tree = SolveTree(root=root, components=tuple(nodes))
        solution = TreeSolution(node_solutions=node_sols)
        simplified = SimplifiedGraph(graph=LayoutGraph(rooms=rooms, edges=()))
        return tree, solution, simplified

    def test_all_optimal_gives_optimal(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        tree, solution, simplified = self._make_fixture({
            "n1": SolverStatus.OPTIMAL,
            "n2": SolverStatus.OPTIMAL,
        })
        result = assembler.assemble(tree, solution, simplified)
        assert result.diagnostics.status == SolverStatus.OPTIMAL

    def test_optimal_as_string_gives_optimal(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        tree, solution, simplified = self._make_fixture({"n1": "optimal"})
        result = assembler.assemble(tree, solution, simplified)
        assert result.diagnostics.status == SolverStatus.OPTIMAL

    def test_timeout_takes_precedence(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        tree, solution, simplified = self._make_fixture({
            "n1": SolverStatus.OPTIMAL,
            "n2": SolverStatus.INFEASIBLE,
            "n3": SolverStatus.TIMEOUT,
        })
        result = assembler.assemble(tree, solution, simplified)
        assert result.diagnostics.status == SolverStatus.TIMEOUT

    def test_infeasible_without_timeout_gives_infeasible(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        tree, solution, simplified = self._make_fixture({
            "n1": SolverStatus.OPTIMAL,
            "n2": SolverStatus.INFEASIBLE,
        })
        result = assembler.assemble(tree, solution, simplified)
        assert result.diagnostics.status == SolverStatus.INFEASIBLE

    def test_mixed_optimal_and_no_status_gives_feasible(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        tree, solution, simplified = self._make_fixture({
            "n1": SolverStatus.OPTIMAL,
            "n2": None,
        })
        result = assembler.assemble(tree, solution, simplified)
        assert result.diagnostics.status == SolverStatus.FEASIBLE

    def test_invalid_status_type_and_string_gives_feasible(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        tree, solution, simplified = self._make_fixture({
            "n1": "not_a_real_status",
            "n2": 12345,  # type: ignore[dict-item]
        })
        result = assembler.assemble(tree, solution, simplified)
        assert result.diagnostics.status == SolverStatus.FEASIBLE

    def test_empty_solution_status(self) -> None:
        from romutil.layout.assembly.hierarchical import _aggregate_status

        assert _aggregate_status(TreeSolution(node_solutions={})) == SolverStatus.FEASIBLE


class TestOffsetParsingEdgeCases:
    """Verify offset parsing helpers handle unexpected types gracefully."""

    def test_invalid_child_offset_types(self) -> None:
        assembler = HierarchicalLayoutAssembler()
        child1 = SolveTreeNode(node_id="c1", node_type="leaf", rooms=frozenset({2}))
        child2 = SolveTreeNode(node_id="c2", node_type="leaf", rooms=frozenset({3}))
        parent = SolveTreeNode(
            node_id="p",
            node_type="block",
            rooms=frozenset({1}),
            children=(child1, child2),
        )
        tree = SolveTree(root=parent, components=(parent,))

        solution = TreeSolution(
            node_solutions={
                "p": NodeSolution(
                    node_id="p",
                    local_coordinates={1: Coord3D(0, 0, 0)},
                    metadata={
                        "child_offsets": {
                            "c1": "not_a_coord",  # Invalid type -> Coord3D(0, 0, 0)
                            "c2": (1, 2),          # Invalid tuple length -> Coord3D(0, 0, 0)
                        }
                    },
                ),
                "c1": NodeSolution(node_id="c1", local_coordinates={2: Coord3D(1, 0, 0)}),
                "c2": NodeSolution(node_id="c2", local_coordinates={3: Coord3D(0, 1, 0)}),
            }
        )
        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={i: _make_dummy_room(i) for i in (1, 2, 3)},
                edges=(),
            )
        )

        result = assembler.assemble(tree, solution, simplified)
        assert result.coordinates[1] == Coord3D(0, 0, 0)
        assert result.coordinates[2] == Coord3D(1, 0, 0)
        assert result.coordinates[3] == Coord3D(0, 1, 0)


class TestHierarchicalLayoutPipelineIntegration:
    """End-to-end integration test connecting stages 2, 3, 4, and 5."""

    def test_full_pipeline_corridor_cycle_articulation(self) -> None:
        """Solve an area with a 4-room cycle, articulation point, and 3-room corridor."""
        # Cycle: 1 -E-> 2 -N-> 3 -W-> 4 -S-> 1
        # Articulation point at 2 connecting to corridor: 2 -E-> 5 -E-> 6 -E-> 7
        r1 = Room(
            vnum=1,
            name="R1",
            description="",
            exits=(
                Exit(direction=Direction.east, dst_vnum=2),
                Exit(direction=Direction.north, dst_vnum=4),
            ),
        )
        r2 = Room(
            vnum=2,
            name="R2",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=1),
                Exit(direction=Direction.north, dst_vnum=3),
                Exit(direction=Direction.east, dst_vnum=5),
            ),
        )
        r3 = Room(
            vnum=3,
            name="R3",
            description="",
            exits=(
                Exit(direction=Direction.south, dst_vnum=2),
                Exit(direction=Direction.west, dst_vnum=4),
            ),
        )
        r4 = Room(
            vnum=4,
            name="R4",
            description="",
            exits=(
                Exit(direction=Direction.east, dst_vnum=3),
                Exit(direction=Direction.south, dst_vnum=1),
            ),
        )
        r5 = Room(
            vnum=5,
            name="R5",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=2),
                Exit(direction=Direction.east, dst_vnum=6),
            ),
        )
        r6 = Room(
            vnum=6,
            name="R6 Corridor",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=5),
                Exit(direction=Direction.east, dst_vnum=7),
            ),
        )
        r7 = Room(
            vnum=7,
            name="R7 End",
            description="",
            exits=(
                Exit(direction=Direction.west, dst_vnum=6),
            ),
        )

        rooms = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6, 7: r7}
        all_exits = tuple(e for r in rooms.values() for e in r.exits)
        world_graph = WorldGraph(rooms=rooms, exits=all_exits)

        solver = HierarchicalLayoutSolver(
            simplifier=HallwayCompactor(),
            tree_builder=SpqrTreeBuilder(),
            tree_solver=HierarchicalTreeSolver(),
            assembler=HierarchicalLayoutAssembler(),
        )

        solved = solver.solve(world_graph)

        assert isinstance(solved, SolvedLayout)
        assert set(solved.coordinates.keys()) == {1, 2, 3, 4, 5, 6, 7}
        assert solved.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert solved.diagnostics.duration_seconds > 0.0

        # Verify hallway room 6 is collinear and between 5 and 7
        c5 = solved.coordinates[5]
        c6 = solved.coordinates[6]
        c7 = solved.coordinates[7]

        # Room 6 must be strictly between 5 and 7 or on the straight line segment
        assert min(c5.x, c7.x) <= c6.x <= max(c5.x, c7.x)
        assert min(c5.y, c7.y) <= c6.y <= max(c5.y, c7.y)
        assert min(c5.z, c7.z) <= c6.z <= max(c5.z, c7.z)

        # Min coordinates must be normalized to origin (0, 0, 0)
        assert min(c.x for c in solved.coordinates.values()) == 0
        assert min(c.y for c in solved.coordinates.values()) == 0
        assert min(c.z for c in solved.coordinates.values()) == 0


class TestHierarchicalAssemblerDataOwnership:
    """Tests verifying canonical top-down ancestor room ownership during coordinate assembly."""

    def test_parent_coordinates_authoritative_over_child_coordinates(self) -> None:
        """Ancestor container room coordinates take precedence over child local coordinates."""
        assembler = HierarchicalLayoutAssembler()

        # Room 10 is a boundary room shared between parent and child.
        # Parent places room 10 at (10, 20, 0) and room 11 at (11, 20, 0).
        # Child contains room 10 with drifted local coordinate (0, 0, 0) and child room 12 at (1, 0, 0).
        child_node = SolveTreeNode(
            node_id="child_1",
            node_type="series_loop",
            rooms=frozenset({10, 12}),
        )
        parent_node = SolveTreeNode(
            node_id="parent_root",
            node_type="rigid_cluster",
            rooms=frozenset({10, 11}),
            children=(child_node,),
        )
        tree = SolveTree(root=parent_node, components=(parent_node,))

        solution = TreeSolution(
            node_solutions={
                "parent_root": NodeSolution(
                    node_id="parent_root",
                    local_coordinates={
                        10: Coord3D(10, 20, 0),
                        11: Coord3D(11, 20, 0),
                    },
                    metadata={"child_offsets": {"child_1": Coord3D(5, 5, 0)}},
                ),
                "child_1": NodeSolution(
                    node_id="child_1",
                    local_coordinates={
                        10: Coord3D(0, 0, 0),  # Attempting to overwrite 10 to child_origin + (0, 0, 0)
                        12: Coord3D(1, 0, 0),
                    },
                ),
            }
        )

        simplified = SimplifiedGraph(
            graph=LayoutGraph(
                rooms={
                    10: _make_dummy_room(10),
                    11: _make_dummy_room(11),
                    12: _make_dummy_room(12),
                },
                edges=(),
            )
        )

        layout = assembler.assemble(tree, solution, simplified)

        # Before normalization:
        # parent placed 10 at (10, 20, 0), 11 at (11, 20, 0).
        # child origin is (5, 5, 0). child placed 12 at (5 + 1, 5 + 0, 0) = (6, 5, 0).
        # If parent is authoritative for 10:
        # 10 is at (10, 20, 0).
        # Normalization subtracts min_x=6, min_y=5, min_z=0:
        # 10 normalized: (10 - 6, 20 - 5, 0) = (4, 15, 0).
        # 11 normalized: (11 - 6, 20 - 5, 0) = (5, 15, 0).
        # 12 normalized: (6 - 6, 5 - 5, 0) = (0, 0, 0).
        # Vector from 10 to 11 is (1, 0, 0), preserving parent geometry.
        assert layout.coordinates[10] == Coord3D(4, 15, 0)
        assert layout.coordinates[11] == Coord3D(5, 15, 0)
        assert layout.coordinates[12] == Coord3D(0, 0, 0)
