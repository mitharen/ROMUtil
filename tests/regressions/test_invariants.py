"""Unit tests verifying layout invariant assertion helper behavior."""

from __future__ import annotations

import pytest

from romutil.layout.geometry import BoundingBox3D
from romutil.models import Direction, Exit, Room
from romutil.types import Coord3D, SolvedLayout, WorldGraph
from tests.regressions.invariants import (
    assert_axis_aligned_exits,
    assert_corridor_non_penetration,
    assert_zero_coordinate_collisions,
)

pytestmark = [pytest.mark.regression]


def _make_world_graph(
    rooms_data: list[tuple[int, list[tuple[Direction, int]]]],
) -> WorldGraph:
    rooms_map: dict[int, Room] = {}
    all_exits: list[Exit] = []
    for vnum, exits_list in rooms_data:
        room_exits: list[Exit] = []
        for direction, dst_vnum in exits_list:
            ex = Exit(direction=direction, dst_vnum=dst_vnum)
            room_exits.append(ex)
            all_exits.append(ex)
        rooms_map[vnum] = Room(
            vnum=vnum,
            name=f"Room {vnum}",
            description=f"Description for room {vnum}",
            exits=tuple(room_exits),
        )
    return WorldGraph(rooms=rooms_map, exits=tuple(all_exits))


class TestCoordinateCollisionAssertions:
    """Tests verifying collision detection across integer coordinates."""

    def test_assert_zero_coordinate_collisions_passes_on_distinct(self) -> None:
        """Unique integer coordinates pass collision assertion without error."""
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(1, 0, 0),
                3: Coord3D(0, 1, 0),
            }
        )
        assert_zero_coordinate_collisions(layout)

    def test_assert_zero_coordinate_collisions_fails_on_duplicate(self) -> None:
        """Duplicate integer coordinates trigger assertion error with room details."""
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(0, 0, 0),
                3: Coord3D(1, 1, 0),
            }
        )
        with pytest.raises(AssertionError) as exc_info:
            assert_zero_coordinate_collisions(layout)
        assert "Detected 1 coordinate collision(s)" in str(exc_info.value)
        assert "rooms [1, 2]" in str(exc_info.value)


class TestAxisAlignedExitAssertions:
    """Tests verifying exit perpendicular shear and cardinal alignment assertions."""

    def test_assert_axis_aligned_exits_passes_on_cardinal_axes(self) -> None:
        """Cardinal exits with zero perpendicular shear pass alignment assertion."""
        graph = _make_world_graph(
            [
                (1, [(Direction.east, 2)]),
                (2, [(Direction.north, 3)]),
                (3, [(Direction.up, 4)]),
                (4, []),
            ]
        )
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(5, 0, 0),
                3: Coord3D(5, 5, 0),
                4: Coord3D(5, 5, 2),
            }
        )
        assert_axis_aligned_exits(layout, graph, [(1, 2), (2, 3), (3, 4)])

    def test_assert_axis_aligned_exits_resolves_inverted_direction(self) -> None:
        """Reverse exit defined on target room correctly resolves alignment axis."""
        graph = _make_world_graph(
            [
                (1, []),
                (2, [(Direction.west, 1)]),
            ]
        )
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(4, 0, 0),
            }
        )
        assert_axis_aligned_exits(layout, graph, [(1, 2)])

    def test_assert_axis_aligned_exits_fails_on_perpendicular_shear(self) -> None:
        """Non-zero perpendicular displacement triggers an assertion error."""
        graph = _make_world_graph(
            [
                (1, [(Direction.east, 2)]),
                (2, []),
            ]
        )
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(5, 1, 0),  # dy=1 shear on East exit
            }
        )
        with pytest.raises(AssertionError) as exc_info:
            assert_axis_aligned_exits(layout, graph, [(1, 2)])
        assert "dy=1" in str(exc_info.value)

    def test_assert_axis_aligned_exits_fails_on_missing_room(self) -> None:
        """Missing layout coordinates raise KeyError."""
        graph = _make_world_graph([(1, [(Direction.east, 2)]), (2, [])])
        layout = SolvedLayout(coordinates={1: Coord3D(0, 0, 0)})
        with pytest.raises(KeyError):
            assert_axis_aligned_exits(layout, graph, [(1, 2)])

    def test_assert_axis_aligned_exits_implicit_cardinal_fallback(self) -> None:
        """Exits without explicit graph direction pass if single-axis aligned and fail if multi-axis."""
        graph = _make_world_graph([(1, []), (2, [])])
        clean_layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(3, 0, 0),
            }
        )
        assert_axis_aligned_exits(clean_layout, graph, [(1, 2)])

        skew_layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(3, 2, 0),
            }
        )
        with pytest.raises(AssertionError):
            assert_axis_aligned_exits(skew_layout, graph, [(1, 2)])


class TestCorridorNonPenetrationAssertions:
    """Tests verifying corridor interior clearance against rooms and keep-out boxes."""

    def test_corridor_non_penetration_passes_on_clear_layout(self) -> None:
        """Corridors passing through unobstructed space pass non-penetration assertions."""
        graph = _make_world_graph(
            [
                (1, [(Direction.east, 2)]),
                (2, []),
                (3, []),
            ]
        )
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(5, 0, 0),
                3: Coord3D(2, 2, 0),  # Off to the side
            }
        )
        assert_corridor_non_penetration(layout, graph)

    def test_corridor_non_penetration_fails_on_room_intersection(self) -> None:
        """Room occupying the interior of a corridor segment triggers an assertion error."""
        graph = _make_world_graph(
            [
                (1, [(Direction.east, 2)]),
                (2, []),
                (3, []),
            ]
        )
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(5, 0, 0),
                3: Coord3D(2, 0, 0),  # Directly along corridor interior
            }
        )
        with pytest.raises(AssertionError) as exc_info:
            assert_corridor_non_penetration(layout, graph)
        assert "penetrates interior room 3" in str(exc_info.value)

    def test_corridor_non_penetration_with_keepout_boxes(self) -> None:
        """Keep-out boxes intersecting corridor interior trigger assertion error."""
        graph = _make_world_graph(
            [
                (1, [(Direction.north, 2)]),
                (2, []),
            ]
        )
        layout = SolvedLayout(
            coordinates={
                1: Coord3D(0, 0, 0),
                2: Coord3D(0, 6, 0),
            }
        )
        clear_box = BoundingBox3D(2, 4, 1, 5, 0, 0)
        assert_corridor_non_penetration(layout, graph, keepout_boxes=[clear_box])

        intersecting_box = BoundingBox3D(-1, 1, 2, 4, 0, 0)
        with pytest.raises(AssertionError) as exc_info:
            assert_corridor_non_penetration(
                layout, graph, keepout_boxes={"child_0": [intersecting_box]}
            )
        assert "intersects keep-out box" in str(exc_info.value)
