"""Pure computational geometry primitives for 3D layout clearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from romutil.types import Coord3D


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    """Axis-aligned 3D bounding box with integer coordinates."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_z: int
    max_z: int

    def __post_init__(self) -> None:
        if self.min_x > self.max_x or self.min_y > self.max_y or self.min_z > self.max_z:
            raise ValueError(
                f"Invalid bounding box boundaries: "
                f"x=({self.min_x}..{self.max_x}), "
                f"y=({self.min_y}..{self.max_y}), "
                f"z=({self.min_z}..{self.max_z})"
            )

    def contains_point(self, point: Coord3D) -> bool:
        """Returns True if point is inside or on the boundary of the bounding box."""
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
            and self.min_z <= point.z <= self.max_z
        )

    def intersects(self, other: BoundingBox3D) -> bool:
        """Returns True if this bounding box intersects other (closed box check)."""
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
            or self.max_z < other.min_z
            or self.min_z > other.max_z
        )

    def volume(self) -> int:
        """Returns the discrete volume (room cell count) of the bounding box."""
        return (
            (self.max_x - self.min_x + 1)
            * (self.max_y - self.min_y + 1)
            * (self.max_z - self.min_z + 1)
        )

    def dimensions(self) -> tuple[int, int, int]:
        """Returns (width, height, depth) span in discrete cells."""
        return (
            self.max_x - self.min_x + 1,
            self.max_y - self.min_y + 1,
            self.max_z - self.min_z + 1,
        )

    def expand(self, amount: int = 1) -> BoundingBox3D:
        """Inflates all boundaries outwards by amount."""
        return BoundingBox3D(
            min_x=self.min_x - amount,
            max_x=self.max_x + amount,
            min_y=self.min_y - amount,
            max_y=self.max_y + amount,
            min_z=self.min_z - amount,
            max_z=self.max_z + amount,
        )

    @classmethod
    def from_coordinates(cls, coords: Iterable[Coord3D]) -> BoundingBox3D:
        """Constructs the minimal bounding box enclosing all given coordinates."""
        coord_iter = iter(coords)
        try:
            first = next(coord_iter)
        except StopIteration:
            raise ValueError("Cannot create BoundingBox3D from empty coordinates") from None

        min_x = max_x = first.x
        min_y = max_y = first.y
        min_z = max_z = first.z

        for pt in coord_iter:
            if pt.x < min_x:
                min_x = pt.x
            elif pt.x > max_x:
                max_x = pt.x
            if pt.y < min_y:
                min_y = pt.y
            elif pt.y > max_y:
                max_y = pt.y
            if pt.z < min_z:
                min_z = pt.z
            elif pt.z > max_z:
                max_z = pt.z

        return cls(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
        )


@dataclass(frozen=True, slots=True)
class Segment3D:
    """3D line segment defined by two endpoints."""

    p1: Coord3D
    p2: Coord3D

    def bounding_box(self) -> BoundingBox3D:
        """Returns the minimal axis-aligned bounding box enclosing the segment."""
        return BoundingBox3D(
            min_x=min(self.p1.x, self.p2.x),
            max_x=max(self.p1.x, self.p2.x),
            min_y=min(self.p1.y, self.p2.y),
            max_y=max(self.p1.y, self.p2.y),
            min_z=min(self.p1.z, self.p2.z),
            max_z=max(self.p1.z, self.p2.z),
        )

    def is_point(self) -> bool:
        """Returns True if the segment is degenerate (p1 == p2)."""
        return self.p1 == self.p2

    def length_manhattan(self) -> int:
        """Returns the Manhattan (L1) distance between endpoints."""
        return (
            abs(self.p1.x - self.p2.x)
            + abs(self.p1.y - self.p2.y)
            + abs(self.p1.z - self.p2.z)
        )

    def is_axis_aligned(self) -> bool:
        """Returns True if the segment is parallel to a coordinate axis or is a point."""
        diff_count = (
            (self.p1.x != self.p2.x)
            + (self.p1.y != self.p2.y)
            + (self.p1.z != self.p2.z)
        )
        return diff_count <= 1

    def intermediate_points(self) -> Sequence[Coord3D]:
        """Returns discrete integer points strictly between p1 and p2 along axis.

        Raises:
            ValueError: If the segment is not axis-aligned.
        """
        if not self.is_axis_aligned():
            raise ValueError(f"Segment {self} is not axis-aligned")
        if self.is_point():
            return ()

        if self.p1.x != self.p2.x:
            step = 1 if self.p2.x > self.p1.x else -1
            return tuple(
                Coord3D(x, self.p1.y, self.p1.z)
                for x in range(self.p1.x + step, self.p2.x, step)
            )
        if self.p1.y != self.p2.y:
            step = 1 if self.p2.y > self.p1.y else -1
            return tuple(
                Coord3D(self.p1.x, y, self.p1.z)
                for y in range(self.p1.y + step, self.p2.y, step)
            )
        # Otherwise z axis
        step = 1 if self.p2.z > self.p1.z else -1
        return tuple(
            Coord3D(self.p1.x, self.p1.y, z)
            for z in range(self.p1.z + step, self.p2.z, step)
        )
