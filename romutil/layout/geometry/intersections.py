"""Pure computational geometry clearance invariants and intersection tests."""

from __future__ import annotations

from fractions import Fraction

from romutil.layout.geometry.primitives import BoundingBox3D, Segment3D
from romutil.types import Coord3D


def point_in_segment_interior(pt: Coord3D, seg: Segment3D) -> bool:
    """Checks if pt lies strictly on the open interior of line segment (p1, p2).

    Returns False if seg is a point, or if pt equals p1 or p2, or if pt
    is not collinear with p1->p2 or lies outside the open interval (p1, p2).
    """
    if seg.is_point() or pt == seg.p1 or pt == seg.p2:
        return False

    dx = seg.p2.x - seg.p1.x
    dy = seg.p2.y - seg.p1.y
    dz = seg.p2.z - seg.p1.z

    vx = pt.x - seg.p1.x
    vy = pt.y - seg.p1.y
    vz = pt.z - seg.p1.z

    # Check zero-delta axes: pt must match seg.p1 along inactive axes
    if dx == 0 and vx != 0:
        return False
    if dy == 0 and vy != 0:
        return False
    if dz == 0 and vz != 0:
        return False

    # Check that ratio t is in (0, 1) along all active axes
    if dx != 0:
        if dx > 0 and not (0 < vx < dx):
            return False
        if dx < 0 and not (dx < vx < 0):
            return False

    if dy != 0:
        if dy > 0 and not (0 < vy < dy):
            return False
        if dy < 0 and not (dy < vy < 0):
            return False

    if dz != 0:
        if dz > 0 and not (0 < vz < dz):
            return False
        if dz < 0 and not (dz < vz < 0):
            return False

    # Check collinearity across active axes via cross products
    if vx * dy != vy * dx:
        return False
    if vx * dz != vz * dx:
        return False
    if vy * dz != vz * dy:
        return False

    return True


def segment_room_penetration(seg: Segment3D, room_coord: Coord3D) -> bool:
    """Returns True if room_coord penetrates the strict interior of seg (excluding endpoints)."""
    if room_coord == seg.p1 or room_coord == seg.p2:
        return False
    return point_in_segment_interior(room_coord, seg)


def segments_intersect_3d(seg1: Segment3D, seg2: Segment3D) -> bool:
    """Accurately tests whether two 3D line segments cross/intersect in R^3.

    - If segments share an endpoint (incident), returns False.
    - If segments are perpendicular and cross, returns True.
    - If segments are skew (cross in 2D projection but have different Z), returns False.
    - If segments are collinear and overlap on an interior interval, returns True.
    - If segments are parallel non-collinear, returns False.
    - If segments are collinear abutting sharing an endpoint, returns False.
    """
    # Quick rejection via bounding boxes
    if not seg1.bounding_box().intersects(seg2.bounding_box()):
        return False

    # Check incident endpoints (sharing an endpoint)
    # Note: if they share an endpoint, they can only be incident / abutting
    # UNLESS they are collinear and overlap on an interval of length > 0.
    shares_endpoint = (
        seg1.p1 == seg2.p1
        or seg1.p1 == seg2.p2
        or seg1.p2 == seg2.p1
        or seg1.p2 == seg2.p2
    )

    # Direction vectors
    u = (seg1.p2.x - seg1.p1.x, seg1.p2.y - seg1.p1.y, seg1.p2.z - seg1.p1.z)
    v = (seg2.p2.x - seg2.p1.x, seg2.p2.y - seg2.p1.y, seg2.p2.z - seg2.p1.z)
    w = (seg2.p1.x - seg1.p1.x, seg2.p1.y - seg1.p1.y, seg2.p1.z - seg1.p1.z)

    # Cross product u x v
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]

    # Check if u and v are parallel (cx == 0 and cy == 0 and cz == 0)
    if cx == 0 and cy == 0 and cz == 0:
        # Cross product w x u to check if collinear
        wxu_x = w[1] * u[2] - w[2] * u[1]
        wxu_y = w[2] * u[0] - w[0] * u[2]
        wxu_z = w[0] * u[1] - w[1] * u[0]

        # If u is 0 vector, seg1 is a point
        if u[0] == 0 and u[1] == 0 and u[2] == 0:
            if v[0] == 0 and v[1] == 0 and v[2] == 0:
                # Both are points
                return False
            # seg1 is point, seg2 is segment
            if shares_endpoint:
                return False
            return point_in_segment_interior(seg1.p1, seg2)

        if v[0] == 0 and v[1] == 0 and v[2] == 0:
            # seg2 is point, seg1 is segment
            if shares_endpoint:
                return False
            return point_in_segment_interior(seg2.p1, seg1)

        # Non-degenerate parallel segments: check if collinear
        if wxu_x != 0 or wxu_y != 0 or wxu_z != 0:
            # Parallel non-collinear
            return False

        # Collinear segments: project onto axis with max span in u
        max_idx = 0
        if abs(u[1]) > abs(u[0]):
            max_idx = 1
        if abs(u[2]) > abs(u[max_idx]):
            max_idx = 2

        c1_a = (seg1.p1.x, seg1.p1.y, seg1.p1.z)[max_idx]
        c1_b = (seg1.p2.x, seg1.p2.y, seg1.p2.z)[max_idx]
        c2_a = (seg2.p1.x, seg2.p1.y, seg2.p1.z)[max_idx]
        c2_b = (seg2.p2.x, seg2.p2.y, seg2.p2.z)[max_idx]

        min1, max1 = min(c1_a, c1_b), max(c1_a, c1_b)
        min2, max2 = min(c2_a, c2_b), max(c2_a, c2_b)

        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)

        # Overlap on an interior interval requires strictly positive overlap span
        return overlap_start < overlap_end

    # u and v are not parallel
    # If they share an endpoint, they are incident and can only meet at that endpoint
    if shares_endpoint:
        return False

    # Check coplanarity via scalar triple product w . (u x v)
    if w[0] * cx + w[1] * cy + w[2] * cz != 0:
        # Skew lines in 3D
        return False

    # Lines are coplanar and not parallel. Solve for s and t using Cramer's rule.
    if cz != 0:
        s = Fraction(w[0] * v[1] - w[1] * v[0], cz)
        t = Fraction(u[1] * w[0] - u[0] * w[1], cz)
    elif cx != 0:
        s = Fraction(w[1] * v[2] - w[2] * v[1], cx)
        t = Fraction(u[2] * w[1] - u[1] * w[2], cx)
    else:  # cy != 0
        s = Fraction(w[2] * v[0] - w[0] * v[2], cy)
        t = Fraction(u[0] * w[2] - u[2] * w[0], cy)

    return 0 <= s <= 1 and 0 <= t <= 1


def segment_box_penetration(seg: Segment3D, box: BoundingBox3D) -> bool:
    """Checks if seg penetrates the strict interior of box.

    Touching the outer boundary or terminating on the boundary does NOT count
    as penetration (allowing valid gateway doorways).
    """
    # Degenerate boxes have empty strict interior
    if box.min_x >= box.max_x or box.min_y >= box.max_y or box.min_z >= box.max_z:
        return False

    # Check degenerate point segment
    if seg.is_point():
        return (
            box.min_x < seg.p1.x < box.max_x
            and box.min_y < seg.p1.y < box.max_y
            and box.min_z < seg.p1.z < box.max_z
        )

    dx = seg.p2.x - seg.p1.x
    dy = seg.p2.y - seg.p1.y
    dz = seg.p2.z - seg.p1.z

    deltas = (dx, dy, dz)
    p1_coords = (seg.p1.x, seg.p1.y, seg.p1.z)
    box_mins = (box.min_x, box.min_y, box.min_z)
    box_maxs = (box.max_x, box.max_y, box.max_z)

    # Check inactive axes (where delta == 0)
    for d, p, b_min, b_max in zip(deltas, p1_coords, box_mins, box_maxs, strict=True):
        if d == 0:
            if not (b_min < p < b_max):
                return False

    # For active axes, compute open interval of t
    t_enter: Fraction | None = None
    t_leave: Fraction | None = None

    for d, p, b_min, b_max in zip(deltas, p1_coords, box_mins, box_maxs, strict=True):
        if d > 0:
            t_min = Fraction(b_min - p, d)
            t_max = Fraction(b_max - p, d)
        elif d < 0:
            t_min = Fraction(b_max - p, d)
            t_max = Fraction(b_min - p, d)
        else:
            continue

        if t_enter is None or t_min > t_enter:
            t_enter = t_min
        if t_leave is None or t_max < t_leave:
            t_leave = t_max

    assert t_enter is not None and t_leave is not None

    # Open interval (t_enter, t_leave) intersected with closed segment [0, 1]
    # Non-empty if and only if t_enter < t_leave and t_enter < 1 and t_leave > 0
    return t_enter < t_leave and t_enter < 1 and t_leave > 0
