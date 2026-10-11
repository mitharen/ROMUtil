"""Debug visualization tools for individual MILP cluster layout solves."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import svgwrite

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import LayoutGraph, NodeSolution, SolveTreeNode
from romutil.layout.milp.inventory import ActiveEdge, ClusterInventory, EndpointRef
from romutil.layout.milp.solver import SolvedClusterExtraction
from romutil.types import Coord3D, SolverStatus

CHILD_PALETTE: Sequence[dict[str, str]] = (
    {"stroke": "#2563eb", "fill": "#eff6ff", "header": "#1d4ed8", "name": "blue"},
    {"stroke": "#059669", "fill": "#ecfdf5", "header": "#047857", "name": "emerald"},
    {"stroke": "#d97706", "fill": "#fffbeb", "header": "#b45309", "name": "amber"},
    {"stroke": "#7c3aed", "fill": "#f5f3ff", "header": "#6d28d9", "name": "violet"},
    {"stroke": "#db2777", "fill": "#fdf2f8", "header": "#be185d", "name": "pink"},
    {"stroke": "#0891b2", "fill": "#ecfeff", "header": "#0e7490", "name": "cyan"},
)

EDGE_CATEGORY_STYLES: Mapping[str, dict[str, str]] = {
    "corridor": {"color": "#059669", "marker": "arrow-emerald", "label": "Corridor"},
    "split_pair": {"color": "#7c3aed", "marker": "arrow-violet", "label": "Split-Pair"},
    "gateway": {"color": "#0284c7", "marker": "arrow-cyan", "label": "Gateway/Bridge"},
    "room_room": {"color": "#475569", "marker": "arrow-slate", "label": "Room-to-Room"},
}


def _resolve_endpoint_coord(
    endpoint: EndpointRef,
    extraction: SolvedClusterExtraction,
    inventory: ClusterInventory,
) -> tuple[int, int, int]:
    """Resolve an EndpointRef to raw cluster 3D coordinates (x, y, z)."""
    if isinstance(endpoint, int):
        if endpoint in extraction.raw_parent_coords:
            return extraction.raw_parent_coords[endpoint]
        for cid, c_sol in inventory.child_solutions.items():
            if endpoint in c_sol.local_coordinates:
                cx, cy, cz = extraction.raw_child_offsets.get(cid, (0, 0, 0))
                loc = c_sol.local_coordinates[endpoint]
                return (cx + loc.x, cy + loc.y, cz + loc.z)
        raise KeyError(f"Endpoint room vnum {endpoint} not found in parent coordinates or child solutions.")

    cid, vnum = endpoint
    cx, cy, cz = extraction.raw_child_offsets.get(cid, (0, 0, 0))
    c_sol = inventory.child_solutions.get(cid)
    if c_sol and vnum in c_sol.local_coordinates:
        loc = c_sol.local_coordinates[vnum]
        return (cx + loc.x, cy + loc.y, cz + loc.z)
    return (cx, cy, cz)


def _classify_edge(edge: ActiveEdge) -> str:
    """Classify an ActiveEdge into a styling category."""
    eid = edge.edge_id.lower()
    if "corridor" in eid:
        return "corridor"
    if "split" in eid:
        return "split_pair"
    if "gateway" in eid or "bridge" in eid or isinstance(edge.src, tuple) or isinstance(edge.dst, tuple):
        return "gateway"
    return "room_room"


def _direction_label(direction: tuple[int, int, int]) -> str:
    """Return a human-readable cardinal abbreviation for a 3D direction vector."""
    dx, dy, dz = direction
    if (dx, dy, dz) == (0, 1, 0):
        return "N"
    if (dx, dy, dz) == (0, -1, 0):
        return "S"
    if (dx, dy, dz) == (1, 0, 0):
        return "E"
    if (dx, dy, dz) == (-1, 0, 0):
        return "W"
    if (dx, dy, dz) == (0, 0, 1):
        return "U"
    if (dx, dy, dz) == (0, 0, -1):
        return "D"
    return f"({dx},{dy},{dz})"


def _get_child_bounding_box(
    child_id: str,
    inventory: ClusterInventory,
) -> BoundingBox3D | None:
    """Retrieve or compute the bounding box for a child cluster."""
    c_sol = inventory.child_solutions.get(child_id)
    if not c_sol:
        return None
    if c_sol.bounding_box is not None:
        return c_sol.bounding_box
    if c_sol.local_coordinates:
        coords = list(c_sol.local_coordinates.values())
        return BoundingBox3D(
            min_x=min(c.x for c in coords),
            max_x=max(c.x for c in coords),
            min_y=min(c.y for c in coords),
            max_y=max(c.y for c in coords),
            min_z=min(c.z for c in coords),
            max_z=max(c.z for c in coords),
        )
    return None


def render_cluster_solve_svg(
    extraction: SolvedClusterExtraction,
    inventory: ClusterInventory,
    output_path: Path | str,
    title: str | None = None,
    graph: LayoutGraph | None = None,
) -> Path:
    """Render an isometric SVG debug diagram of a solved layout cluster.

    Visualizes parent rooms at their solved coordinates, active edges color-coded
    by interface category, and child cluster macro-boxes and keep-out sub-boxes
    positioned at their solved offsets.

    Args:
        extraction: SolvedClusterExtraction containing raw parent coordinates and child offsets.
        inventory: ClusterInventory containing rooms, children, interfaces, and active edges.
        output_path: Target filesystem path for the output SVG file.
        title: Optional custom header title (defaults to cluster node ID).
        graph: Optional LayoutGraph providing room names and metadata.

    Returns:
        Path object pointing to the written SVG file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    effective_graph = graph if graph is not None else inventory.graph

    scale = 75.0
    lift_x = 20.0
    lift_y = -24.0

    def proj(x: float, y: float, z: float) -> tuple[float, float]:
        return (x * scale + z * lift_x, -y * scale + z * lift_y)

    # Collect screen bounds across all elements
    all_points: list[tuple[float, float]] = []

    for vnum in inventory.rooms:
        if vnum in extraction.raw_parent_coords:
            rx, ry, rz = extraction.raw_parent_coords[vnum]
            px, py = proj(rx, ry, rz)
            all_points.extend([(px - 35, py - 35), (px + 35, py + 35)])

    for child in inventory.children:
        cid = child.node_id
        cx, cy, cz = extraction.raw_child_offsets.get(cid, (0, 0, 0))
        bbox = _get_child_bounding_box(cid, inventory)
        if bbox is not None:
            corners = [
                (cx + bbox.min_x - 0.45, cy + bbox.min_y - 0.45, cz + bbox.min_z),
                (cx + bbox.max_x + 0.45, cy + bbox.min_y - 0.45, cz + bbox.min_z),
                (cx + bbox.max_x + 0.45, cy + bbox.max_y + 0.45, cz + bbox.min_z),
                (cx + bbox.min_x - 0.45, cy + bbox.max_y + 0.45, cz + bbox.min_z),
                (cx + bbox.min_x - 0.45, cy + bbox.min_y - 0.45, cz + bbox.max_z),
                (cx + bbox.max_x + 0.45, cy + bbox.min_y - 0.45, cz + bbox.max_z),
                (cx + bbox.max_x + 0.45, cy + bbox.max_y + 0.45, cz + bbox.max_z),
                (cx + bbox.min_x - 0.45, cy + bbox.max_y + 0.45, cz + bbox.max_z),
            ]
            for c in corners:
                all_points.append(proj(*c))
        else:
            all_points.append(proj(cx, cy, cz))

    for ae in inventory.active_edges:
        try:
            s_pt = _resolve_endpoint_coord(ae.src, extraction, inventory)
            d_pt = _resolve_endpoint_coord(ae.dst, extraction, inventory)
            all_points.append(proj(*s_pt))
            all_points.append(proj(*d_pt))
        except KeyError:
            continue

    if not all_points:
        all_points.append((0.0, 0.0))
        all_points.append((100.0, 100.0))

    min_sx = min(p[0] for p in all_points)
    max_sx = max(p[0] for p in all_points)
    min_sy = min(p[1] for p in all_points)
    max_sy = max(p[1] for p in all_points)

    pad_x = 70.0
    pad_y = 60.0
    header_height = 160.0

    width = max(980.0, (max_sx - min_sx) + 2 * pad_x)
    height = max(680.0, (max_sy - min_sy) + header_height + 2 * pad_y)

    offset_x = pad_x - min_sx
    offset_y = header_height + pad_y - min_sy

    def to_canvas(x: float, y: float, z: float) -> tuple[float, float]:
        sx, sy = proj(x, y, z)
        return (sx + offset_x, sy + offset_y)

    dwg = svgwrite.Drawing(
        filename=str(out_file),
        size=(f"{width:.0f}px", f"{height:.0f}px"),
    )
    dwg.viewbox(0, 0, width, height)

    # Defs: Markers and Styles
    for style_def in EDGE_CATEGORY_STYLES.values():
        m_id = style_def["marker"]
        marker = dwg.marker(
            id=m_id,
            insert=(8, 4),
            size=(8, 8),
            orient="auto",
        )
        marker.add(dwg.polygon(points=[(0, 0), (8, 4), (0, 8)], fill=style_def["color"]))
        dwg.defs.add(marker)

    # Canvas Background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="#f8fafc"))

    # Header Card
    hdr_g = dwg.g(id="header")
    hdr_card = dwg.rect(
        insert=(30, 20),
        size=(width - 60, 68),
        rx=8,
        ry=8,
        fill="#ffffff",
        stroke="#e2e8f0",
        stroke_width=1.5,
    )
    hdr_g.add(hdr_card)

    cluster_title = title if title is not None else f"Cluster {inventory.node.node_id}"
    hdr_g.add(
        dwg.text(
            cluster_title,
            insert=(50, 48),
            font_size="18px",
            font_family="system-ui, -apple-system, sans-serif",
            font_weight="bold",
            fill="#0f172a",
        )
    )

    node_type_val = getattr(inventory.node.node_type, "value", str(inventory.node.node_type))
    cut_type_val = getattr(inventory.node.cut_type, "value", str(inventory.node.cut_type)) if inventory.node.cut_type else "NONE"
    status_val = getattr(extraction.solver_status, "value", str(extraction.solver_status))
    metrics_str = f"Type: {node_type_val.upper()}  |  Cut: {cut_type_val.upper()}  |  Status: {status_val.upper()}  |  Rooms: {len(inventory.rooms)}  |  Children: {len(inventory.children)}  |  Active Edges: {len(inventory.active_edges)}"

    hdr_g.add(
        dwg.text(
            metrics_str,
            insert=(50, 72),
            font_size="12px",
            font_family="system-ui, -apple-system, sans-serif",
            fill="#475569",
        )
    )
    dwg.add(hdr_g)

    # Legend Card
    legend_g = dwg.g(id="legend")
    legend_card = dwg.rect(
        insert=(30, 96),
        size=(width - 60, 44),
        rx=8,
        ry=8,
        fill="#ffffff",
        stroke="#e2e8f0",
        stroke_width=1.5,
    )
    legend_g.add(legend_card)

    # Legend Items
    lx = 50.0
    ly = 122.0

    # Item: Parent Room
    legend_g.add(dwg.rect(insert=(lx, ly - 10), size=(14, 14), rx=3, ry=3, fill="#ffffff", stroke="#2563eb", stroke_width=2))
    legend_g.add(dwg.text("Parent Room", insert=(lx + 20, ly + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155"))
    lx += 125.0

    # Item: Boundary Port
    legend_g.add(dwg.rect(insert=(lx, ly - 10), size=(14, 14), rx=3, ry=3, fill="#fef3c7", stroke="#d97706", stroke_width=2))
    legend_g.add(dwg.text("Boundary Port", insert=(lx + 20, ly + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155"))
    lx += 130.0

    # Item: Child Macro-Box
    legend_g.add(dwg.rect(insert=(lx, ly - 10), size=(14, 14), rx=3, ry=3, fill="#eff6ff", stroke="#2563eb", stroke_width=1.5, stroke_dasharray="4,2"))
    legend_g.add(dwg.text("Child Box", insert=(lx + 20, ly + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155"))
    lx += 105.0

    # Item: Sub-Box Keep-Out
    legend_g.add(dwg.rect(insert=(lx, ly - 10), size=(14, 14), rx=2, ry=2, fill="#e0e7ff", stroke="#4f46e5", stroke_width=1.5, stroke_dasharray="2,2"))
    legend_g.add(dwg.text("Keep-Out Box", insert=(lx + 20, ly + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155"))
    lx += 120.0

    # Item: Edge styles
    for cat_name, cat_style in EDGE_CATEGORY_STYLES.items():
        legend_g.add(dwg.line(start=(lx, ly - 3), end=(lx + 20, ly - 3), stroke=cat_style["color"], stroke_width=2.5))
        legend_g.add(dwg.text(cat_style["label"], insert=(lx + 26, ly + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155"))
        lx += 125.0

    # Item: Transverse Slack
    legend_g.add(dwg.line(start=(lx, ly - 3), end=(lx + 20, ly - 3), stroke="#475569", stroke_width=2.0, stroke_dasharray="4,3"))
    legend_g.add(dwg.text("Transverse Slack", insert=(lx + 26, ly + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155"))

    dwg.add(legend_g)

    # Layer: Child Clusters
    children_g = dwg.g(id="child-clusters")
    for idx, child in enumerate(inventory.children):
        cid = child.node_id
        pal = CHILD_PALETTE[idx % len(CHILD_PALETTE)]
        cx, cy, cz = extraction.raw_child_offsets.get(cid, (0, 0, 0))
        bbox = _get_child_bounding_box(cid, inventory)
        sub_boxes = inventory.child_sub_boxes.get(cid, ())

        c_group = dwg.g(id=f"child-{cid}")

        if bbox is not None:
            gx0 = cx + bbox.min_x - 0.45
            gx1 = cx + bbox.max_x + 0.45
            gy0 = cy + bbox.min_y - 0.45
            gy1 = cy + bbox.max_y + 0.45
            gz0 = cz + bbox.min_z
            gz1 = cz + bbox.max_z

            p_nw0 = to_canvas(gx0, gy1, gz0)
            p_ne0 = to_canvas(gx1, gy1, gz0)
            p_se0 = to_canvas(gx1, gy0, gz0)
            p_sw0 = to_canvas(gx0, gy0, gz0)

            if gz0 == gz1:
                # 2D Flat bounding box polygon
                box_poly = dwg.polygon(
                    points=[p_nw0, p_ne0, p_se0, p_sw0],
                    fill=pal["fill"],
                    fill_opacity=0.45,
                    stroke=pal["stroke"],
                    stroke_width=2.0,
                    stroke_dasharray="6,4",
                )
                box_poly.set_desc(
                    title=f"Child Cluster: {cid}\nOffset: ({cx}, {cy}, {cz})\nBounding Box: X=[{bbox.min_x}..{bbox.max_x}], Y=[{bbox.min_y}..{bbox.max_y}], Z=[{bbox.min_z}..{bbox.max_z}]"
                )
                c_group.add(box_poly)
            else:
                p_nw1 = to_canvas(gx0, gy1, gz1)
                p_ne1 = to_canvas(gx1, gy1, gz1)
                p_se1 = to_canvas(gx1, gy0, gz1)
                p_sw1 = to_canvas(gx0, gy0, gz1)

                # Bottom face
                c_group.add(
                    dwg.polygon(
                        points=[p_nw0, p_ne0, p_se0, p_sw0],
                        fill=pal["fill"],
                        fill_opacity=0.25,
                        stroke=pal["stroke"],
                        stroke_width=1.0,
                        stroke_dasharray="3,3",
                    )
                )
                # Vertical edges
                for p_bot, p_top in [(p_nw0, p_nw1), (p_ne0, p_ne1), (p_se0, p_se1), (p_sw0, p_sw1)]:
                    c_group.add(
                        dwg.line(start=p_bot, end=p_top, stroke=pal["stroke"], stroke_width=1.5, stroke_dasharray="4,2")
                    )
                # Top face
                top_poly = dwg.polygon(
                    points=[p_nw1, p_ne1, p_se1, p_sw1],
                    fill=pal["fill"],
                    fill_opacity=0.45,
                    stroke=pal["stroke"],
                    stroke_width=2.0,
                    stroke_dasharray="6,4",
                )
                top_poly.set_desc(
                    title=f"Child Cluster: {cid} (3D Prism)\nOffset: ({cx}, {cy}, {cz})\nBounding Box: X=[{bbox.min_x}..{bbox.max_x}], Y=[{bbox.min_y}..{bbox.max_y}], Z=[{bbox.min_z}..{bbox.max_z}]"
                )
                c_group.add(top_poly)

            # Sub-boxes (keep-out boxes)
            for k, sb in enumerate(sub_boxes):
                sx0 = cx + sb.min_x - 0.42
                sx1 = cx + sb.max_x + 0.42
                sy0 = cy + sb.min_y - 0.42
                sy1 = cy + sb.max_y + 0.42
                sz = cz + sb.max_z

                sp_nw = to_canvas(sx0, sy1, sz)
                sp_ne = to_canvas(sx1, sy1, sz)
                sp_se = to_canvas(sx1, sy0, sz)
                sp_sw = to_canvas(sx0, sy0, sz)

                sb_poly = dwg.polygon(
                    points=[sp_nw, sp_ne, sp_se, sp_sw],
                    fill=pal["stroke"],
                    fill_opacity=0.12,
                    stroke=pal["stroke"],
                    stroke_width=1.5,
                    stroke_dasharray="3,3",
                )
                sb_poly.set_desc(
                    title=f"Keep-Out Sub-box #{k+1} ({cid})\nBounds: X=[{sb.min_x}..{sb.max_x}], Y=[{sb.min_y}..{sb.max_y}], Z=[{sb.min_z}..{sb.max_z}]"
                )
                c_group.add(sb_poly)

            # Header tag for child
            tag_pos = p_nw1 if gz0 != gz1 else p_nw0
            tag_rect = dwg.rect(
                insert=(tag_pos[0] + 4, tag_pos[1] - 18),
                size=(max(110, len(cid) * 8 + 30), 18),
                rx=4,
                ry=4,
                fill=pal["header"],
            )
            c_group.add(tag_rect)
            c_group.add(
                dwg.text(
                    f"Child: {cid}",
                    insert=(tag_pos[0] + 8, tag_pos[1] - 5),
                    font_size="10px",
                    font_weight="bold",
                    font_family="system-ui, sans-serif",
                    fill="#ffffff",
                )
            )

        # Internal constituent rooms & boundary ports inside child
        c_sol = inventory.child_solutions.get(cid)
        if c_sol and c_sol.local_coordinates:
            from romutil.layout.milp.inventory import get_child_boundary_ports
            c_boundary_ports = get_child_boundary_ports(child)
            for cvnum, cloc in c_sol.local_coordinates.items():
                cpx, cpy = to_canvas(cx + cloc.x, cy + cloc.y, cz + cloc.z)
                is_cport = (cvnum in c_boundary_ports) or (cvnum in inventory.boundary_ports)
                dot_r = 10 if is_cport else 7
                dot_fill = "#fef3c7" if is_cport else "#f1f5f9"
                dot_stroke = "#d97706" if is_cport else pal["stroke"]
                dot = dwg.circle(center=(cpx, cpy), r=dot_r, fill=dot_fill, stroke=dot_stroke, stroke_width=1.5)
                dot.set_desc(title=f"Child Internal Room: {cvnum}\nParent Coord: ({cx + cloc.x}, {cy + cloc.y}, {cz + cloc.z})\nBoundary Port: {is_cport}")
                c_group.add(dot)
                c_group.add(
                    dwg.text(
                        str(cvnum),
                        insert=(cpx, cpy + 3),
                        font_size="8px",
                        font_weight="bold",
                        font_family="system-ui, sans-serif",
                        fill="#0f172a",
                        text_anchor="middle",
                    )
                )

        children_g.add(c_group)
    dwg.add(children_g)

    # Layer: Active Edges
    edges_g = dwg.g(id="active-edges")
    for ae in inventory.active_edges:
        try:
            s_pt = _resolve_endpoint_coord(ae.src, extraction, inventory)
            d_pt = _resolve_endpoint_coord(ae.dst, extraction, inventory)
        except KeyError:
            continue

        x1, y1 = to_canvas(*s_pt)
        x2, y2 = to_canvas(*d_pt)

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        category = _classify_edge(ae)
        cat_style = EDGE_CATEGORY_STYLES[category]

        line_start = (x1, y1)
        line_end = (x2, y2)

        if dist > 44.0:
            ux = dx / dist
            uy = dy / dist
            line_start = (x1 + ux * 22.0, y1 + uy * 22.0)
            line_end = (x2 - ux * 22.0, y2 - uy * 22.0)

        edge_elem = dwg.line(
            start=line_start,
            end=line_end,
            stroke=cat_style["color"],
            stroke_width=2.5 if not ae.allow_transverse_slack else 2.0,
        )
        if ae.allow_transverse_slack:
            edge_elem["stroke-dasharray"] = "5,4"
        edge_elem["marker-end"] = f"url(#{cat_style['marker']})"

        desc_text = (
            f"Edge: {ae.edge_id}\nCategory: {cat_style['label']}\nDirection: {_direction_label(ae.direction)}\n"
            f"Min Length: {ae.min_length}\nTransverse Slack: {ae.allow_transverse_slack}\n"
            f"Src: {ae.src} -> Dst: {ae.dst}"
        )
        edge_elem.set_desc(title=desc_text)

        edge_subgroup = dwg.g(id=f"edge-{ae.edge_id}")
        edge_subgroup.add(edge_elem)

        # Midpoint label badge
        if dist > 30.0:
            mx = (line_start[0] + line_end[0]) / 2.0
            my = (line_start[1] + line_end[1]) / 2.0
            dir_str = _direction_label(ae.direction)
            slack_tag = " ~slack~" if ae.allow_transverse_slack else ""
            lbl_text = f"{dir_str} (L>={ae.min_length}){slack_tag}"

            badge_w = len(lbl_text) * 6.5 + 10
            badge_rect = dwg.rect(
                insert=(mx - badge_w / 2.0, my - 8),
                size=(badge_w, 16),
                rx=4,
                ry=4,
                fill="#ffffff",
                stroke=cat_style["color"],
                stroke_width=1.0,
            )
            badge_txt = dwg.text(
                lbl_text,
                insert=(mx, my + 4),
                font_size="9px",
                font_family="system-ui, sans-serif",
                font_weight="bold",
                fill=cat_style["color"],
                text_anchor="middle",
            )
            edge_subgroup.add(badge_rect)
            edge_subgroup.add(badge_txt)

        edges_g.add(edge_subgroup)
    dwg.add(edges_g)

    # Layer: Parent Rooms
    rooms_g = dwg.g(id="parent-rooms")
    for vnum in inventory.rooms:
        if vnum not in extraction.raw_parent_coords:
            continue
        rx, ry, rz = extraction.raw_parent_coords[vnum]
        cx, cy = to_canvas(rx, ry, rz)

        is_port = vnum in inventory.boundary_ports
        room_subgroup = dwg.g(id=f"room-{vnum}")

        r_box = dwg.rect(
            insert=(cx - 24, cy - 24),
            size=(48, 48),
            rx=7,
            ry=7,
            fill="#fef3c7" if is_port else "#ffffff",
            stroke="#d97706" if is_port else "#2563eb",
            stroke_width=2.5 if is_port else 2.0,
        )
        room_name = ""
        if effective_graph and vnum in effective_graph.rooms:
            room_name = effective_graph.rooms[vnum].name

        r_box.set_desc(
            title=f"Room {vnum}: {room_name}\nCoord: ({rx}, {ry}, {rz})\nBoundary Port: {is_port}"
        )
        room_subgroup.add(r_box)

        # Coordinate label above box
        room_subgroup.add(
            dwg.text(
                f"({rx},{ry},{rz})",
                insert=(cx, cy - 27),
                font_size="9px",
                font_family="system-ui, sans-serif",
                fill="#64748b",
                text_anchor="middle",
            )
        )

        # Vnum center label
        room_subgroup.add(
            dwg.text(
                str(vnum),
                insert=(cx, cy + 4 if not room_name else cy),
                font_size="13px",
                font_weight="bold",
                font_family="system-ui, sans-serif",
                fill="#0f172a",
                text_anchor="middle",
            )
        )

        # Truncated room name below
        if room_name:
            trunc_name = room_name[:12] + "..." if len(room_name) > 12 else room_name
            room_subgroup.add(
                dwg.text(
                    trunc_name,
                    insert=(cx, cy + 15),
                    font_size="8px",
                    font_family="system-ui, sans-serif",
                    fill="#475569",
                    text_anchor="middle",
                )
            )

        # Port tag badge
        if is_port:
            port_tag = dwg.rect(
                insert=(cx + 8, cy - 23),
                size=(15, 9),
                rx=2,
                ry=2,
                fill="#d97706",
            )
            room_subgroup.add(port_tag)
            room_subgroup.add(
                dwg.text(
                    "P",
                    insert=(cx + 15.5, cy - 16),
                    font_size="7px",
                    font_weight="bold",
                    font_family="system-ui, sans-serif",
                    fill="#ffffff",
                    text_anchor="middle",
                )
            )

        rooms_g.add(room_subgroup)
    dwg.add(rooms_g)

    dwg.save(pretty=True)
    return out_file


__all__ = [
    "render_cluster_solve_svg",
]
