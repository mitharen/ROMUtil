"""Debug visualization tools for individual MILP cluster layout solves."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import svgwrite

from romutil.layout.geometry import BoundingBox3D
from romutil.layout.ir import LayoutGraph, NodeSolution, SolveTree, SolveTreeNode
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
    return c_sol.bounding_box


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
    header_height = 185.0

    width = max(1080.0, (max_sx - min_sx) + 2 * pad_x)
    height = max(700.0, (max_sy - min_sy) + header_height + 2 * pad_y)

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

    btn_w = 72.0
    btn_h = 28.0
    btn_x = width - 30.0 - btn_w - 20.0
    btn_y = 35.0

    idx_link = dwg.a(
        href="index.html",
        target="_self",
        style="cursor: pointer; text-decoration: none;",
    )
    idx_link.set_desc(title="Back to Solve Tree Index")
    idx_link.add(
        dwg.rect(
            insert=(btn_x, btn_y),
            size=(btn_w, btn_h),
            rx=6,
            ry=6,
            fill="#2563eb",
        )
    )
    idx_link.add(
        dwg.text(
            "Index ↗",
            insert=(btn_x + btn_w / 2.0, btn_y + 18.0),
            font_size="12px",
            font_weight="bold",
            font_family="system-ui, -apple-system, sans-serif",
            fill="#ffffff",
            text_anchor="middle",
        )
    )
    hdr_g.add(idx_link)

    parent_id = inventory.node.metadata.get("parent_id")
    if parent_id:
        p_btn_w = max(80.0, len(str(parent_id)) * 7.0 + 30.0)
        p_btn_x = btn_x - p_btn_w - 10.0
        p_link = dwg.a(
            href=f"cluster_{parent_id}.svg",
            target="_self",
            style="cursor: pointer; text-decoration: none;",
        )
        p_link.set_desc(title=f"Navigate to parent cluster {parent_id}")
        p_link.add(
            dwg.rect(
                insert=(p_btn_x, btn_y),
                size=(p_btn_w, btn_h),
                rx=6,
                ry=6,
                fill="#f1f5f9",
                stroke="#cbd5e1",
                stroke_width=1.0,
            )
        )
        p_link.add(
            dwg.text(
                f"↑ {parent_id}",
                insert=(p_btn_x + p_btn_w / 2.0, btn_y + 18.0),
                font_size="11px",
                font_weight="bold",
                font_family="system-ui, -apple-system, sans-serif",
                fill="#334155",
                text_anchor="middle",
            )
        )
        hdr_g.add(p_link)

    dwg.add(hdr_g)

    # Legend Card (Two-Tier)
    legend_g = dwg.g(id="legend")
    legend_card = dwg.rect(
        insert=(30, 96),
        size=(width - 60, 74),
        rx=8,
        ry=8,
        fill="#ffffff",
        stroke="#e2e8f0",
        stroke_width=1.5,
    )
    legend_g.add(legend_card)

    # Row 1: Nodes & Interfaces
    lx1 = 50.0
    ly1 = 119.0

    # Item: Parent Room
    legend_g.add(
        dwg.rect(insert=(lx1, ly1 - 10), size=(14, 14), rx=3, ry=3, fill="#ffffff", stroke="#2563eb", stroke_width=2)
    )
    legend_g.add(
        dwg.text("Parent Room", insert=(lx1 + 20, ly1 + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155")
    )
    lx1 += 120.0

    # Item: Parent Port
    legend_g.add(
        dwg.rect(insert=(lx1, ly1 - 10), size=(14, 14), rx=3, ry=3, fill="#fef3c7", stroke="#d97706", stroke_width=2)
    )
    legend_g.add(
        dwg.text("Parent Port", insert=(lx1 + 20, ly1 + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155")
    )
    lx1 += 115.0

    # Item: Child Room
    legend_g.add(
        dwg.circle(center=(lx1 + 7, ly1 - 3), r=6, fill="#f1f5f9", stroke="#2563eb", stroke_width=1.5)
    )
    legend_g.add(
        dwg.text("Child Room", insert=(lx1 + 20, ly1 + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155")
    )
    lx1 += 110.0

    # Item: Child Port
    legend_g.add(
        dwg.circle(center=(lx1 + 7, ly1 - 3), r=6, fill="#fef3c7", stroke="#d97706", stroke_width=1.5)
    )
    legend_g.add(
        dwg.text("Child Port", insert=(lx1 + 20, ly1 + 2), font_size="11px", font_family="system-ui, sans-serif", fill="#334155")
    )

    # Row 2: Envelopes & Corridors
    lx2 = 50.0
    ly2 = 150.0

    # Item: Child Macro-Box (Envelope)
    legend_g.add(
        dwg.rect(
            insert=(lx2, ly2 - 10),
            size=(14, 14),
            rx=3,
            ry=3,
            fill="#eff6ff",
            stroke="#2563eb",
            stroke_width=1.5,
            stroke_dasharray="4,2",
        )
    )
    legend_g.add(
        dwg.text(
            "Child Macro-Box (Envelope)",
            insert=(lx2 + 20, ly2 + 2),
            font_size="11px",
            font_family="system-ui, sans-serif",
            fill="#334155",
        )
    )
    lx2 += 195.0

    # Item: Keep-Out Sub-Box (Obstacle)
    legend_g.add(
        dwg.rect(
            insert=(lx2, ly2 - 10),
            size=(14, 14),
            rx=2,
            ry=2,
            fill="#e0e7ff",
            stroke="#4f46e5",
            stroke_width=1.5,
            stroke_dasharray="2,2",
        )
    )
    legend_g.add(
        dwg.text(
            "Keep-Out Sub-Box (Obstacle)",
            insert=(lx2 + 20, ly2 + 2),
            font_size="11px",
            font_family="system-ui, sans-serif",
            fill="#334155",
        )
    )
    lx2 += 200.0

    # Item: Edge styles
    for cat_name, cat_style in EDGE_CATEGORY_STYLES.items():
        legend_g.add(
            dwg.line(start=(lx2, ly2 - 3), end=(lx2 + 18, ly2 - 3), stroke=cat_style["color"], stroke_width=2.5)
        )
        legend_g.add(
            dwg.text(
                cat_style["label"],
                insert=(lx2 + 24, ly2 + 2),
                font_size="11px",
                font_family="system-ui, sans-serif",
                fill="#334155",
            )
        )
        if cat_name == "corridor":
            lx2 += 80.0
        elif cat_name == "split_pair":
            lx2 += 90.0
        elif cat_name == "gateway":
            lx2 += 122.0
        else:
            lx2 += 112.0

    # Item: Transverse Slack
    legend_g.add(
        dwg.line(
            start=(lx2, ly2 - 3),
            end=(lx2 + 18, ly2 - 3),
            stroke="#475569",
            stroke_width=2.0,
            stroke_dasharray="4,3",
        )
    )
    legend_g.add(
        dwg.text(
            "Transverse Slack",
            insert=(lx2 + 24, ly2 + 2),
            font_size="11px",
            font_family="system-ui, sans-serif",
            fill="#334155",
        )
    )

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
                box_link = dwg.a(href=f"cluster_{cid}.svg", target="_self", style="cursor: pointer;")
                box_link.set_desc(title=f"Click to view child cluster {cid}")
                box_link.add(box_poly)
                c_group.add(box_link)
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
                top_link = dwg.a(href=f"cluster_{cid}.svg", target="_self", style="cursor: pointer;")
                top_link.set_desc(title=f"Click to view child cluster {cid}")
                top_link.add(top_poly)
                c_group.add(top_link)

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
            tag_w = max(110, len(cid) * 8 + 30)
            tag_h = 18
            tag_x = tag_pos[0] + 6
            tag_y = tag_pos[1] + 6

            tag_link = dwg.a(
                href=f"cluster_{cid}.svg",
                target="_self",
                style="cursor: pointer; text-decoration: none;",
            )
            tag_link.set_desc(title=f"Click to view child cluster {cid}")

            tag_rect = dwg.rect(
                insert=(tag_x, tag_y),
                size=(tag_w, tag_h),
                rx=4,
                ry=4,
                fill=pal["header"],
            )
            tag_link.add(tag_rect)
            tag_link.add(
                dwg.text(
                    f"Child: {cid}",
                    insert=(tag_x + 8, tag_y + 13),
                    font_size="10px",
                    font_weight="bold",
                    font_family="system-ui, sans-serif",
                    fill="#ffffff",
                )
            )
            c_group.add(tag_link)

        # Internal constituent rooms & boundary ports inside child
        c_sol = inventory.child_solutions.get(cid)
        if c_sol and c_sol.local_coordinates:
            from romutil.layout.milp.inventory import get_child_boundary_ports

            c_boundary_ports = get_child_boundary_ports(child)
            for cvnum, cloc in c_sol.local_coordinates.items():
                cpx, cpy = to_canvas(cx + cloc.x, cy + cloc.y, cz + cloc.z)
                is_cport = (cvnum in c_boundary_ports) or (cvnum in inventory.boundary_ports)
                dot_r = 15 if is_cport else 13
                dot_fill = "#fef3c7" if is_cport else "#f1f5f9"
                dot_stroke = "#d97706" if is_cport else pal["stroke"]
                dot = dwg.circle(center=(cpx, cpy), r=dot_r, fill=dot_fill, stroke=dot_stroke, stroke_width=1.5)
                dot.set_desc(
                    title=f"Child Internal Room: {cvnum}\nParent Coord: ({cx + cloc.x}, {cy + cloc.y}, {cz + cloc.z})\nBoundary Port: {is_cport}"
                )
                c_group.add(dot)
                c_group.add(
                    dwg.text(
                        str(cvnum),
                        insert=(cpx, cpy + 4),
                        font_size="11px",
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


def _node_type_css_class(node_type_str: str) -> str:
    """Map a TreeNodeType name to a corresponding badge CSS styling class."""
    s = node_type_str.lower()
    if "rigid" in s:
        return "type-rigid"
    if "series" in s:
        return "type-series"
    if "split" in s or "parallel" in s:
        return "type-split"
    if "block" in s:
        return "type-block"
    if "cycle" in s:
        return "type-cycle"
    return "type-leaf"


def render_debug_solve_tree_index(
    tree: SolveTree,
    output_dir: Path | str,
    area_name: str | None = None,
) -> Path:
    """Render an interactive hierarchical HTML dashboard for the solve tree and debug SVGs.

    Produces an index.html file featuring a two-pane interactive browser:
    - Left pane: Collapsible hierarchical SolveTree navigation with type badges, cut contracts,
      room counts, child metrics, and live search filtering.
    - Right pane: Embedded SVG preview dynamically displaying cluster solve diagrams with drill-down links.

    Args:
        tree: Hierarchical SolveTree containing root and disconnected component nodes.
        output_dir: Target filesystem directory where index.html and cluster SVGs are located.
        area_name: Optional human-readable name of the area being visualized.

    Returns:
        Path to the generated index.html file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / "index.html"

    # Traverse all nodes to compute aggregate metrics
    all_nodes: list[SolveTreeNode] = []

    def _traverse_nodes(n: SolveTreeNode) -> None:
        all_nodes.append(n)
        for ch in n.children:
            _traverse_nodes(ch)

    _traverse_nodes(tree.root)
    for comp in tree.components:
        _traverse_nodes(comp)

    total_nodes = len(all_nodes)
    all_rooms: set[int] = {vnum for n in all_nodes for vnum in n.rooms}
    total_rooms = len(all_rooms)

    def _calc_depth(n: SolveTreeNode) -> int:
        if not n.children:
            return 1
        return 1 + max(_calc_depth(ch) for ch in n.children)

    tree_depth = max([_calc_depth(tree.root)] + [_calc_depth(c) for c in tree.components])

    clean_area_name = area_name
    if not clean_area_name:
        if out_dir.name and out_dir.name != "debug_solves":
            clean_area_name = out_dir.name
        elif out_dir.parent and out_dir.parent.name and out_dir.parent.name != "debug_solves":
            clean_area_name = out_dir.parent.name

    display_area_title = clean_area_name.title() if clean_area_name else "Solve Tree"

    def _render_tree_item(node: SolveTreeNode, is_root: bool = False) -> str:
        escaped_id = html.escape(node.node_id)
        node_type_val = getattr(node.node_type, "value", str(node.node_type))
        node_type_label = html.escape(node_type_val.upper())
        type_class = _node_type_css_class(node_type_val)

        cut_badge_html = ""
        if node.cut_type is not None:
            cut_val = getattr(node.cut_type, "value", str(node.cut_type))
            cut_badge_html = f'<span class="badge-cut">{html.escape(cut_val.upper())}</span>'

        rooms_sorted = sorted(node.rooms)
        rooms_attr = html.escape(",".join(str(r) for r in rooms_sorted))
        room_count = len(node.rooms)
        child_count = len(node.children)

        svg_filename = f"cluster_{node.node_id}.svg"
        active_cls = " active" if is_root else ""

        if node.children:
            children_html = "".join(_render_tree_item(c, False) for c in node.children)
            return (
                f'<details class="tree-item" open data-node-id="{escaped_id}" data-rooms="{rooms_attr}" data-type="{html.escape(node_type_val)}">\n'
                f'  <summary class="tree-row{active_cls}" onclick="handleRowClick(event, \'{escaped_id}\', \'{node_type_label}\', {room_count}, {child_count})">\n'
                f'    <span class="caret">▶</span>\n'
                f'    <span class="badge-type {type_class}">{node_type_label}</span>\n'
                f'    {cut_badge_html}\n'
                f'    <span class="node-title" title="{escaped_id}">{escaped_id}</span>\n'
                f'    <span class="pill-count">{room_count}r</span>\n'
                f'    <span class="pill-count">{child_count}c</span>\n'
                f'    <a href="{svg_filename}" target="_blank" class="btn-svg-ext" title="Open {svg_filename} in new tab" onclick="event.stopPropagation();">↗</a>\n'
                f'  </summary>\n'
                f'  <div class="tree-children">\n'
                f'{children_html}'
                f'  </div>\n'
                f'</details>\n'
            )
        else:
            return (
                f'<div class="tree-item leaf" data-node-id="{escaped_id}" data-rooms="{rooms_attr}" data-type="{html.escape(node_type_val)}">\n'
                f'  <div class="tree-row{active_cls}" onclick="handleRowClick(event, \'{escaped_id}\', \'{node_type_label}\', {room_count}, 0)">\n'
                f'    <span class="leaf-dot">•</span>\n'
                f'    <span class="badge-type {type_class}">{node_type_label}</span>\n'
                f'    {cut_badge_html}\n'
                f'    <span class="node-title" title="{escaped_id}">{escaped_id}</span>\n'
                f'    <span class="pill-count">{room_count}r</span>\n'
                f'    <a href="{svg_filename}" target="_blank" class="btn-svg-ext" title="Open {svg_filename} in new tab" onclick="event.stopPropagation();">↗</a>\n'
                f'  </div>\n'
                f'</div>\n'
            )

    root_tree_html = _render_tree_item(tree.root, is_root=True)
    components_html = ""
    if tree.components:
        comp_items = "".join(_render_tree_item(c, False) for c in tree.components)
        components_html = (
            f'<div class="component-divider">\n'
            f'  <span>Disconnected Components ({len(tree.components)})</span>\n'
            f'</div>\n'
            f'{comp_items}\n'
        )

    area_nav_links = ""
    if clean_area_name:
        area_nav_links = (
            f'<a href="../../{clean_area_name}.html" class="nav-btn" title="View compiled area map">Area Map ({clean_area_name}.html) ↗</a>'
        )

    root_id_escaped = html.escape(tree.root.node_id)
    root_type_val = getattr(tree.root.node_type, "value", str(tree.root.node_type))
    root_type_label = html.escape(root_type_val.upper())
    root_rooms_count = len(tree.root.rooms)
    root_child_count = len(tree.root.children)
    initial_svg = f"cluster_{tree.root.node_id}.svg"

    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SolveTree Explorer - {html.escape(display_area_title)}</title>
  <style>
    :root {{
      --bg-header: #0f172a;
      --border-color: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --active-bg: #eff6ff;
      --active-border: #93c5fd;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f8fafc;
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      background: var(--bg-header);
      color: #f8fafc;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #1e293b;
      flex-shrink: 0;
      gap: 16px;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .header-title {{
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.01em;
      white-space: nowrap;
    }}
    .header-metrics {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .metric-chip {{
      background: #1e293b;
      padding: 3px 8px;
      border-radius: 5px;
      border: 1px solid #334155;
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
    }}
    .metric-chip .label {{ color: #94a3b8; }}
    .metric-chip .val {{ font-weight: 600; color: #f8fafc; font-family: ui-monospace, monospace; }}
    .badge-status {{
      background: #065f46;
      color: #6ee7b7;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 10px;
      letter-spacing: 0.04em;
    }}
    .header-nav {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }}
    .nav-btn {{
      background: #1e293b;
      color: #93c5fd;
      border: 1px solid #3b82f6;
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 11px;
      text-decoration: none;
      font-weight: 600;
      transition: background 0.15s, color 0.15s;
      white-space: nowrap;
    }}
    .nav-btn:hover {{
      background: var(--primary);
      color: #ffffff;
    }}
    .main-layout {{
      display: flex;
      flex: 1;
      height: calc(100vh - 48px);
      overflow: hidden;
    }}
    .sidebar {{
      width: 440px;
      flex-shrink: 0;
      background: #ffffff;
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .sidebar-controls {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-color);
      background: #f8fafc;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .search-input {{
      width: 100%;
      padding: 6px 10px;
      border: 1px solid var(--border-color);
      border-radius: 5px;
      font-size: 12px;
      outline: none;
      transition: border-color 0.15s;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    .btn-row {{
      display: flex;
      gap: 6px;
      justify-content: flex-end;
    }}
    .btn-sm {{
      background: #f1f5f9;
      border: 1px solid var(--border-color);
      color: #475569;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      cursor: pointer;
      font-weight: 600;
    }}
    .btn-sm:hover {{ background: #e2e8f0; }}
    .tree-scroll {{
      flex: 1;
      overflow-y: auto;
      padding: 8px 6px;
    }}
    details.tree-item {{
      margin: 0;
      border-left: none;
      padding-left: 0;
    }}
    div.tree-item.leaf {{
      margin: 0;
      border-left: none;
      padding-left: 0;
    }}
    .tree-scroll > details.tree-item,
    .tree-scroll > div.tree-item.leaf,
    .tree-scroll > .tree-item {{
      margin-left: 0;
      border-left: none;
      padding-left: 0;
    }}
    .tree-children {{
      margin-left: 14px;
      padding-left: 6px;
      border-left: 1px dashed #cbd5e1;
    }}
    .tree-row {{
      list-style: none;
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 5px 6px;
      border-radius: 5px;
      cursor: pointer;
      user-select: none;
      transition: background 0.1s;
      font-size: 11px;
      margin-bottom: 1px;
    }}
    .tree-row::-webkit-details-marker {{ display: none; }}
    .tree-row:hover {{ background: #f1f5f9; }}
    .tree-row.active {{
      background: var(--active-bg);
      border: 1px solid var(--active-border);
    }}
    .caret {{
      display: inline-block;
      width: 12px;
      text-align: center;
      font-size: 9px;
      color: #94a3b8;
      transition: transform 0.15s;
      flex-shrink: 0;
    }}
    details[open] > summary > .caret {{
      transform: rotate(90deg);
    }}
    .leaf-dot {{
      display: inline-block;
      width: 12px;
      text-align: center;
      font-size: 10px;
      color: #cbd5e1;
      flex-shrink: 0;
    }}
    .badge-type {{
      font-size: 9px;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 3px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .type-rigid {{ background: #dbeafe; color: #1e40af; }}
    .type-series {{ background: #f3e8ff; color: #6b21a8; }}
    .type-split {{ background: #e0e7ff; color: #3730a3; }}
    .type-block {{ background: #d1fae5; color: #065f46; }}
    .type-cycle {{ background: #fef3c7; color: #92400e; }}
    .type-leaf {{ background: #f1f5f9; color: #475569; }}
    .badge-cut {{
      font-size: 8px;
      font-weight: 600;
      padding: 1px 4px;
      border-radius: 3px;
      text-transform: uppercase;
      background: #fae8ff;
      color: #86198f;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .node-title {{
      font-family: ui-monospace, monospace;
      font-weight: 600;
      color: #0f172a;
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      min-width: 0;
    }}
    .pill-count {{
      font-size: 9px;
      color: #64748b;
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 8px;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .btn-svg-ext {{
      color: #94a3b8;
      text-decoration: none;
      padding: 1px 3px;
      border-radius: 3px;
      font-size: 11px;
      line-height: 1;
      flex-shrink: 0;
    }}
    .btn-svg-ext:hover {{ color: var(--primary); background: #e0e7ff; }}
    .component-divider {{
      padding: 8px 6px 4px 6px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #64748b;
      border-top: 1px solid var(--border-color);
      margin-top: 8px;
    }}
    .preview-pane {{
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #f8fafc;
      overflow: hidden;
    }}
    .preview-bar {{
      padding: 8px 16px;
      background: #ffffff;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }}
    .preview-meta {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .active-title {{
      font-size: 14px;
      font-weight: 700;
      font-family: ui-monospace, monospace;
      color: #0f172a;
    }}
    .btn-preview-ext {{
      background: var(--primary);
      color: #ffffff;
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 11px;
      font-weight: 600;
      text-decoration: none;
      transition: background 0.15s;
    }}
    .btn-preview-ext:hover {{ background: #1d4ed8; }}
    .preview-frame-wrap {{
      flex: 1;
      position: relative;
      background: #f8fafc;
    }}
    .preview-frame-wrap iframe {{
      width: 100%;
      height: 100%;
      border: none;
      display: block;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="header-title">SolveTree Explorer: {html.escape(display_area_title)}</div>
      <div class="header-metrics">
        <div class="metric-chip"><span class="label">Root:</span><span class="val">{root_id_escaped}</span></div>
        <div class="metric-chip"><span class="label">Nodes:</span><span class="val">{total_nodes}</span></div>
        <div class="metric-chip"><span class="label">Depth:</span><span class="val">{tree_depth}</span></div>
        <div class="metric-chip"><span class="label">Rooms:</span><span class="val">{total_rooms}</span></div>
        <span class="badge-status">OPTIMAL</span>
      </div>
    </div>
    <div class="header-nav">
      {area_nav_links}
    </div>
  </header>
  <div class="main-layout">
    <div class="sidebar">
      <div class="sidebar-controls">
        <input type="text" id="node-search" class="search-input" placeholder="Filter by Node ID, Type, or Room VNUM..." oninput="filterTree(this.value)">
        <div class="btn-row">
          <button type="button" class="btn-sm" onclick="expandAll()">Expand All</button>
          <button type="button" class="btn-sm" onclick="collapseAll()">Collapse All</button>
        </div>
      </div>
      <div class="tree-scroll">
        {root_tree_html}
        {components_html}
      </div>
    </div>
    <div class="preview-pane">
      <div class="preview-bar">
        <div class="preview-meta">
          <span class="active-title" id="active-title">Cluster {root_id_escaped}</span>
          <span class="badge-type type-rigid" id="active-type">{root_type_label}</span>
          <span class="pill-count" id="active-rooms">{root_rooms_count} rooms</span>
          <span class="pill-count" id="active-children">{root_child_count} children</span>
        </div>
        <div>
          <a href="{initial_svg}" target="_blank" class="btn-preview-ext" id="active-ext-link">Open Full SVG ↗</a>
        </div>
      </div>
      <div class="preview-frame-wrap">
        <iframe id="svg-frame" src="{initial_svg}" title="Cluster Solve Diagram"></iframe>
      </div>
    </div>
  </div>
  <script>
    function handleRowClick(event, nodeId, nodeType, roomCount, childCount) {{
      selectCluster(nodeId, nodeType, roomCount, childCount);
    }}

    function selectCluster(nodeId, nodeType, roomCount, childCount) {{
      document.getElementById('active-title').textContent = 'Cluster ' + nodeId;
      const typeEl = document.getElementById('active-type');
      typeEl.textContent = nodeType;
      typeEl.className = 'badge-type ' + getNodeTypeClass(nodeType);

      document.getElementById('active-rooms').textContent = roomCount + ' rooms';
      document.getElementById('active-children').textContent = childCount + ' children';
      document.getElementById('active-ext-link').href = 'cluster_' + nodeId + '.svg';
      document.getElementById('svg-frame').src = 'cluster_' + nodeId + '.svg';

      document.querySelectorAll('.tree-row').forEach(el => el.classList.remove('active'));
      const activeItem = document.querySelector(`[data-node-id="${{nodeId}}"]`);
      if (activeItem) {{
        const row = activeItem.querySelector('.tree-row');
        if (row) row.classList.add('active');
      }}
    }}

    function getNodeTypeClass(typeStr) {{
      const s = (typeStr || '').toLowerCase();
      if (s.includes('rigid')) return 'type-rigid';
      if (s.includes('series')) return 'type-series';
      if (s.includes('split') || s.includes('parallel')) return 'type-split';
      if (s.includes('block')) return 'type-block';
      if (s.includes('cycle')) return 'type-cycle';
      return 'type-leaf';
    }}

    function filterTree(query) {{
      query = (query || '').toLowerCase().trim();
      const items = document.querySelectorAll('.tree-item');
      if (!query) {{
        items.forEach(el => el.style.display = '');
        return;
      }}
      items.forEach(el => {{
        const id = (el.getAttribute('data-node-id') || '').toLowerCase();
        const rooms = (el.getAttribute('data-rooms') || '').toLowerCase();
        const type = (el.getAttribute('data-type') || '').toLowerCase();
        const match = id.includes(query) || rooms.includes(query) || type.includes(query);
        if (match) {{
          el.style.display = '';
          let parent = el.parentElement;
          while (parent) {{
            if (parent.tagName === 'DETAILS') {{
              parent.open = true;
              parent.style.display = '';
            }}
            parent = parent.parentElement;
          }}
        }} else {{
          const childMatches = el.querySelectorAll(`.tree-item[data-node-id*="${{query}}"]`);
          if (childMatches.length > 0) {{
            el.style.display = '';
          }} else {{
            el.style.display = 'none';
          }}
        }}
      }});
    }}

    function expandAll() {{
      document.querySelectorAll('details.tree-item').forEach(d => d.open = true);
    }}

    function collapseAll() {{
      document.querySelectorAll('details.tree-item').forEach(d => d.open = false);
      const root = document.querySelector('details.tree-item');
      if (root) root.open = true;
    }}

    // Synchronize left pane if a child link is clicked inside the SVG iframe
    const frame = document.getElementById('svg-frame');
    frame.addEventListener('load', () => {{
      try {{
        const p = frame.contentWindow.location.pathname;
        const match = p.match(/cluster_(.+)\\.svg$/);
        if (match) {{
          const nodeId = match[1];
          const activeItem = document.querySelector(`[data-node-id="${{nodeId}}"]`);
          if (activeItem) {{
            document.querySelectorAll('.tree-row').forEach(el => el.classList.remove('active'));
            const row = activeItem.querySelector('.tree-row');
            if (row) {{
              row.classList.add('active');
              row.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
            }}
            const typeStr = activeItem.getAttribute('data-type') || '';
            document.getElementById('active-title').textContent = 'Cluster ' + nodeId;
            const typeEl = document.getElementById('active-type');
            typeEl.textContent = typeStr.toUpperCase();
            typeEl.className = 'badge-type ' + getNodeTypeClass(typeStr);
            document.getElementById('active-ext-link').href = 'cluster_' + nodeId + '.svg';
          }}
        }}
      }} catch (e) {{
        // Fallback for cross-origin or local restrictions
      }}
    }});
  </script>
</body>
</html>
"""
    index_file.write_text(content, encoding="utf-8")
    return index_file


__all__ = [
    "render_cluster_solve_svg",
    "render_debug_solve_tree_index",
]
