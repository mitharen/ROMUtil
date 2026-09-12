"""Unit and integration tests for the unified renderer architecture."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

import romutil
from romutil.graph import graph
from romutil.models import AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.renderers import (
    HTML_TEMPLATE,
    RENDERERS,
    BaseRenderer,
    HTMLRenderer,
    JSONRenderer,
    Plotter,
    SVGRenderer,
    _DynamicPalette,
    build_area_json,
    export_html,
    export_json,
    generate_html_viewer,
    get_renderer,
    render_map,
)
import romutil.renderers.html as html_module


@pytest.fixture
def sample_rdb() -> tuple[dict[int, Room], AreaHeader]:
    """Create a sample solved room database with elevation layers."""
    r1 = Room(
        RoomDef(
            vnum=100,
            name="Ground Room",
            description="A room on the ground floor.",
            exits=(
                ExitDef(direction=0, dst_vnum=101),
                ExitDef(direction=4, dst_vnum=102),
            ),
        )
    )
    r1.x, r1.y, r1.z = 0, 0, 0

    r2 = Room(
        RoomDef(
            vnum=101,
            name="North Room",
            description="A room to the north.",
            exits=(ExitDef(direction=2, dst_vnum=100),),
        )
    )
    r2.x, r2.y, r2.z = 0, 1, 0

    r3 = Room(
        RoomDef(
            vnum=102,
            name="Upper Room",
            description="A room on the second floor.",
            exits=(ExitDef(direction=5, dst_vnum=100),),
        )
    )
    r3.x, r3.y, r3.z = 0, 0, 1

    header = AreaHeader(
        filename="sample.are",
        name="Sample Area",
        builder="Builder",
        vnum_min=100,
        vnum_max=102,
    )
    rdb = {100: r1, 101: r2, 102: r3}
    return rdb, header


class TestBaseRendererProtocol:
    """Test BaseRenderer protocol enforcement and runtime checkability."""

    def test_protocol_conformance(self):
        svg = SVGRenderer()
        json_r = JSONRenderer()
        html_r = HTMLRenderer()

        assert isinstance(svg, BaseRenderer)
        assert isinstance(json_r, BaseRenderer)
        assert isinstance(html_r, BaseRenderer)

    def test_non_conforming_object(self):
        class NotARenderer:
            def plot(self):
                pass

        assert not isinstance(NotARenderer(), BaseRenderer)


class TestRendererRegistry:
    """Test renderer registry, lookup, and dispatch."""

    def test_registry_contains_core_formats(self):
        assert "svg" in RENDERERS
        assert "json" in RENDERERS
        assert "html" in RENDERERS
        assert RENDERERS["svg"] is SVGRenderer
        assert RENDERERS["json"] is JSONRenderer
        assert RENDERERS["html"] is HTMLRenderer

    @pytest.mark.parametrize(
        "fmt,expected_cls",
        [
            ("svg", SVGRenderer),
            ("SVG", SVGRenderer),
            (".svg", SVGRenderer),
            ("json", JSONRenderer),
            ("JSON", JSONRenderer),
            (".json", JSONRenderer),
            ("html", HTMLRenderer),
            ("HTML", HTMLRenderer),
            (".html", HTMLRenderer),
        ],
    )
    def test_get_renderer_by_string(self, fmt, expected_cls):
        renderer = get_renderer(fmt)
        assert isinstance(renderer, expected_cls)
        assert isinstance(renderer, BaseRenderer)

    def test_get_renderer_with_instance(self):
        svg = SVGRenderer()
        assert get_renderer(svg) is svg

    def test_get_renderer_with_class(self):
        renderer = get_renderer(SVGRenderer)
        assert isinstance(renderer, SVGRenderer)

    def test_get_renderer_with_custom_duck_type_class(self):
        class CustomClass:
            def render(self, rdb, output_path, header=None, **options):
                return Path(output_path)

        renderer = get_renderer(CustomClass)
        assert isinstance(renderer, CustomClass)

    def test_get_renderer_with_non_conforming_class(self):
        class BadClass:
            pass

        with pytest.raises(TypeError, match="does not implement BaseRenderer"):
            get_renderer(BadClass)

    def test_get_renderer_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported renderer format 'pdf'"):
            get_renderer("pdf")

    def test_get_renderer_invalid_type(self):
        with pytest.raises(TypeError, match="Expected format string or BaseRenderer"):
            get_renderer(42)  # type: ignore

    def test_custom_renderer_registration(self, tmp_path):
        class MockCustomRenderer:
            def render(self, rdb, output_path, header=None, **options):
                p = Path(output_path)
                p.write_text("custom", encoding="utf-8")
                return p

        RENDERERS["custom"] = MockCustomRenderer
        try:
            r = get_renderer("custom")
            assert isinstance(r, MockCustomRenderer)
            out = tmp_path / "test.custom"
            res = r.render({}, out)
            assert res.exists()
            assert res.read_text(encoding="utf-8") == "custom"
        finally:
            del RENDERERS["custom"]


class TestRenderMapDispatcher:
    """Test render_map dispatch and file extension inference."""

    def test_render_map_svg_inferred(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "map.svg"
        result = render_map(rdb, out, header=header)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "elevation-0" in content
        assert "elevation-1" in content

    def test_render_map_json_inferred(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "map.json"
        result = render_map(rdb, out, header=header)
        assert result == out
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["area"]["name"] == "Sample Area"
        assert len(data["rooms"]) == 3

    def test_render_map_html_inferred(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "viewer.html"
        result = render_map(rdb, out, header=header)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Sample Area" in content

    def test_render_map_explicit_format(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "output_without_ext"
        result = render_map(rdb, out, fmt="json", header=header)
        assert result == out
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["rooms"]) == 3

    def test_render_map_default_to_svg_when_no_ext_and_no_fmt(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "output_no_ext"
        result = render_map(rdb, out, header=header)
        assert result == out
        assert out.exists()
        assert "<svg" in out.read_text(encoding="utf-8")

    def test_render_map_unsupported_format(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "output.unsupported"
        with pytest.raises(ValueError, match="Unsupported renderer format"):
            render_map(rdb, out, fmt="unsupported")


class TestSVGRenderer:
    """Test SVGRenderer specific functionality and options."""

    def test_svg_split_levels(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "test_split.svg"
        renderer = SVGRenderer()
        result = renderer.render(rdb, out, header=header, split_levels=True)
        assert result == out
        z0 = tmp_path / "test_split_z0.svg"
        z1 = tmp_path / "test_split_z1.svg"
        assert z0.exists()
        assert z1.exists()
        assert "<svg" in z0.read_text(encoding="utf-8")
        assert "<svg" in z1.read_text(encoding="utf-8")

    def test_svg_split_levels_custom_outbase(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "placeholder.svg"
        custom_base = tmp_path / "custom_base"
        renderer = SVGRenderer()
        renderer.render(rdb, out, header=header, split_levels=True, outbase=str(custom_base))
        z0 = tmp_path / "custom_base_z0.svg"
        z1 = tmp_path / "custom_base_z1.svg"
        assert z0.exists()
        assert z1.exists()

    def test_svg_target_z(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "level1.svg"
        renderer = SVGRenderer()
        result = renderer.render(rdb, out, header=header, target_z=1)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "elevation-1" in content
        assert "elevation-0" not in content

    def test_svg_explicit_exits(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        exits = [
            Exit(src=100, dst=101, direction=Direction.north),
        ]
        out = tmp_path / "custom_exits.svg"
        renderer = SVGRenderer()
        renderer.render(rdb, out, header=header, exits=exits)
        assert out.exists()

    def test_svg_empty_rooms(self, tmp_path):
        renderer = SVGRenderer()
        out = tmp_path / "empty.svg"
        renderer.render({}, out)
        assert out.exists()
        assert "<svg" in out.read_text(encoding="utf-8")

    def test_svg_split_levels_empty_unique_zs(self, tmp_path):
        # Room without coordinates
        r = Room(RoomDef(vnum=1, name="Void", description="Empty"))
        out = tmp_path / "empty_split.svg"
        renderer = SVGRenderer()
        renderer.render({1: r}, out, split_levels=True)
        z0 = tmp_path / "empty_split_z0.svg"
        assert z0.exists()


class TestJSONAndHTMLRenderers:
    """Test JSONRenderer and HTMLRenderer options and behavior."""

    def test_json_renderer_options(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "custom.json"
        renderer = JSONRenderer()
        custom_bounds = {"min_x": -5, "max_x": 5, "min_y": -5, "max_y": 5, "min_z": 0, "max_z": 1}
        renderer.render(rdb, out, header=header, indent=4, bounds=custom_bounds)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["bounds"]["min_x"] == -5
        assert data["bounds"]["max_x"] == 5

    def test_html_renderer_options(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "custom.html"
        renderer = HTMLRenderer()
        custom_bounds = {"min_x": -2, "max_x": 8, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 2}
        renderer.render(rdb, out, header=header, title="Custom Title", bounds=custom_bounds)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Custom Title" in content
        assert "Sample Area" in content


class TestTemplateAssetLoading:
    """Test template asset loading via importlib.resources and fallback paths."""

    def test_template_constant_populated(self):
        assert isinstance(HTML_TEMPLATE, str)
        assert len(HTML_TEMPLATE) > 1000
        assert "<!DOCTYPE html>" in HTML_TEMPLATE
        assert "__TITLE__" in HTML_TEMPLATE
        assert "__JSON_DATA__" in HTML_TEMPLATE

    def test_load_template_via_fallback(self):
        with patch("importlib.resources.files", side_effect=Exception("Simulated resources failure")):
            template = html_module._load_html_template()
            assert isinstance(template, str)
            assert "<!DOCTYPE html>" in template

    def test_load_template_failure_raises(self):
        with patch("importlib.resources.files", side_effect=Exception("Simulated resources failure")):
            with patch("romutil.renderers.html.Path.is_file", return_value=False):
                with pytest.raises(Exception, match="Simulated resources failure"):
                    html_module._load_html_template()


class TestPackageRootExports:
    """Verify renderer public API exports on the package root."""

    def test_package_root_exports(self):
        assert romutil.BaseRenderer is BaseRenderer
        assert romutil.SVGRenderer is SVGRenderer
        assert romutil.JSONRenderer is JSONRenderer
        assert romutil.HTMLRenderer is HTMLRenderer
        assert romutil.render_map is render_map
        assert romutil.get_renderer is get_renderer
        assert romutil.RENDERERS is RENDERERS
        assert romutil.HTML_TEMPLATE == HTML_TEMPLATE
        assert romutil.Plotter is Plotter
        assert romutil.build_area_json is build_area_json
        assert romutil.export_json is export_json
        assert romutil.generate_html_viewer is generate_html_viewer
        assert romutil.export_html is export_html


class TestGraphPresentationDecoupling:
    """Test that graph() delegates presentation rendering cleanly."""

    def test_graph_delegates_to_svg_renderer(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "graph_output.svg"
        graph(rdb, str(out), header)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "elevation-0" in content

    def test_graph_split_levels(self, sample_rdb, tmp_path):
        rdb, header = sample_rdb
        out = tmp_path / "graph_split.svg"
        graph(rdb, str(out), header, split_levels=True)
        z0 = tmp_path / "graph_split_z0.svg"
        z1 = tmp_path / "graph_split_z1.svg"
        assert z0.exists()
        assert z1.exists()

    def test_graph_empty_returns_early(self):
        # Empty rdb and no exits should return early without error
        assert graph({}, "dummy.svg", None) is None


class TestWebViewerVisualAndUsabilityOptimizations:
    """Unit and regression tests for Task 9c web viewer visual & usability enhancements."""

    @pytest.fixture
    def multi_floor_rdb(self) -> tuple[dict[int, Room], AreaHeader]:
        """Create a multi-floor room database with one-way and relaxed exits."""
        r1 = Room(
            RoomDef(
                vnum=100,
                name="Ground Hall",
                description="Grand hall on the ground floor.",
                exits=(
                    ExitDef(direction=0, dst_vnum=101),  # Two-way north
                    ExitDef(direction=4, dst_vnum=200),  # Inter-floor up (Z=0 -> Z=1)
                    ExitDef(direction=1, dst_vnum=105),  # One-way east (no return)
                    ExitDef(direction=2, dst_vnum=999),  # Stub south (external)
                ),
            )
        )
        r1.x, r1.y, r1.z = 0, 0, 0

        r2 = Room(
            RoomDef(
                vnum=101,
                name="North Antechamber",
                description="Antechamber north of the hall.",
                exits=(ExitDef(direction=2, dst_vnum=100),),
            )
        )
        r2.x, r2.y, r2.z = 0, 1, 0

        r3 = Room(
            RoomDef(
                vnum=200,
                name="Tower Balcony",
                description="Upper balcony overlooking the courtyard.",
                exits=(ExitDef(direction=5, dst_vnum=100),),  # Inter-floor down (Z=1 -> Z=0)
            )
        )
        r3.x, r3.y, r3.z = 0, 0, 1

        r4 = Room(
            RoomDef(
                vnum=105,
                name="Secret Cellar",
                description="A one-way trap cellar with no reciprocal return exit.",
                exits=(),  # No exits back to 100
            )
        )
        r4.x, r4.y, r4.z = 2, 0, 0

        header = AreaHeader(
            filename="tower.are",
            name="The Mystic Tower",
            builder="Archmage",
            vnum_min=100,
            vnum_max=200,
        )
        rdb = {100: r1, 101: r2, 200: r3, 105: r4}
        return rdb, header

    def test_viewport_auto_centering_logic(self, multi_floor_rdb):
        """Verify that generated HTML incorporates bounding box and centroid auto-centering logic."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify centroid and bounds computation function exists
        assert "function computeIsometricCentroidAndBounds()" in html
        assert "centroidX = (minPx + maxPx) / 2" in html
        assert "centroidY = (minPy + maxPy) / 2" in html
        assert "boxW = maxPx - minPx" in html
        assert "boxH = maxPy - minPy" in html

        # 2. Verify auto-centering and fitting in resetView
        assert "function resetView()" in html
        assert "computeIsometricCentroidAndBounds()" in html
        assert "fitZoom = Math.min(" in html
        assert "panX = vpW / 2 - centroidX * zoom" in html
        assert "panY = vpH / 2 - centroidY * zoom" in html

        # 3. Verify reset button binds to resetView
        assert "document.getElementById('btn-reset-view').addEventListener('click', resetView);" in html

    def test_dual_end_elevation_gradient_definitions(self, multi_floor_rdb):
        """Verify that inter-floor exits generate dual-end elevation gradients."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify linearGradient creation in defs for inter-floor transitions
        assert "isInterFloor = r.coords.z !== target.coords.z" in html
        assert "document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient')" in html
        assert "grad.setAttribute('gradientUnits', 'userSpaceOnUse')" in html
        assert "getRoomColor(r.coords.z)" in html
        assert "getRoomColor(target.coords.z)" in html

        # 2. Verify stroke URL assignment and CSS class
        assert "line.style.stroke = strokeStyle" in html
        assert "inter-floor" in html
        assert ".exit-line.inter-floor" in html

    def test_inclusive_floor_filtering_predicate(self, multi_floor_rdb):
        """Verify inclusive floor filter predicate where exits are visible if either endpoint matches."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify exit lines store endpoint elevation attributes
        assert "line.setAttribute('data-src-z', r.coords.z);" in html
        assert "line.setAttribute('data-dst-z', target.coords.z);" in html

        # 2. Verify inclusive floor filter visibility predicate with active floor set
        assert "activeFloors.has(srcZ) || activeFloors.has(dstZ)" in html

    def test_contextual_tooltips_markup_and_explanations(self, multi_floor_rdb):
        """Verify clear, accessible tooltips for one-way red exits and boundary/relaxation warnings."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Exact tooltip wording constants embedded in script
        assert "One-Way Exit: This exit has no reciprocal return path from the destination room." in html
        assert "Geometric Relaxation / Boundary: Exit distance relaxed due to a non-Euclidean loop contradiction or connects to an external boundary stub." in html

        # 2. Hover listener bindings for exit lines
        assert "line.addEventListener('mouseenter', (ev) => showExitTooltip(" in html
        assert "function showExitTooltip(info, ev)" in html

        # 3. Accessible title attributes in info pane exit tags
        assert "red-badge" in html
        assert "warn-badge" in html
        assert "tag.setAttribute('title', tooltipParts.join(' | '));" in html

    def test_incoming_oneway_exit_indexing_and_inspector(self, multi_floor_rdb):
        """Verify incoming one-way exits are indexed and presented with jump-to navigation in sidebar."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Reverse topological index for incoming one-way exits
        assert "const incomingOneWays = new Map();" in html
        assert "incomingOneWays.get(ex.dst).push({" in html

        # 2. Sidebar incoming exits section and badges
        assert 'id="incoming-exits-section"' in html
        assert 'id="card-room-incoming"' in html
        assert "INCOMING EXITS / ENTRANCES" in html

        # 3. Interactive jump-to click handler
        assert "centerOnRoom(inc.srcVnum);" in html
        assert "selectRoom(inc.srcVnum);" in html

    def test_interactive_edge_selection_markup_and_handlers(self, multi_floor_rdb):
        """Verify selectEdge handler, edge click listeners, CSS rules, and edge inspector card."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Assert selectEdge function and click event listeners are present on .exit-line
        assert "function selectEdge(lineElem, exitInfo)" in html
        assert "line.addEventListener('click', (ev) => {" in html
        assert "selectEdge(line, exitInfo);" in html
        assert "selectEdge(line, stubInfo);" in html

        # 2. Assert .exit-line.highlighted and .exit-line.selected CSS rules exist
        assert ".exit-line.highlighted" in html
        assert "stroke: #38bdf8 !important;" in html
        assert "stroke-width: 3.5 !important;" in html
        assert ".exit-line.selected" in html
        assert "stroke: #f59e0b !important;" in html
        assert "stroke-width: 4 !important;" in html
        assert ".room-group.selected-endpoint .room-box" in html

        # 3. Assert edge inspector card markup / fields are present in the HTML template
        assert 'id="card-edge-inspector"' in html
        assert 'id="edge-src-label"' in html
        assert 'id="btn-jump-edge-src"' in html
        assert 'id="edge-dst-label"' in html
        assert 'id="btn-jump-edge-dst"' in html
        assert 'id="edge-direction-label"' in html
        assert 'id="edge-type-label"' in html
        assert 'id="edge-geometry-label"' in html
        assert 'id="edge-elevation-label"' in html

    def test_room_selection_highlights_attached_edges(self, multi_floor_rdb):
        """Verify selectRoom queries attached exit lines and toggles highlighted and connected-neighbor."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Assert selectRoom queries/iterates attached exit lines (data-src / data-dst) and toggles highlighted
        assert "canvasRoot.querySelectorAll('.exit-line').forEach(line => {" in html
        assert "srcV === vnum || dstV === vnum" in html
        assert "line.classList.add('highlighted');" in html
        assert "line.classList.remove('highlighted', 'selected');" in html

        # 2. Assert neighbor rooms get connected-neighbor class
        assert "neighborGroup.classList.add('connected-neighbor');" in html
        assert ".room-group.connected-neighbor .room-box" in html
        assert "stroke: #38bdf8;" in html

    def test_canvas_click_deselection(self, multi_floor_rdb):
        """Assert canvas/viewport click listener clears room and edge selections."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Assert canvas/viewport click listener is present
        assert "viewport.addEventListener('click', (e) => {" in html
        assert "deselectAll();" in html

        # 2. Assert deselectAll function clears room and edge selections
        assert "function deselectAll()" in html
        assert "selectedVnum = null;" in html
        assert "selectedEdge = null;" in html
        assert "g.classList.remove('selected', 'highlighted', 'connected-neighbor', 'selected-endpoint');" in html
        assert "l.classList.remove('selected', 'highlighted');" in html

    def test_standalone_html_export_end_to_end(self, multi_floor_rdb, tmp_path):
        """End-to-end integration test writing HTML viewer to disk and verifying all enhancements."""
        rdb, header = multi_floor_rdb
        out_file = tmp_path / "interactive_map.html"
        renderer = HTMLRenderer()
        result = renderer.render(rdb, out_file, header=header, title="Interactive Tower")

        assert result == out_file
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")

        # Check document title & header
        assert "Interactive Tower" in content
        assert "The Mystic Tower" in content

        # Check all visual & usability optimizations are present in rendered file
        assert "computeIsometricCentroidAndBounds" in content
        assert "linearGradient" in content
        assert "activeFloors.has(srcZ) || activeFloors.has(dstZ)" in content
        assert "One-Way Exit: This exit has no reciprocal return path" in content
        assert "incoming-exits-section" in content
        assert "selectEdge" in content
        assert "card-edge-inspector" in content
        assert "connected-neighbor" in content
        assert "btn-collapse-sidebar" in content
        assert "btn-expand-sidebar" in content
        assert "btn-toggle-sidebar" in content
        assert "floor-pill" in content
        assert "panToMinimapCoord" in content

    def test_arbitrary_multi_floor_selection(self, multi_floor_rdb):
        """Verify multi-floor level selection controls, active floor set, and legend toggling (Task 9h)."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify floor selector container and pills markup
        assert 'id="floor-selector"' in html
        assert 'id="floor-pill-list"' in html
        assert 'id="btn-floor-all"' in html
        assert "floor-pill" in html
        assert "data-z=" in html

        # 2. Verify active floor tracking and toggle functions
        assert "const activeFloors = new Set(zValues);" in html
        assert "function toggleFloor(z)" in html
        assert "function toggleAllFloors()" in html
        assert "function applyFloorFilter()" in html

        # 3. Verify inclusive inter-floor predicate and dimming
        assert "activeFloors.has(srcZ) || activeFloors.has(dstZ)" in html
        assert "line.classList.toggle('dimmed', !isVisible);" in html
        assert "g.classList.toggle('dimmed', !isVisible);" in html

        # 4. Verify elevation legend interactivity
        assert 'id="card-elevation-legend"' in html
        assert "legend-item" in html
        assert "leg.addEventListener('click', () => toggleFloor(z));" in html

    def test_collapsible_sidebar_controls_and_layout(self, multi_floor_rdb):
        """Verify collapsible sidebar toggle button, expand tab, and collapsed CSS layout (Task 9h)."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify toggle/collapse/expand button markup and accessibility attributes
        assert 'id="btn-collapse-sidebar"' in html
        assert 'aria-label="Collapse sidebar"' in html
        assert 'id="btn-expand-sidebar"' in html
        assert 'aria-label="Open sidebar"' in html
        assert 'id="btn-toggle-sidebar"' in html

        # 2. Verify collapsed sidebar and minimap repositioning CSS
        assert "aside.sidebar.collapsed" in html
        assert "#app.collapsed aside.sidebar" in html
        assert "#app.collapsed .minimap-container" in html
        assert "right: 16px;" in html
        assert "right: 360px;" in html

        # 3. Verify sidebar toggle state handling in script
        assert "function setSidebarCollapsed(collapsed)" in html
        assert "function toggleSidebar()" in html
        assert "sidebar.classList.toggle('collapsed', collapsed);" in html

    def test_minimap_drag_navigation_and_cursors(self, multi_floor_rdb):
        """Verify minimap drag-scroll event listeners, coordinate mapping math, and cursor styles (Task 9h)."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify interactive mousedown, mousemove, mouseup, mouseleave listeners on minimap
        assert "minimap.addEventListener('mousedown'," in html
        assert "minimap.addEventListener('mousemove'," in html
        assert "minimap.addEventListener('mouseup'," in html
        assert "minimap.addEventListener('mouseleave'," in html

        # 2. Verify coordinate mapping and viewport centering math
        assert "function panToMinimapCoord(clientX, clientY)" in html
        assert "worldPx = (canvasX - padding) / scale;" in html
        assert "worldPy = (canvasY - padding) / scale;" in html
        assert "panX = vpRect.width / 2 - worldPx * zoom;" in html
        assert "panY = vpRect.height / 2 - worldPy * zoom;" in html
        assert "updateTransform();" in html

        # 3. Verify responsive cursor CSS styles
        assert "cursor: grab;" in html
        assert "cursor: grabbing;" in html
        assert "#minimap.dragging" in html

    def test_mobile_touch_gestures_and_pinch_zoom(self, multi_floor_rdb):
        """Verify mobile touch drag-to-pan, minimap touch navigation, and pinch-to-zoom (Task 9i)."""
        rdb, header = multi_floor_rdb
        data = build_area_json(rdb, area_meta=header)
        html = generate_html_viewer(data)

        # 1. Verify touch-action: none is present in CSS styling for #map-container, #map-svg, and #minimap
        assert "touch-action: none;" in html
        assert "#map-container" in html
        assert "#minimap" in html
        assert "#map-svg" in html

        # Verify touch-action: none is specifically defined in #map-container and #minimap CSS rule blocks
        assert "#map-container" in html and "touch-action: none;" in html
        assert "#minimap" in html and "touch-action: none;" in html

        # 2. Verify single-finger touch drag-to-pan on main container / SVG
        assert "mapContainer.addEventListener('touchstart'," in html
        assert "mapContainer.addEventListener('touchmove'," in html
        assert "mapContainer.addEventListener('touchend'," in html
        assert "mapContainer.addEventListener('touchcancel'," in html
        assert "panX = t.clientX - touchStartX;" in html
        assert "panY = t.clientY - touchStartY;" in html

        # 3. Verify minimap touch event listeners for real-time panning
        assert "minimap.addEventListener('touchstart'," in html
        assert "minimap.addEventListener('touchmove'," in html
        assert "minimap.addEventListener('touchend'," in html
        assert "minimap.addEventListener('touchcancel'," in html
        assert "handleMinimapTouch" in html
        assert "panToMinimapCoord(touch.clientX, touch.clientY);" in html

        # 4. Verify two-finger pinch-to-zoom Euclidean distance and midpoint centering math
        assert "Math.hypot(x2 - x1, y2 - y1)" in html
        assert "const factor = currentDist / lastPinchDist;" in html
        assert "panX = midX - (midX - panX) * (newZoom / zoom);" in html
        assert "panY = midY - (midY - panY) * (newZoom / zoom);" in html

    def test_mobile_touch_empty_area_support(self):
        """Verify touch listeners and touch-action CSS are present even on empty room databases."""
        data = build_area_json({})
        html = generate_html_viewer(data)
        assert "touch-action: none;" in html
        assert "#map-container" in html
        assert "#minimap" in html
        assert "mapContainer.addEventListener('touchstart'," in html
        assert "minimap.addEventListener('touchstart'," in html
        assert "Math.hypot(x2 - x1, y2 - y1)" in html


class TestAreaProvenanceSerialization:
    """Test area provenance ingestion and serialization in JSON and HTML (Task 10c)."""

    def test_area_provenance_json_serialization_multi_area(self):
        """Verify area_name and area_file are serialized per room in multi-area databases."""
        r1 = Room(
            RoomDef(vnum=100, name="Midgaard Gate", description="Gate", area_name="Midgaard", area_file="midgaard.are")
        )
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room(
            RoomDef(vnum=200, name="Graveyard Entrance", description="Entrance", area_name="Graveyard", area_file="grave.are")
        )
        r2.x, r2.y, r2.z = 1, 0, 0

        header = AreaHeader(
            filename="composite.are",
            name="Midgaard Metropolitan",
            builder="Various",
            vnum_min=100,
            vnum_max=200,
        )
        data = build_area_json({100: r1, 200: r2}, area_meta=header)

        assert data["area"]["name"] == "Midgaard Metropolitan"
        assert data["area"]["file"] == "composite.are"

        rooms = {r["vnum"]: r for r in data["rooms"]}
        assert rooms[100]["area_name"] == "Midgaard"
        assert rooms[100]["area_file"] == "midgaard.are"
        assert rooms[200]["area_name"] == "Graveyard"
        assert rooms[200]["area_file"] == "grave.are"

    def test_area_provenance_json_fallback_to_header(self):
        """Verify rooms without explicit area provenance fallback to header metadata."""
        r1 = Room(RoomDef(vnum=10, name="Single Room", description="Desc"))
        r1.x, r1.y, r1.z = 0, 0, 0

        header = AreaHeader(
            filename="single.are",
            name="Single Domain",
            builder="Solo",
            vnum_min=10,
            vnum_max=10,
        )
        data = build_area_json({10: r1}, area_meta=header)

        assert len(data["rooms"]) == 1
        assert data["rooms"][0]["area_name"] == "Single Domain"
        assert data["rooms"][0]["area_file"] == "single.are"


class TestMultiAreaViewerUI:
    """Test interactive zone/area filtering, legend, styling, and inspector in HTML viewer (Task 10c)."""

    @pytest.fixture
    def composite_viewer_html(self) -> str:
        r1 = Room(
            RoomDef(vnum=100, name="Town Square", description="A lively square.", area_name="Midgaard", area_file="midgaard.are")
        )
        r1.x, r1.y, r1.z = 0, 0, 0
        r2 = Room(
            RoomDef(vnum=200, name="Cemetery Gate", description="A dark gate.", area_name="Graveyard", area_file="grave.are")
        )
        r2.x, r2.y, r2.z = 1, 0, 0
        r3 = Room(
            RoomDef(vnum=300, name="Assembly Line", description="Factory machines.", area_name="Mob Factory", area_file="mobfact.are")
        )
        r3.x, r3.y, r3.z = 2, 0, 0

        header = AreaHeader(
            filename="composite.are",
            name="Midgaard + Graveyard + Mob Factory",
            builder="Builders",
            vnum_min=100,
            vnum_max=300,
        )
        data = build_area_json({100: r1, 200: r2, 300: r3}, area_meta=header)
        return generate_html_viewer(data)

    def test_zone_selector_and_legend_markup_present(self, composite_viewer_html: str):
        html = composite_viewer_html
        # Zone selector pill container in header
        assert 'id="zone-selector"' in html
        assert 'id="zone-pill-list"' in html
        assert 'id="btn-zone-all"' in html
        assert "zone-pill" in html

        # Zone legend card in sidebar
        assert 'id="card-zone-legend"' in html
        assert 'id="zone-legend-list"' in html
        assert 'id="btn-zone-legend-all"' in html
        assert 'id="btn-color-mode"' in html

    def test_zone_detection_and_palette_logic(self, composite_viewer_html: str):
        html = composite_viewer_html
        assert "const distinctAreas = Array.from(" in html
        assert "const isMultiArea = distinctAreas.length > 1;" in html
        assert "const zonePalette = [" in html
        assert "function getZoneColor(areaName)" in html
        assert "const activeZones = new Set(distinctAreas);" in html
        assert "function toggleZone(areaName)" in html
        assert "function toggleAllZones()" in html
        assert "function highlightZone(areaName)" in html
        assert "function clearZoneHighlight()" in html
        assert "function updateRoomVisualStyles()" in html

    def test_room_inspector_area_provenance_display(self, composite_viewer_html: str):
        html = composite_viewer_html
        # Room detail card provenance fields
        assert 'id="card-room-area-row"' in html
        assert 'id="card-room-area"' in html
        assert "function showRoomDetails(room)" in html
        assert "window.showRoomDetails = showRoomDetails;" in html
        assert "showRoomDetails(room);" in html

        # Meta string and title include area provenance
        assert "' | Area: ' + room.area_name" in html
        assert "'Area: ' + room.area_name" in html
        assert "areaElem.textContent = areaLabel" in html

    def test_zone_stroke_tinting_and_minimap_color_support(self, composite_viewer_html: str):
        html = composite_viewer_html
        assert "g.style.setProperty('--zone-color', getZoneColor(r.area_name));" in html
        assert ".room-group[data-area] .room-box" in html
        assert "g.setAttribute('data-area', r.area_name);" in html
        assert "colorMode === 'zone'" in html
        assert "getZoneColor(r.area_name)" in html
