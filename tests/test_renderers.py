"""Unit and integration tests for the unified renderer architecture."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

import romutil
import romutil.exporter as legacy_exporter
import romutil.plotter as legacy_plotter
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


class TestBackwardCompatibilityFacades:
    """Verify backward-compatibility re-exports across legacy modules and package root."""

    def test_legacy_plotter_imports(self):
        assert legacy_plotter.Plotter is Plotter
        assert legacy_plotter._DynamicPalette is _DynamicPalette
        assert legacy_plotter.Direction is Direction

    def test_legacy_exporter_imports(self):
        assert legacy_exporter.build_area_json is build_area_json
        assert legacy_exporter.export_json is export_json
        assert legacy_exporter.generate_html_viewer is generate_html_viewer
        assert legacy_exporter.export_html is export_html
        assert legacy_exporter.HTML_TEMPLATE == HTML_TEMPLATE

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

        # 2. Verify inclusive floor filter visibility predicate
        assert "activeZ === null || activeZ === 'all' || activeZ === srcZ || activeZ === dstZ" in html

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

        # Check all 5 visual & usability optimizations are present in rendered file
        assert "computeIsometricCentroidAndBounds" in content
        assert "linearGradient" in content
        assert "activeZ === srcZ || activeZ === dstZ" in content
        assert "One-Way Exit: This exit has no reciprocal return path" in content
        assert "incoming-exits-section" in content
