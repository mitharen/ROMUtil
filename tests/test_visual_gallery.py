import json
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Validation Helper Functions
# ---------------------------------------------------------------------------

def validate_svg_file(svg_path: Path) -> ET.Element:
    """Parses an SVG file, verifies root tag and basic attributes, returns root element."""
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_path}")
    tree = ET.parse(svg_path)
    root = tree.getroot()
    tag_clean = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag_clean.lower() != "svg":
        raise ValueError(f"Root element is <{root.tag}>, expected <svg>")
    return root


def validate_svg_content(content: str) -> ET.Element:
    """Parses an SVG string and verifies root tag, returns root element."""
    root = ET.fromstring(content)
    tag_clean = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag_clean.lower() != "svg":
        raise ValueError(f"Root element is <{root.tag}>, expected <svg>")
    return root


def validate_html_viewer_file(html_path: Path) -> dict:
    """Validates structure and zero-dependency invariant of an HTML viewer file."""
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    content = html_path.read_text(encoding="utf-8")
    return validate_html_viewer_content(content)


def validate_html_viewer_content(content: str) -> dict:
    """Validates structure and zero-dependency invariant of HTML viewer content.

    Returns the parsed embedded room database JSON dictionary.
    """
    if "<!DOCTYPE html>" not in content and "<!doctype html>" not in content:
        raise ValueError("Missing <!DOCTYPE html> declaration")
    if "<html" not in content or "</html>" not in content:
        raise ValueError("Malformed HTML: missing <html> tags")

    # Security / Offline Invariant: Zero external scripts or stylesheets
    for tag in re.findall(r"<script\b[^>]*>", content, re.IGNORECASE):
        if "src=" in tag.lower():
            raise ValueError(f"Forbidden external script tag: {tag}")

    for tag in re.findall(r"<link\b[^>]*>", content, re.IGNORECASE):
        if "stylesheet" in tag.lower():
            raise ValueError(f"Forbidden external stylesheet tag: {tag}")

    external_urls = re.findall(r"""(?:src|href)=['"]https?://[^'">]+['"]""", content, re.IGNORECASE)
    if external_urls:
        raise ValueError(f"Forbidden external resource URLs found: {external_urls}")

    if not re.search(r"""<svg\b[^>]*id=['"]map-svg['"]""", content) and "<svg" not in content:
        raise ValueError("Missing SVG map element in HTML viewer")
    if not re.search(r"""<canvas\b[^>]*id=['"]minimap['"]""", content):
        raise ValueError("Missing minimap canvas element in HTML viewer")

    # Extract and parse embedded JSON data
    match = re.search(
        r"""<script\s+id=['"]romutil-data['"]\s+type=['"]application/json['"]>(.*?)</script>""",
        content,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Missing embedded romutil-data JSON script tag")

    raw_json = match.group(1).strip()
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("Embedded JSON payload is not a dictionary")
    for key in ("area", "bounds", "rooms"):
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in embedded JSON payload")

    return data




# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_gallery_artifacts(tmp_path_factory):
    """Dynamically generates sample SVG maps and HTML viewer into an ephemeral directory."""
    out_dir = tmp_path_factory.mktemp("gallery")
    fixture_area = REPO_ROOT / "tests" / "fixtures" / "dialects" / "rom24.are"
    outbase = out_dir / "rom24"

    # Generate split-level SVGs
    subprocess.run(
        [sys.executable, "-m", "romutil.cli", str(fixture_area), "-outbase", str(outbase), "--split-levels"],
        check=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    # Generate standalone HTML viewer
    subprocess.run(
        [sys.executable, "-m", "romutil.cli", str(fixture_area), "-outbase", str(outbase), "--format", "html"],
        check=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    return out_dir


class TestGalleryAssetsGeneration:
    """Verifies that map and viewer artifacts are dynamically generated with correct outputs."""

    def test_dynamic_generation_creates_expected_files(self, sample_gallery_artifacts):
        assert (sample_gallery_artifacts / "rom24_z0.svg").exists()
        assert (sample_gallery_artifacts / "rom24.html").exists()

    def test_pages_builder_discovery_and_index_generation(self, tmp_path):
        from scripts.build_pages import resolve_candidate_areas, generate_index_html
        candidates = resolve_candidate_areas()
        assert len(candidates) >= 1
        dummy_rendered = [("fixture", "rom24")]

        # Case 1: Asset files do not exist -> verify 'N/A' fallback is emitted
        generate_index_html(dummy_rendered, tmp_path)
        index_file = tmp_path / "index.html"
        assert index_file.exists()
        content_missing = index_file.read_text(encoding="utf-8")
        assert "ROMUtil Map Directory" in content_missing
        assert "rom24" in content_missing
        assert "N/A" in content_missing
        assert '<a href="rom24.html">' not in content_missing
        assert '<a href="rom240.svg">' not in content_missing

        # Case 2: Asset files exist -> verify valid <a href= links and 'N/A' is absent
        (tmp_path / "rom24.html").write_text("<!DOCTYPE html><html><body>viewer</body></html>", encoding="utf-8")
        (tmp_path / "rom240.svg").write_text("<svg></svg>", encoding="utf-8")
        generate_index_html(dummy_rendered, tmp_path)
        content_present = index_file.read_text(encoding="utf-8")
        assert '<a href="rom24.html">Interactive Viewer</a>' in content_present
        assert '<a href="rom240.svg">SVG Map</a>' in content_present
        assert "N/A" not in content_present

        # Case 2b: Fallback to <name>.svg when <name>0.svg is not present
        (tmp_path / "rom240.svg").unlink()
        (tmp_path / "rom24.svg").write_text("<svg></svg>", encoding="utf-8")
        generate_index_html(dummy_rendered, tmp_path)
        content_single_svg = index_file.read_text(encoding="utf-8")
        assert '<a href="rom24.svg">SVG Map</a>' in content_single_svg
        assert "N/A" not in content_single_svg

    def test_render_area_success(self, tmp_path):
        from scripts.build_pages import render_area
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "dialects" / "rom24.are"
        render_area("rom24", fixture_path, is_dir=False, outdir=tmp_path)
        assert (tmp_path / "rom24.html").exists()
        assert (tmp_path / "rom240.svg").exists()
        validate_html_viewer_file(tmp_path / "rom24.html")
        validate_svg_file(tmp_path / "rom240.svg")

    def test_render_area_failure_raises(self, tmp_path):
        from scripts.build_pages import render_area
        nonexistent = tmp_path / "nonexistent.are"
        with pytest.raises(RuntimeError, match="Failed to render"):
            render_area("nonexistent", nonexistent, is_dir=False, outdir=tmp_path)

        nonexistent_dir = tmp_path / "nonexistent_dir"
        with pytest.raises(RuntimeError, match="Failed to render"):
            render_area("nonexistent_dir", nonexistent_dir, is_dir=True, outdir=tmp_path)

        # Test failure during SVG rendering stage
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Simulated SVG generation failure"),
            ]
            with pytest.raises(RuntimeError, match="Failed to render SVG"):
                render_area("mock_area", tmp_path / "mock.are", is_dir=False, outdir=tmp_path)

    def test_build_pages_main_e2e(self, tmp_path):
        from scripts.build_pages import main

        fixture_area = REPO_ROOT / "tests" / "fixtures" / "dialects" / "rom24.are"
        mock_candidates = [("fixture", "rom24", fixture_area, False)]

        with patch("scripts.build_pages.resolve_candidate_areas", return_value=mock_candidates):
            with patch("sys.argv", ["build_pages.py", "--outdir", str(tmp_path)]):
                main()

        index_file = tmp_path / "index.html"
        assert index_file.exists()
        content = index_file.read_text(encoding="utf-8")
        assert "ROMUtil Map Directory" in content
        assert "rom24" in content
        assert '<a href="rom24.html">Interactive Viewer</a>' in content
        assert '<a href="rom240.svg">SVG Map</a>' in content
        assert "N/A" not in content
        assert (tmp_path / "rom24.html").exists()
        assert (tmp_path / "rom240.svg").exists()


class TestSvgAssetIntegrity:
    """Verifies XML well-formedness, root elements, and visual shapes in SVG outputs."""

    def test_generated_svgs_parseable_xml(self, sample_gallery_artifacts):
        svg_files = list(sample_gallery_artifacts.glob("*.svg"))
        assert len(svg_files) >= 1
        for svg_file in svg_files:
            root = validate_svg_file(svg_file)
            tag_clean = root.tag.split("}")[-1] if "}" in root.tag else root.tag
            assert tag_clean == "svg", f"Root tag of {svg_file.name} is {root.tag}"

    def test_map_svg_contains_visual_elements(self, sample_gallery_artifacts):
        for svg_file in sample_gallery_artifacts.glob("*.svg"):
            root = validate_svg_file(svg_file)
            elements = list(root.iter())
            tag_names = {el.tag.split("}")[-1] for el in elements}
            assert any(t in tag_names for t in ("polygon", "rect", "path", "g", "line")), (
                f"{svg_file.name} contains no graphical layout elements"
            )


class TestHtmlAssetIntegrity:
    """Verifies standalone HTML application requirements and zero external dependencies."""

    def test_generated_html_viewers_valid_structure(self, sample_gallery_artifacts):
        html_files = list(sample_gallery_artifacts.glob("*.html"))
        assert len(html_files) >= 1
        for html_path in html_files:
            data = validate_html_viewer_file(html_path)
            assert "rooms" in data
            assert isinstance(data["rooms"], list)
            assert len(data["rooms"]) > 0

    def test_html_zero_external_dependencies(self, sample_gallery_artifacts):
        for html_file in sample_gallery_artifacts.glob("*.html"):
            validate_html_viewer_file(html_file)




class TestNegativeAndBoundaryCases:
    """Negative tests for asset validation, corruption detection, and broken link handling."""

    def test_corrupt_svg_raises_parse_error(self):
        corrupt_xml = "<svg><polygon points='1,2 3,4'><invalid"
        with pytest.raises(ET.ParseError):
            validate_svg_content(corrupt_xml)

    def test_svg_missing_svg_root_tag_rejected(self):
        non_svg_xml = "<root><child value='1'/></root>"
        with pytest.raises(ValueError, match="expected <svg>"):
            validate_svg_content(non_svg_xml)

    def test_html_missing_doctype_rejected(self):
        bad_html = "<html><body><svg id='map-svg'></svg><canvas id='minimap'></canvas></body></html>"
        with pytest.raises(ValueError, match="Missing <!DOCTYPE html>"):
            validate_html_viewer_content(bad_html)

    def test_html_missing_map_svg_rejected(self):
        bad_html = "<!DOCTYPE html><html><body><canvas id='minimap'></canvas></body></html>"
        with pytest.raises(ValueError, match="Missing SVG map element"):
            validate_html_viewer_content(bad_html)

    def test_html_missing_minimap_canvas_rejected(self):
        bad_html = "<!DOCTYPE html><html><body><svg id='map-svg'></svg></body></html>"
        with pytest.raises(ValueError, match="Missing minimap canvas element"):
            validate_html_viewer_content(bad_html)

    def test_html_missing_json_payload_rejected(self):
        bad_html = (
            "<!DOCTYPE html><html><body>"
            "<svg id='map-svg'></svg><canvas id='minimap'></canvas>"
            "</body></html>"
        )
        with pytest.raises(ValueError, match="Missing embedded romutil-data"):
            validate_html_viewer_content(bad_html)

    def test_html_corrupt_json_payload_rejected(self):
        bad_html = (
            "<!DOCTYPE html><html><body>"
            "<svg id='map-svg'></svg><canvas id='minimap'></canvas>"
            '<script id="romutil-data" type="application/json">{invalid json}</script>'
            "</body></html>"
        )
        with pytest.raises(json.JSONDecodeError):
            validate_html_viewer_content(bad_html)

    def test_html_with_forbidden_external_script_detected(self):
        bad_html = (
            "<!DOCTYPE html><html><head>"
            '<script src="https://cdn.example.com/malicious.js"></script>'
            "</head><body><svg id='map-svg'></svg><canvas id='minimap'></canvas>"
            '<script id="romutil-data" type="application/json">{"area":{},"bounds":{},"rooms":[]}</script>'
            "</body></html>"
        )
        with pytest.raises(ValueError, match="Forbidden external script"):
            validate_html_viewer_content(bad_html)

    def test_html_with_forbidden_external_stylesheet_detected(self):
        bad_html = (
            "<!DOCTYPE html><html><head>"
            '<link rel="stylesheet" href="https://fonts.example.com/roboto.css">'
            "</head><body><svg id='map-svg'></svg><canvas id='minimap'></canvas>"
            '<script id="romutil-data" type="application/json">{"area":{},"bounds":{},"rooms":[]}</script>'
            "</body></html>"
        )
        with pytest.raises(ValueError, match="Forbidden external stylesheet"):
            validate_html_viewer_content(bad_html)


    def test_nonexistent_file_path_raises_file_not_found(self, tmp_path):
        nonexistent = tmp_path / "phantom.svg"
        with pytest.raises(FileNotFoundError):
            validate_svg_file(nonexistent)

        with pytest.raises(FileNotFoundError):
            validate_html_viewer_file(tmp_path / "phantom.html")
