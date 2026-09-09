import json
from pathlib import Path
import re
import subprocess
import sys
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
        generate_index_html(dummy_rendered, tmp_path)
        index_file = tmp_path / "index.html"
        assert index_file.exists()
        content = index_file.read_text(encoding="utf-8")
        assert "ROMUtil Map Directory" in content
        assert "rom24" in content


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
