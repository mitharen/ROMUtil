import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS_DIR = REPO_ROOT / "docs" / "assets"
README_FILE = REPO_ROOT / "README.md"
DESIGN_FILE = REPO_ROOT / "DESIGN.md"


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


def extract_markdown_relative_links(markdown_text: str):
    """Extracts all relative markdown file links, image embeds, and HTML tags.

    Yields tuples of (link_type, display_text, clean_target_path).
    """
    # 1. Standard Markdown links [text](target)
    for m in re.finditer(r"(?<!\!)\[([^\]]+)\]\(([^)#]+)(?:#[^)]*)?\)", markdown_text):
        target = m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        yield ("link", m.group(1), target)

    # 2. Markdown image embeds ![alt](target)
    for m in re.finditer(r"!\[([^\]]*)\]\(([^)#]+)(?:#[^)]*)?\)", markdown_text):
        target = m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        yield ("image", m.group(1), target)

    # 3. HTML <img> tags: <img src="target">
    for m in re.finditer(r"""<img\b[^>]*src=['"]([^'">#]+)['"][^>]*>""", markdown_text, re.IGNORECASE):
        target = m.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        yield ("html_img", "img", target)

    # 4. HTML <a> tags: <a href="target">
    for m in re.finditer(r"""<a\b[^>]*href=['"]([^'">#]+)['"][^>]*>""", markdown_text, re.IGNORECASE):
        target = m.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        yield ("html_a", "a", target)


def find_broken_relative_links(markdown_text: str, base_dir: Path):
    """Returns a list of broken relative targets that do not exist on disk."""
    broken = []
    for link_type, text, target in extract_markdown_relative_links(markdown_text):
        resolved = (base_dir / target).resolve()
        if not resolved.exists():
            broken.append((link_type, text, target, str(resolved)))
    return broken


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

class TestGalleryAssetsExist:
    """Verifies that all pre-rendered gallery assets are present on disk."""

    def test_docs_assets_directory_present(self):
        assert DOCS_ASSETS_DIR.exists(), f"Missing directory: {DOCS_ASSETS_DIR}"
        assert DOCS_ASSETS_DIR.is_dir()

    @pytest.mark.parametrize(
        "expected_file",
        [
            "school.svg",
            "school_z0.svg",
            "school_z1.svg",
            "smurf.svg",
            "demo_tower.svg",
            "pipeline_diagram.svg",
        ],
    )
    def test_canonical_svg_assets_exist(self, expected_file):
        asset_path = DOCS_ASSETS_DIR / expected_file
        assert asset_path.exists(), f"Expected SVG asset missing: {asset_path}"
        assert asset_path.is_file()
        assert asset_path.stat().st_size > 200, f"Asset file too small: {asset_path}"

    @pytest.mark.parametrize(
        "expected_html",
        [
            "school.html",
            "smurf.html",
            "demo_tower.html",
        ],
    )
    def test_canonical_html_viewers_exist(self, expected_html):
        viewer_path = DOCS_ASSETS_DIR / expected_html
        assert viewer_path.exists(), f"Expected HTML viewer missing: {viewer_path}"
        assert viewer_path.is_file()
        assert viewer_path.stat().st_size > 1000, f"HTML viewer file too small: {viewer_path}"


class TestSvgAssetIntegrity:
    """Verifies XML well-formedness, root elements, and visual shapes in SVG assets."""

    def test_all_gallery_svgs_parseable_xml(self):
        svg_files = list(DOCS_ASSETS_DIR.glob("*.svg"))
        assert len(svg_files) >= 6, f"Expected at least 6 SVG files, found {len(svg_files)}"
        for svg_file in svg_files:
            root = validate_svg_file(svg_file)
            tag_clean = root.tag.split("}")[-1] if "}" in root.tag else root.tag
            assert tag_clean == "svg", f"Root tag of {svg_file.name} is {root.tag}"

    def test_map_svg_contains_visual_elements(self):
        for map_name in ("school.svg", "smurf.svg", "demo_tower.svg"):
            root = validate_svg_file(DOCS_ASSETS_DIR / map_name)
            elements = list(root.iter())
            tag_names = {el.tag.split("}")[-1] for el in elements}
            assert any(t in tag_names for t in ("polygon", "rect", "path", "g", "line")), (
                f"{map_name} contains no graphical layout elements"
            )

    def test_split_level_svg_integrity(self):
        for z_name in ("school_z0.svg", "school_z1.svg"):
            root = validate_svg_file(DOCS_ASSETS_DIR / z_name)
            assert root is not None
            text_content = (DOCS_ASSETS_DIR / z_name).read_text(encoding="utf-8")
            assert "<svg" in text_content

    def test_pipeline_diagram_svg_content(self):
        diagram_path = DOCS_ASSETS_DIR / "pipeline_diagram.svg"
        root = validate_svg_file(diagram_path)
        content = diagram_path.read_text(encoding="utf-8")
        assert "ROMUtil" in content
        assert "MILP" in content
        assert "parser.py" in content
        assert "solver.py" in content
        assert "plotter.py" in content


class TestHtmlAssetIntegrity:
    """Verifies standalone HTML application requirements and zero external dependencies."""

    @pytest.mark.parametrize("html_name", ["school.html", "smurf.html", "demo_tower.html"])
    def test_gallery_html_viewers_valid_structure(self, html_name):
        html_path = DOCS_ASSETS_DIR / html_name
        data = validate_html_viewer_file(html_path)
        assert "rooms" in data
        assert isinstance(data["rooms"], list)
        assert len(data["rooms"]) > 0

    def test_school_html_data_fidelity(self):
        data = validate_html_viewer_file(DOCS_ASSETS_DIR / "school.html")
        assert len(data["rooms"]) == 59
        assert data["area"]["name"].lower() == "mud school"
        assert data["bounds"]["min_x"] == 0
        assert data["bounds"]["max_x"] > 0

    def test_demo_tower_html_data_fidelity(self):
        data = validate_html_viewer_file(DOCS_ASSETS_DIR / "demo_tower.html")
        assert len(data["rooms"]) == 4
        room_names = [r["name"] for r in data["rooms"]]
        assert "Tower Entrance" in room_names
        assert "Wizard Study" in room_names

    def test_gallery_html_zero_external_dependencies(self):
        for html_file in DOCS_ASSETS_DIR.glob("*.html"):
            validate_html_viewer_file(html_file)


class TestReadmeLinkAndVisualIntegrity:
    """Verifies all relative links and image paths in README.md resolve to existing files."""

    def test_readme_contains_gallery_and_badges(self):
        assert README_FILE.exists()
        content = README_FILE.read_text(encoding="utf-8")
        assert "## Visual Gallery & Interactive Previews" in content
        assert "pipeline_diagram.svg" in content
        assert "school.svg" in content
        assert "school.html" in content

    def test_readme_all_relative_links_and_images_resolve(self):
        content = README_FILE.read_text(encoding="utf-8")
        broken = find_broken_relative_links(content, REPO_ROOT)
        assert len(broken) == 0, f"Broken links found in README.md: {broken}"

        all_links = list(extract_markdown_relative_links(content))
        assert len(all_links) >= 10, f"Expected >= 10 relative links in README.md, found {len(all_links)}"

    def test_readme_embedded_images_resolve_to_assets(self):
        content = README_FILE.read_text(encoding="utf-8")
        img_targets = [
            target for _, _, target in extract_markdown_relative_links(content)
            if "assets" in target and target.endswith((".svg", ".html"))
        ]
        assert len(img_targets) >= 5, f"Expected >= 5 asset references, found {len(img_targets)}"
        for target in img_targets:
            resolved = (REPO_ROOT / target).resolve()
            assert resolved.exists(), f"Asset target does not exist: {target} (resolved: {resolved})"


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

    def test_broken_relative_link_detection(self, tmp_path):
        synthetic_md = """
# Test Markdown
Here is a valid link: [Valid File](valid.txt)
Here is a broken link: [Broken File](nonexistent_file.txt)
Here is an image link: ![Broken Image](missing_image.svg)
Here is an HTML tag: <img src="nonexistent_html_img.png" />
Here is an external link: [External](https://example.com/foo)
"""
        (tmp_path / "valid.txt").write_text("hello", encoding="utf-8")
        broken = find_broken_relative_links(synthetic_md, tmp_path)
        assert len(broken) == 3
        targets = {t[2] for t in broken}
        assert targets == {"nonexistent_file.txt", "missing_image.svg", "nonexistent_html_img.png"}

    def test_nonexistent_file_path_raises_file_not_found(self, tmp_path):
        nonexistent = tmp_path / "phantom.svg"
        with pytest.raises(FileNotFoundError):
            validate_svg_file(nonexistent)

        with pytest.raises(FileNotFoundError):
            validate_html_viewer_file(tmp_path / "phantom.html")
