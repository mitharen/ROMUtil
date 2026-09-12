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
    fixture_area = REPO_ROOT / "tests" / "fixtures" / "areas" / "smurf.are"
    outbase = out_dir / "smurf"

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


@pytest.mark.slow
@pytest.mark.integration
class TestGalleryAssetsGeneration:
    """Verifies that map and viewer artifacts are dynamically generated with correct outputs."""

    def test_dynamic_generation_creates_expected_files(self, sample_gallery_artifacts):
        assert (sample_gallery_artifacts / "smurf_z0.svg").exists()
        assert (sample_gallery_artifacts / "smurf.html").exists()

    def test_pages_builder_discovery_and_index_generation(self, tmp_path):
        from scripts.build_pages import resolve_candidate_areas, generate_index_html
        candidates = resolve_candidate_areas()
        assert len(candidates) == 7
        candidate_names = {c[1] for c in candidates}
        assert candidate_names == {
            "arachnos",
            "chapel",
            "midgaard",
            "midgaard_metropolitan",
            "school",
            "shire",
            "smurf",
        }

        # Verify individual showcase zones
        single_candidates = [c for c in candidates if c[0] == "ROM 2.4 / QuickMUD"]
        assert len(single_candidates) == 6
        for cat, name, path, is_dir in single_candidates:
            assert isinstance(path, Path)
            assert path.is_file()
            assert not is_dir

        # Verify composite showcase cluster
        comp_candidates = [c for c in candidates if c[0] == "ROM 2.4 / Composite"]
        assert len(comp_candidates) == 1
        cat, name, paths, is_dir = comp_candidates[0]
        assert name == "midgaard_metropolitan"
        assert isinstance(paths, list)
        assert len(paths) == 4
        assert [p.name for p in paths] == ["midgaard.are", "hood.are", "grave.are", "mobfact.are"]
        assert all(p.is_file() for p in paths)
        assert not is_dir

        dummy_rendered = [
            ("ROM 2.4 / QuickMUD", "smurf", 1.23),
            ("ROM 2.4 / Composite", "midgaard_metropolitan", 45.67),
        ]

        # Case 1: Asset files do not exist -> verify 'N/A' fallback is emitted
        generate_index_html(dummy_rendered, tmp_path)
        index_file = tmp_path / "index.html"
        assert index_file.exists()
        content_missing = index_file.read_text(encoding="utf-8")
        assert "ROMUtil Map Directory" in content_missing
        assert "<th>Solve Time</th>" in content_missing
        assert "ROM 2.4 / QuickMUD" in content_missing
        assert "ROM 2.4 / Composite" in content_missing
        assert "smurf" in content_missing
        assert "midgaard_metropolitan" in content_missing
        assert "<td><code>1.23s</code></td>" in content_missing
        assert "<td><code>45.67s</code></td>" in content_missing
        assert "N/A" in content_missing
        assert '<a href="smurf.html">' not in content_missing
        assert '<a href="midgaard_metropolitan.html">' not in content_missing

        # Case 2: Asset files exist -> verify valid <a href= links, solve times, and 'N/A' is absent
        (tmp_path / "smurf.html").write_text("<!DOCTYPE html><html><body>viewer</body></html>", encoding="utf-8")
        (tmp_path / "smurf0.svg").write_text("<svg></svg>", encoding="utf-8")
        (tmp_path / "midgaard_metropolitan.html").write_text("<!DOCTYPE html><html><body>viewer</body></html>", encoding="utf-8")
        (tmp_path / "midgaard_metropolitan0.svg").write_text("<svg></svg>", encoding="utf-8")
        generate_index_html(dummy_rendered, tmp_path)
        content_present = index_file.read_text(encoding="utf-8")
        assert "<th>Solve Time</th>" in content_present
        assert "<td><code>1.23s</code></td>" in content_present
        assert "<td><code>45.67s</code></td>" in content_present
        assert '<a href="smurf.html">Interactive Viewer</a>' in content_present
        assert '<a href="smurf0.svg">SVG Map</a>' in content_present
        assert '<a href="midgaard_metropolitan.html">Interactive Viewer</a>' in content_present
        assert '<a href="midgaard_metropolitan0.svg">SVG Map</a>' in content_present
        assert "N/A" not in content_present

        # Case 2b: Fallback to <name>.svg when <name>0.svg is not present
        (tmp_path / "smurf0.svg").unlink()
        (tmp_path / "smurf.svg").write_text("<svg></svg>", encoding="utf-8")
        generate_index_html(dummy_rendered, tmp_path)
        content_single_svg = index_file.read_text(encoding="utf-8")
        assert '<a href="smurf.svg">SVG Map</a>' in content_single_svg
        assert "N/A" not in content_single_svg

        # Case 3: Missing solve time emits 'N/A' in timing column
        dummy_no_time = [("ROM 2.4 / QuickMUD", "smurf")]
        generate_index_html(dummy_no_time, tmp_path)
        content_no_time = index_file.read_text(encoding="utf-8")
        assert "<td>N/A</td>" in content_no_time

    def test_showcase_area_discovery_retires_dialect_fixtures(self):
        from scripts.build_pages import resolve_candidate_areas
        candidates = resolve_candidate_areas()
        discovered_names = {c[1] for c in candidates}
        retired_fixtures = {"ackmud", "anatolia", "circlemud", "circle_world", "envy20", "merc22", "rom24", "smaug"}
        assert discovered_names.isdisjoint(retired_fixtures)
        for cat, name, path, is_dir in candidates:
            assert cat in ("ROM 2.4 / QuickMUD", "ROM 2.4 / Composite")
            if isinstance(path, list):
                assert all(p.is_file() for p in path)
            else:
                assert path.is_file()
            assert not is_dir

    def test_curated_showcase_areas_parsing_and_integrity(self):
        from romutil.parser import parse_file
        areas_dir = REPO_ROOT / "tests" / "fixtures" / "areas"
        expected_counts = {
            "arachnos.are": 56,
            "chapel.are": 67,
            "midgaard.are": 143,
            "school.are": 59,
            "shire.are": 58,
            "smurf.are": 29,
        }
        for filename, min_rooms in expected_counts.items():
            area_path = areas_dir / filename
            assert area_path.is_file(), f"Missing bundled showcase area: {filename}"
            area = parse_file(str(area_path))
            assert len(area.rooms) >= min_rooms, f"{filename} has {len(area.rooms)} rooms, expected >={min_rooms}"

    def test_quickmud_env_override(self, tmp_path, monkeypatch):
        from scripts.build_pages import resolve_candidate_areas
        custom_dir = tmp_path / "custom_qm"
        custom_dir.mkdir()
        smurf_src = REPO_ROOT / "tests" / "fixtures" / "areas" / "smurf.are"
        (custom_dir / "smurf.are").write_text(smurf_src.read_text(encoding="utf-8"), encoding="utf-8")

        monkeypatch.setenv("QUICKMUD_AREA_DIR", str(custom_dir))
        candidates = resolve_candidate_areas()
        assert any(c[1] == "smurf" and c[2].parent == custom_dir for c in candidates)

    def test_render_area_success(self, tmp_path):
        from scripts.build_pages import render_area
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "smurf.are"
        elapsed = render_area("smurf", fixture_path, is_dir=False, outdir=tmp_path)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0
        assert (tmp_path / "smurf.html").exists()
        assert (tmp_path / "smurf0.svg").exists()
        validate_html_viewer_file(tmp_path / "smurf.html")
        validate_svg_file(tmp_path / "smurf0.svg")

    def test_render_area_failure_raises(self, tmp_path):
        from scripts.build_pages import render_area
        nonexistent = tmp_path / "nonexistent.are"
        with pytest.raises(RuntimeError, match="Failed to render"):
            render_area("nonexistent", nonexistent, is_dir=False, outdir=tmp_path)

        nonexistent_dir = tmp_path / "nonexistent_dir"
        with pytest.raises(RuntimeError, match="Failed to render"):
            render_area("nonexistent_dir", nonexistent_dir, is_dir=True, outdir=tmp_path)

        # Test failure during rendering execution
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Simulated generation failure"
            )
            with pytest.raises(RuntimeError, match="Failed to render"):
                render_area("mock_area", tmp_path / "mock.are", is_dir=False, outdir=tmp_path)

    def test_render_area_invokes_cli_single_solve_with_timeout(self, tmp_path):
        from scripts.build_pages import render_area
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            elapsed = render_area("test_area", tmp_path / "test.are", is_dir=False, outdir=tmp_path)
            assert isinstance(elapsed, float)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "--format" in cmd
            fmt_idx = cmd.index("--format")
            assert cmd[fmt_idx + 1] == "html,svg"
            assert "--solver-timeout" in cmd
            timeout_idx = cmd.index("--solver-timeout")
            assert cmd[timeout_idx + 1] == "180"

    def test_render_area_circle_dir_invokes_cli_with_timeout(self, tmp_path):
        from scripts.build_pages import render_area
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            render_area("circle_area", tmp_path / "circle_dir", is_dir=True, outdir=tmp_path, solver_timeout=42)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "--circle-dir" in cmd
            assert "--format" in cmd
            fmt_idx = cmd.index("--format")
            assert cmd[fmt_idx + 1] == "html,svg"
            assert "--solver-timeout" in cmd
            timeout_idx = cmd.index("--solver-timeout")
            assert cmd[timeout_idx + 1] == "42"

    def test_render_area_multiple_source_files_invokes_cli(self, tmp_path):
        from scripts.build_pages import render_area
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            source_paths = [
                tmp_path / "midgaard.are",
                tmp_path / "hood.are",
                tmp_path / "grave.are",
                tmp_path / "mobfact.are",
            ]
            render_area(
                "midgaard_metropolitan",
                source_paths,
                is_dir=False,
                outdir=tmp_path,
                solver_timeout=90,
            )

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            for p in source_paths:
                assert str(p) in cmd
            p_indices = [cmd.index(str(p)) for p in source_paths]
            assert p_indices == sorted(p_indices)
            assert "-outbase" in cmd
            assert "--format" in cmd
            fmt_idx = cmd.index("--format")
            assert cmd[fmt_idx + 1] == "html,svg"
            assert "--solver-timeout" in cmd
            timeout_idx = cmd.index("--solver-timeout")
            assert cmd[timeout_idx + 1] == "90"

    def test_render_area_multiple_source_files_execution(self, tmp_path):
        from scripts.build_pages import render_area
        grave_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "grave.are"
        mobfact_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "mobfact.are"
        render_area("test_composite", [grave_path, mobfact_path], is_dir=False, outdir=tmp_path, solver_timeout=15)
        assert (tmp_path / "test_composite.html").exists()
        assert (tmp_path / "test_composite0.svg").exists()
        html_data = validate_html_viewer_file(tmp_path / "test_composite.html")
        validate_svg_file(tmp_path / "test_composite0.svg")
        room_vnums = {r["vnum"] for r in html_data["rooms"]}
        assert any(3600 <= v <= 3699 for v in room_vnums), "Graveyard rooms missing"
        assert any(9400 <= v <= 9499 for v in room_vnums), "Mob Factory rooms missing"

    def test_composite_viewer_multi_area_provenance_and_inspector(self, tmp_path):
        """Verify multi-area provenance serialization, zone controls, and inspector attribution (Task 10c)."""
        from scripts.build_pages import render_area

        grave_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "grave.are"
        mobfact_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "mobfact.are"
        render_area("test_prov", [grave_path, mobfact_path], is_dir=False, outdir=tmp_path, solver_timeout=15)

        html_path = tmp_path / "test_prov.html"
        assert html_path.exists()
        html_content = html_path.read_text(encoding="utf-8")
        html_data = validate_html_viewer_content(html_content)

        # 1. Verify JSON data contains area_name and area_file provenance for every room
        rooms = html_data["rooms"]
        assert len(rooms) > 0
        area_names = {r.get("area_name") for r in rooms}
        area_files = {r.get("area_file") for r in rooms}
        assert "Graveyard" in area_names
        assert "Mob Factory" in area_names
        assert "grave.are" in area_files
        assert "mobfact.are" in area_files

        # 2. Verify UI elements for zone filtering & legend
        assert 'id="zone-selector"' in html_content
        assert 'id="card-zone-legend"' in html_content
        assert 'id="btn-color-mode"' in html_content
        assert 'id="card-room-area-row"' in html_content
        assert 'id="card-room-area"' in html_content
        assert "function showRoomDetails(room)" in html_content
        assert "getZoneColor" in html_content
        assert "toggleZone" in html_content

    def test_build_pages_composite_area_e2e(self, tmp_path):
        from scripts.build_pages import main

        grave_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "grave.are"
        mobfact_path = REPO_ROOT / "tests" / "fixtures" / "areas" / "mobfact.are"
        mock_candidates = [
            ("ROM 2.4 / Composite", "test_composite", [grave_path, mobfact_path], False)
        ]

        with patch("scripts.build_pages.resolve_candidate_areas", return_value=mock_candidates):
            with patch("sys.argv", ["build_pages.py", "--outdir", str(tmp_path), "--solver-timeout", "15"]):
                main()

        index_file = tmp_path / "index.html"
        assert index_file.exists()
        content = index_file.read_text(encoding="utf-8")
        assert "ROMUtil Map Directory" in content
        assert "ROM 2.4 / Composite" in content
        assert "test_composite" in content
        assert "<th>Solve Time</th>" in content
        assert re.search(r"<td><code>\d+\.\d{2}s</code></td>", content)
        assert '<a href="test_composite.html">Interactive Viewer</a>' in content
        assert '<a href="test_composite0.svg">SVG Map</a>' in content
        assert "N/A" not in content

        assert (tmp_path / "test_composite.html").exists()
        assert (tmp_path / "test_composite0.svg").exists()
        html_data = validate_html_viewer_file(tmp_path / "test_composite.html")
        validate_svg_file(tmp_path / "test_composite0.svg")
        assert len(html_data["rooms"]) > 0

    def test_composite_showcase_constituents_parsing_and_integrity(self):
        from romutil.parser import parse_file
        areas_dir = REPO_ROOT / "tests" / "fixtures" / "areas"
        expected_counts = {
            "midgaard.are": 143,
            "hood.are": 70,
            "grave.are": 30,
            "mobfact.are": 25,
        }
        for filename, min_rooms in expected_counts.items():
            area_path = areas_dir / filename
            assert area_path.is_file(), f"Missing bundled composite constituent: {filename}"
            area = parse_file(str(area_path))
            assert len(area.rooms) >= min_rooms, f"{filename} has {len(area.rooms)} rooms, expected >={min_rooms}"

    def test_build_pages_main_e2e(self, tmp_path):
        from scripts.build_pages import main

        fixture_area = REPO_ROOT / "tests" / "fixtures" / "areas" / "smurf.are"
        mock_candidates = [("ROM 2.4 / QuickMUD", "smurf", fixture_area, False)]

        with patch("scripts.build_pages.resolve_candidate_areas", return_value=mock_candidates):
            with patch("sys.argv", ["build_pages.py", "--outdir", str(tmp_path)]):
                main()

        index_file = tmp_path / "index.html"
        assert index_file.exists()
        content = index_file.read_text(encoding="utf-8")
        assert "ROMUtil Map Directory" in content
        assert "ROM 2.4 / QuickMUD" in content
        assert "smurf" in content
        assert "<th>Solve Time</th>" in content
        assert re.search(r"<td><code>\d+\.\d{2}s</code></td>", content)
        assert '<a href="smurf.html">Interactive Viewer</a>' in content
        assert '<a href="smurf0.svg">SVG Map</a>' in content
        assert "N/A" not in content
        assert (tmp_path / "smurf.html").exists()
        assert (tmp_path / "smurf0.svg").exists()



    def test_candidate_area_tuple_compatibility_and_timeout(self):
        from scripts.build_pages import CandidateArea
        cand = CandidateArea("ROM 2.4 / Composite", "test_slug", [Path("a.are")], False, solver_timeout=300)
        assert len(cand) == 4
        cat, name, paths, is_dir = cand
        assert cat == "ROM 2.4 / Composite"
        assert name == "test_slug"
        assert paths == [Path("a.are")]
        assert is_dir is False
        assert cand.solver_timeout == 300

    def test_build_pages_cli_timeout_and_composite_timeout(self, tmp_path):
        from scripts.build_pages import main, CandidateArea

        cand_single = CandidateArea("ROM 2.4 / QuickMUD", "smurf", Path("smurf.are"), False)
        cand_comp = CandidateArea("ROM 2.4 / Composite", "midgaard_metropolitan", [Path("m.are"), Path("h.are")], False)
        cand_custom = CandidateArea("ROM 2.4 / Composite", "custom_cluster", [Path("c.are")], False, solver_timeout=250)

        mock_candidates = [cand_single, cand_comp, cand_custom]

        # Case 1: CLI with --composite-solver-timeout overrides composite timeouts
        with patch("scripts.build_pages.resolve_candidate_areas", return_value=mock_candidates):
            with patch("scripts.build_pages.render_area", return_value=1.5) as mock_render:
                with patch("scripts.build_pages.generate_index_html") as mock_gen_idx:
                    with patch("sys.argv", [
                        "build_pages.py",
                        "--outdir", str(tmp_path),
                        "--solver-timeout", "120",
                        "--composite-solver-timeout", "240",
                    ]):
                        main()

                    assert mock_render.call_count == 3
                    # single area uses --solver-timeout (120)
                    assert mock_render.call_args_list[0].kwargs["solver_timeout"] == 120
                    # composite area uses --composite-solver-timeout (240)
                    assert mock_render.call_args_list[1].kwargs["solver_timeout"] == 240
                    # custom cluster composite also overridden by CLI --composite-solver-timeout (240)
                    assert mock_render.call_args_list[2].kwargs["solver_timeout"] == 240

                    # Verify timing information passed to generate_index_html
                    mock_gen_idx.assert_called_once()
                    rendered_list = mock_gen_idx.call_args[0][0]
                    assert all(len(item) == 3 and item[2] == 1.5 for item in rendered_list)

        # Case 2: Without --composite-solver-timeout, candidate custom timeout is respected
        with patch("scripts.build_pages.resolve_candidate_areas", return_value=mock_candidates):
            with patch("scripts.build_pages.render_area", return_value=2.0) as mock_render:
                with patch("scripts.build_pages.generate_index_html"):
                    with patch("sys.argv", [
                        "build_pages.py",
                        "--outdir", str(tmp_path),
                    ]):
                        main()

                    assert mock_render.call_count == 3
                    # single area defaults to 180
                    assert mock_render.call_args_list[0].kwargs["solver_timeout"] == 180
                    # composite area without candidate timeout defaults to 180
                    assert mock_render.call_args_list[1].kwargs["solver_timeout"] == 180
                    # custom cluster uses its candidate-level timeout (250)
                    assert mock_render.call_args_list[2].kwargs["solver_timeout"] == 250

    def test_build_pages_5tuple_candidate_timeout(self, tmp_path):
        from scripts.build_pages import main
        cand_5tuple = ("ROM 2.4 / Composite", "five_tuple_comp", [Path("a.are")], False, 210)
        with patch("scripts.build_pages.resolve_candidate_areas", return_value=[cand_5tuple]):
            with patch("scripts.build_pages.render_area", return_value=3.14) as mock_render:
                with patch("scripts.build_pages.generate_index_html"):
                    with patch("sys.argv", ["build_pages.py", "--outdir", str(tmp_path)]):
                        main()
                    assert mock_render.call_args_list[0].kwargs["solver_timeout"] == 210


@pytest.mark.slow
@pytest.mark.integration
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


@pytest.mark.slow
@pytest.mark.integration
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

    def test_html_viewer_interactive_ux_enhancements(self, sample_gallery_artifacts):
        """Verify multi-floor controls, collapsible sidebar, and minimap drag navigation (Task 9h)."""
        html_files = list(sample_gallery_artifacts.glob("*.html"))
        assert len(html_files) >= 1
        for html_file in html_files:
            content = html_file.read_text(encoding="utf-8")

            # 1. Multi-floor selection controls
            assert 'id="floor-selector"' in content
            assert 'id="floor-pill-list"' in content
            assert 'id="btn-floor-all"' in content
            assert "floor-pill" in content
            assert "toggleFloor" in content
            assert "toggleAllFloors" in content
            assert "activeFloors.has(srcZ) || activeFloors.has(dstZ)" in content

            # 2. Collapsible sidebar
            assert 'id="btn-collapse-sidebar"' in content
            assert 'id="btn-expand-sidebar"' in content
            assert 'id="btn-toggle-sidebar"' in content
            assert "aside.sidebar.collapsed" in content
            assert "#app.collapsed aside.sidebar" in content
            assert "#app.collapsed .minimap-container" in content
            assert "setSidebarCollapsed" in content

            # 3. Minimap drag navigation
            assert "panToMinimapCoord" in content
            assert "minimap.addEventListener('mousedown'" in content
            assert "minimap.addEventListener('mousemove'" in content
            assert "minimap.addEventListener('mouseup'" in content
            assert "minimap.addEventListener('mouseleave'" in content
            assert "cursor: grab;" in content
            assert "cursor: grabbing;" in content


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
