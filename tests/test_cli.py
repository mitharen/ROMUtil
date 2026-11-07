"""Integration and unit tests for romutil.cli and HierarchicalLayoutSolver production pipeline."""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import pytest

from romutil.cli import cli, main
from romutil.layout import (
    ChainedSimplifier,
    HallwayCompactor,
    HierarchicalLayoutAssembler,
    HierarchicalLayoutSolver,
    HierarchicalTreeSolver,
    PassthroughSimplifier,
    PipelineTreeBuilder,
    SpqrTreeBuilder,
    WarpSeverer,
    production,
)
from romutil.models import (
    AreaData,
    Room,
)
from romutil.parser import Parser
from romutil.renderers import RENDERERS
from romutil.renderers.base import BaseRenderer
from romutil.types import Coord3D, LayoutOptions, SolvedLayout, SolverStatus, WorldGraph

AREAS_DIR = Path(__file__).parent / "fixtures" / "areas"
DIALECTS_DIR = Path(__file__).parent / "fixtures" / "dialects"


class TestHierarchicalLayoutSolverProductionFactory:
    """Tests for the HierarchicalLayoutSolver.production() factory method."""

    def test_production_default_stages(self) -> None:
        """Verify production() constructs standard 5-stage pipeline."""
        solver = HierarchicalLayoutSolver.production()
        assert isinstance(solver, HierarchicalLayoutSolver)
        assert isinstance(solver.simplifier, ChainedSimplifier)
        assert len(solver.simplifier.simplifiers) == 2
        assert isinstance(solver.simplifier.simplifiers[0], HallwayCompactor)
        assert isinstance(solver.simplifier.simplifiers[1], WarpSeverer)
        assert isinstance(solver.tree_builder, PipelineTreeBuilder)
        assert isinstance(solver.tree_solver, HierarchicalTreeSolver)
        assert isinstance(solver.assembler, HierarchicalLayoutAssembler)

    def test_production_convenience_export(self) -> None:
        """Verify production factory exported at package level romutil.layout."""
        solver = production()
        assert isinstance(solver, HierarchicalLayoutSolver)
        assert isinstance(solver.simplifier, ChainedSimplifier)
        assert isinstance(solver.tree_builder, SpqrTreeBuilder)

    def test_production_custom_overrides(self) -> None:
        """Verify production() respects custom stage overrides."""
        custom_simplifier = PassthroughSimplifier()
        solver = HierarchicalLayoutSolver.production(simplifier=custom_simplifier)
        assert solver.simplifier is custom_simplifier
        assert isinstance(solver.tree_builder, SpqrTreeBuilder)
        assert isinstance(solver.tree_solver, HierarchicalTreeSolver)
        assert isinstance(solver.assembler, HierarchicalLayoutAssembler)


class TestCliPipelineEndToEnd:
    """End-to-end integration tests running CLI main() with real area fixtures."""

    @pytest.mark.parametrize(
        ("fixture_path", "expected_room_count"),
        [
            (AREAS_DIR / "school.are", 59),
            (DIALECTS_DIR / "rom24.are", 5),
            (AREAS_DIR / "arachnos.are", 56),
        ],
    )
    def test_cli_end_to_end_formats(
        self, tmp_path: Path, fixture_path: Path, expected_room_count: int
    ) -> None:
        """Test CLI main() with school.are, rom24.are, arachnos.are generating SVG, HTML, and JSON."""
        outbase = str(tmp_path / fixture_path.stem)

        with pytest.raises(SystemExit) as exc:
            main([fixture_path], outbase=outbase, fmt="svg,html,json")
        assert exc.value.code == 0

        # Verify SVG output
        svg_file = tmp_path / f"{fixture_path.stem}.svg"
        assert svg_file.exists(), f"Expected {svg_file} to exist"
        svg_text = svg_file.read_text(encoding="utf-8")
        assert "<svg" in svg_text and "</svg>" in svg_text
        assert "viewBox" in svg_text

        # Verify HTML output
        html_file = tmp_path / f"{fixture_path.stem}.html"
        assert html_file.exists(), f"Expected {html_file} to exist"
        html_text = html_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_text or "<html" in html_text
        assert "rooms" in html_text

        # Verify JSON output
        json_file = tmp_path / f"{fixture_path.stem}.json"
        assert json_file.exists(), f"Expected {json_file} to exist"
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "rooms" in data
        assert len(data["rooms"]) == expected_room_count

    def test_cli_split_levels(self, tmp_path: Path) -> None:
        """Verify CLI produces elevation-specific SVGs with --split-levels."""
        fixture_path = AREAS_DIR / "school.are"
        outbase = str(tmp_path / "school_split")

        with pytest.raises(SystemExit) as exc:
            main([fixture_path], outbase=outbase, split_levels=True, fmt="svg")
        assert exc.value.code == 0

        # With split_levels=True, level-specific SVGs are produced
        split_svgs = list(tmp_path.glob("school_split*.svg"))
        assert len(split_svgs) >= 1

    def test_cli_pipeline_execution(self, tmp_path: Path) -> None:
        """Verify CLI executes the canonical pipeline cleanly."""
        fixture_path = DIALECTS_DIR / "rom24.are"
        outbase = str(tmp_path / "rom24_canonical")

        with pytest.raises(SystemExit) as exc:
            main([fixture_path], outbase=outbase, fmt="json")
        assert exc.value.code == 0
        assert (tmp_path / "rom24_canonical.json").exists()

    def test_cli_solver_timeout_option(self, tmp_path: Path) -> None:
        """Verify CLI main accepts custom solver timeout."""
        fixture_path = DIALECTS_DIR / "rom24.are"
        outbase = str(tmp_path / "rom24_timeout")

        with pytest.raises(SystemExit) as exc:
            main([fixture_path], outbase=outbase, fmt="json", solver_timeout=30)
        assert exc.value.code == 0
        assert (tmp_path / "rom24_timeout.json").exists()


class TestSolvedLayoutCoordinates:
    """Verify production solver SolvedLayout geometry, non-overlapping constraints, and validity."""

    def test_rom24_coordinates_strictly_non_overlapping(self) -> None:
        """rom24.are solves with strictly non-overlapping valid integer coordinates."""
        area = Parser().parse((DIALECTS_DIR / "rom24.are").read_text(encoding="latin-1"))
        world_graph = WorldGraph.from_area_data(area)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph, LayoutOptions(timeout_seconds=30))

        assert isinstance(layout, SolvedLayout)
        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(layout.coordinates) == len(world_graph.rooms) == 5

        # Strictly non-overlapping coordinates
        coords_list = list(layout.coordinates.values())
        unique_coords = set(coords_list)
        assert len(unique_coords) == len(coords_list) == 5

        # Valid coordinates normalized to origin
        for c in coords_list:
            assert isinstance(c, Coord3D)
            assert isinstance(c.x, int) and isinstance(c.y, int) and isinstance(c.z, int)
            assert c.x >= 0 and c.y >= 0 and c.z >= 0

        assert min(c.x for c in coords_list) == 0
        assert min(c.y for c in coords_list) == 0
        assert min(c.z for c in coords_list) == 0

    def test_arachnos_coordinates_strictly_non_overlapping(self) -> None:
        """arachnos.are solves with strictly non-overlapping valid integer coordinates."""
        area = Parser().parse((AREAS_DIR / "arachnos.are").read_text(encoding="latin-1"))
        world_graph = WorldGraph.from_area_data(area)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph, LayoutOptions(timeout_seconds=60))

        assert isinstance(layout, SolvedLayout)
        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(layout.coordinates) == len(world_graph.rooms) == 56

        # Strictly non-overlapping coordinates across all 56 rooms
        coords_list = list(layout.coordinates.values())
        unique_coords = set(coords_list)
        assert len(unique_coords) == len(coords_list) == 56

        # Valid coordinates normalized to origin
        for c in coords_list:
            assert isinstance(c, Coord3D)
            assert isinstance(c.x, int) and isinstance(c.y, int) and isinstance(c.z, int)
            assert c.x >= 0 and c.y >= 0 and c.z >= 0

        assert min(c.x for c in coords_list) == 0
        assert min(c.y for c in coords_list) == 0
        assert min(c.z for c in coords_list) == 0

    def test_school_coordinates_validity(self) -> None:
        """school.are solves with complete room placement and valid integer coordinates."""
        area = Parser().parse((AREAS_DIR / "school.are").read_text(encoding="latin-1"))
        world_graph = WorldGraph.from_area_data(area)

        solver = HierarchicalLayoutSolver.production()
        layout = solver.solve(world_graph, LayoutOptions(timeout_seconds=60))

        assert isinstance(layout, SolvedLayout)
        assert layout.diagnostics.status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL)
        assert len(layout.coordinates) == len(world_graph.rooms) == 59

        # Coordinates are valid integers normalized to origin
        coords_list = list(layout.coordinates.values())
        for c in coords_list:
            assert isinstance(c, Coord3D)
            assert isinstance(c.x, int) and isinstance(c.y, int) and isinstance(c.z, int)
            assert c.x >= 0 and c.y >= 0 and c.z >= 0

        assert min(c.x for c in coords_list) == 0
        assert min(c.y for c in coords_list) == 0
        assert min(c.z for c in coords_list) == 0

        # Highly separated (at least 58 distinct locations for 59 rooms)
        unique_coords = set(coords_list)
        assert len(unique_coords) >= 58


class TestCliCoverage:
    """Targeted coverage tests for romutil.cli branches."""

    def test_main_fmt_types(self, tmp_path: Path):
        are_file = DIALECTS_DIR / "rom24.are"
        out = str(tmp_path / "fmt_test")

        # fmt as list
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, fmt=["svg"])
        assert exc.value.code == 0

        # fmt as tuple
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, fmt=("svg",))
        assert exc.value.code == 0

        # fmt as set
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, fmt={"svg"})
        assert exc.value.code == 0

        # invalid fmt type
        with pytest.raises(TypeError, match="Expected format string or sequence"):
            main([are_file], out, fmt=999)

        # empty format string
        with pytest.raises(ValueError, match="No formats specified"):
            main([are_file], out, fmt="")

        # empty format list
        with pytest.raises(ValueError, match="No formats specified"):
            main([are_file], out, fmt=[])

        # unsupported format
        with pytest.raises(ValueError, match="Unsupported format 'unknown'"):
            main([are_file], out, fmt="unknown")

    def test_main_directory_input(self, tmp_path: Path):
        circle_dir = DIALECTS_DIR / "circle_world"
        out = str(tmp_path / "dir_test")
        with pytest.raises(SystemExit) as exc:
            main([circle_dir], out, fmt="json")
        assert exc.value.code == 0
        assert (tmp_path / "dir_test.json").exists()

    def test_main_file_stream_input(self, tmp_path: Path):
        are_file = DIALECTS_DIR / "rom24.are"
        content = are_file.read_text(encoding="latin-1")
        stream = io.StringIO(content)
        stream.name = "stream_test.are"
        out = str(tmp_path / "stream_out")

        with pytest.raises(SystemExit) as exc:
            main([stream], out, fmt="json")
        assert exc.value.code == 0
        assert (tmp_path / "stream_out.json").exists()

    def test_main_corrupt_content_graceful(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        corrupt = io.StringIO("BOGUS DATA WITHOUT ROOMS\n")
        corrupt.name = "corrupt.are"
        out = str(tmp_path / "corrupt_out")

        with pytest.raises(SystemExit) as exc:
            main([corrupt], out, fmt="svg")
        assert exc.value.code == 0
        assert "No rooms to plot" in caplog.text

    def test_main_parser_exception_handling(self, tmp_path: Path):
        are_file = DIALECTS_DIR / "rom24.are"
        out = str(tmp_path / "err_out")

        with patch("romutil.cli.Parser.parse", side_effect=ValueError("Simulated parse error")):
            with pytest.raises(SystemExit) as exc:
                main([are_file], out)
            assert exc.value.code == 0

    def test_main_parser_returns_non_area_data_raises(self, tmp_path: Path):
        are_file = DIALECTS_DIR / "rom24.are"
        out = str(tmp_path / "type_err_out")

        with patch("romutil.cli.Parser.parse", return_value="Not an AreaData"):
            with pytest.raises(TypeError, match="Expected AreaData from parser"):
                main([are_file], out)

    def test_main_circle_dir_non_area_data_raises(self, tmp_path: Path):
        circle_dir = DIALECTS_DIR / "circle_world"
        out = str(tmp_path / "circle_type_err")

        with patch("romutil.cli.parse_circlemud_directory", return_value="Not an AreaData"):
            with pytest.raises(TypeError, match="Expected AreaData from parser"):
                main([], out, circle_dir=circle_dir)

    def test_main_fallback_area_header(self, tmp_path: Path):
        r1 = Room(vnum=5, name="R5", description="Desc")
        area_no_hdr = AreaData(header=None, rooms=(r1,))
        synth = tmp_path / "synthetic.are"
        synth.write_text("dummy", encoding="latin-1")
        out = str(tmp_path / "fallback_out")

        with patch("romutil.cli.Parser.parse", return_value=area_no_hdr):
            with pytest.raises(SystemExit) as exc:
                main([synth], out, fmt="json")
            assert exc.value.code == 0
            assert (tmp_path / "fallback_out.json").exists()

    def test_main_split_levels(self, tmp_path: Path):
        are_file = DIALECTS_DIR / "rom24.are"
        out = str(tmp_path / "split_test")

        # split_levels=True
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, split_levels=True, fmt="svg")
        assert exc.value.code == 0

        # split_levels=False
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, split_levels=False, fmt="svg")
        assert exc.value.code == 0

    def test_main_custom_format_dispatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class MockCustom(BaseRenderer):
            def render(self, *args, **options):
                out = args[2] if len(args) > 2 else args[1]
                p = Path(out)
                p.write_text("custom-out", encoding="utf-8")
                return p

        monkeypatch.setitem(RENDERERS, "txt", MockCustom)
        are_file = DIALECTS_DIR / "rom24.are"
        out = str(tmp_path / "custom_test")

        with pytest.raises(SystemExit) as exc:
            main([are_file], out, fmt="txt")
        assert exc.value.code == 0
        assert (tmp_path / "custom_test.txt").read_text(encoding="utf-8") == "custom-out"

    def test_cli_branches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        are_file = DIALECTS_DIR / "rom24.are"
        out = str(tmp_path / "cli_out")

        # debug flag
        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "-outbase", out, "-d"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

        # no args error
        monkeypatch.setattr("sys.argv", ["romutil"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code != 0

        # empty format string error
        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "--format", "  ,  "])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code != 0

        # unsupported format error
        monkeypatch.setattr("sys.argv", ["romutil", str(are_file), "--format", "badfmt"])
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code != 0

        # solver-timeout flag
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", out, "--solver-timeout", "10"],
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0

        # debug-solve flag
        debug_dir = tmp_path / "cli_debug"
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", out, "--debug-solve", "--debug-solve-dir", str(debug_dir)],
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        assert debug_dir.is_dir()
        # Verify cluster svg files were generated
        svgs = list(debug_dir.glob("*.svg"))
        assert len(svgs) > 0

        # debug-solve flag with default dir
        monkeypatch.setattr(
            "sys.argv",
            ["romutil", str(are_file), "-outbase", out, "--debug-solve"],
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 0
        default_debug = Path("debug_solves")
        if default_debug.is_dir():
            shutil.rmtree(default_debug)
