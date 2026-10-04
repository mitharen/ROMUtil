"""Integration tests for the unified CLI pipeline and HierarchicalLayoutSolver production factory."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from romutil.cli import cli, main
from romutil.layout import (
    ChainedSimplifier,
    HallwayCompactor,
    HierarchicalLayoutAssembler,
    HierarchicalLayoutSolver,
    HierarchicalTreeSolver,
    PassthroughSimplifier,
    SpqrTreeBuilder,
    WarpSeverer,
    production,
)
from romutil.parser import Parser
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
        assert isinstance(solver.tree_builder, SpqrTreeBuilder)
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

        # Verify copyfile <outbase>0.svg for non-split SVG
        svg_0_file = tmp_path / f"{fixture_path.stem}0.svg"
        assert svg_0_file.exists()

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

    def test_cli_no_hierarchical_fallback(self, tmp_path: Path) -> None:
        """Verify CLI executes gracefully when hierarchical is disabled."""
        fixture_path = DIALECTS_DIR / "rom24.are"
        outbase = str(tmp_path / "rom24_flat")

        with pytest.raises(SystemExit) as exc:
            main([fixture_path], outbase=outbase, fmt="json", hierarchical=False)
        assert exc.value.code == 0
        assert (tmp_path / "rom24_flat.json").exists()

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
