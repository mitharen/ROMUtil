import json
import os
from pathlib import Path
import pytest
import sys

# Add scripts directory to sys.path for importing profile_solver
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import profile_solver
from profile_solver import (
    profile_area,
    find_area_file,
    format_profile_markdown_table,
    format_stage_breakdown_table,
    SolverProfileResult,
    IterationMetrics,
    StageTimings,
    main as profile_main,
)

from tests.conftest import SAMPLE_AREAS_DIR


class TestSolverProfilingUnit:
    """Unit tests for the solver profiling instrumentation and metrics collection."""

    def test_find_area_file_success(self):
        if not SAMPLE_AREAS_DIR:
            pytest.skip("QuickMUD area directory not found")
        path = find_area_file("smurf.are", search_dir=SAMPLE_AREAS_DIR)
        assert path.is_file()
        assert path.name == "smurf.are"

    def test_find_area_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            find_area_file("nonexistent_mystery_area.are", search_dir="/nonexistent/dir")

    def test_profile_empty_room_area(self, tmp_path):
        """Positive test for an area with no exits / trivial layout."""
        are_file = tmp_path / "empty_room.are"
        are_file.write_text(
            "#AREA\n"
            "empty.are~\n"
            "Empty Room Area~\n"
            "{ 1 10 } Builder Empty Area~\n"
            "1 1\n\n"
            "#ROOMS\n"
            "#1\n"
            "Solitary Room~\n"
            "A quiet room with no exits.~\n"
            "0 0 0\n"
            "S\n"
            "#0\n\n"
            "#$\n"
        )
        res = profile_area(are_file, cbc_sec_limit=10)
        assert isinstance(res, SolverProfileResult)
        assert res.original_rooms == 1
        assert res.original_exits == 0
        assert res.model_exits == 0
        assert res.success is True
        assert res.final_termination_condition == "trivial_empty"

        d = res.to_dict()
        assert "timings" in d
        assert "iterations" in d
        assert d["original_rooms"] == 1

    def test_profile_linear_corridor_area(self, tmp_path):
        """Positive test for corridor condensation and solve pipeline metrics."""
        are_file = tmp_path / "corridor.are"
        # 3 rooms in a straight line: 1 <-> 2 <-> 3
        # Room 2 should be condensed away
        are_file.write_text(
            "#AREA\n"
            "corridor.are~\n"
            "Corridor Area~\n"
            "{ 1 10 } Builder Corridor~\n"
            "1 3\n\n"
            "#ROOMS\n"
            "#1\n"
            "Room 1~\n"
            "West end.~\n"
            "0 0 0\n"
            "D1\n"
            "East door~\n"
            "~\n"
            "0 0 2\n"
            "S\n"
            "#2\n"
            "Room 2~\n"
            "Corridor hallway.~\n"
            "0 0 0\n"
            "D1\n"
            "East door~\n"
            "~\n"
            "0 0 3\n"
            "D3\n"
            "West door~\n"
            "~\n"
            "0 0 1\n"
            "S\n"
            "#3\n"
            "Room 3~\n"
            "East end.~\n"
            "0 0 0\n"
            "D3\n"
            "West door~\n"
            "~\n"
            "0 0 2\n"
            "S\n"
            "#0\n\n"
            "#$\n"
        )
        res = profile_area(are_file, cbc_sec_limit=15)
        assert isinstance(res, SolverProfileResult)
        assert res.original_rooms == 3
        assert res.condensed_rooms == 1
        assert res.active_rooms == 2
        assert res.timings.condensation_sec >= 0.0
        assert res.timings.initial_solve_sec > 0.0
        assert res.success is True
        assert res.final_termination_condition == "optimal"

    def test_profile_smurf_metrics(self):
        """Positive test running profile_area on smurf.are."""
        if not SAMPLE_AREAS_DIR:
            pytest.skip("QuickMUD area directory not found")
        smurf_path = Path(SAMPLE_AREAS_DIR) / "smurf.are"
        if not smurf_path.is_file():
            pytest.skip("smurf.are not present")

        res = profile_area(smurf_path, cbc_sec_limit=30, max_iterations=2)
        assert isinstance(res, SolverProfileResult)
        assert res.area_name == "Smurfville"
        assert res.original_rooms == 29
        assert res.original_exits == 63
        assert res.condensed_rooms == 6
        assert res.active_rooms == 23
        assert res.dummy_rooms == 1
        assert res.model_exits == 26
        assert res.initial_candidate_pairs == 279
        assert len(res.iterations) > 0
        assert res.timings.total_pipeline_sec > 0.0

        # Structured dict output check
        out_dict = res.to_dict()
        assert out_dict["area_name"] == "Smurfville"
        assert len(out_dict["iterations"]) == len(res.iterations)
        assert "parsing_sec" in out_dict["timings"]

    def test_format_markdown_tables(self):
        """Verify markdown table formatting utilities."""
        dummy_timings = StageTimings(
            parsing_sec=0.01,
            graph_construction_sec=0.005,
            condensation_sec=0.001,
            dummy_allocation_sec=0.001,
            non_euler_formulation_sec=0.01,
            non_euler_solve_sec=0.05,
            layout_formulation_sec=0.02,
            initial_solve_sec=0.10,
            overlap_detection_total_sec=0.03,
            collision_solves_total_sec=0.20,
            coordinate_restoration_sec=0.001,
            total_pipeline_sec=0.428,
        )
        dummy_res = SolverProfileResult(
            area_name="TestArea",
            area_path="/fake/test.are",
            original_rooms=10,
            original_exits=20,
            condensed_rooms=2,
            active_rooms=8,
            dummy_rooms=1,
            model_exits=9,
            one_way_exits=1,
            initial_candidate_pairs=36,
            initial_binary_vars=9,
            initial_constraints=50,
            initial_termination_condition="optimal",
            timings=dummy_timings,
            iterations=[],
            total_collision_iterations=0,
            final_termination_condition="optimal",
            success=True,
        )

        md_overview = format_profile_markdown_table([dummy_res])
        assert "| `TestArea` |" in md_overview
        assert "| Scale (Rooms / Exits) |" in md_overview

        md_breakdown = format_stage_breakdown_table([dummy_res])
        assert "| `TestArea` |" in md_breakdown
        assert "| Parsing |" in md_breakdown


class TestSolverProfilingCliAndNegative:
    """Negative tests and CLI interface tests."""

    def test_cli_execution_with_output(self, tmp_path, monkeypatch):
        """Positive test for CLI execution creating json and markdown outputs."""
        if not SAMPLE_AREAS_DIR:
            pytest.skip("QuickMUD area directory not found")

        json_out = tmp_path / "out.json"
        md_out = tmp_path / "out.md"
        test_args = [
            "profile_solver.py",
            "--areas", "smurf.are",
            "--areas-dir", SAMPLE_AREAS_DIR,
            "--timeout", "10",
            "--max-iterations", "1",
            "--json-out", str(json_out),
            "--markdown-out", str(md_out),
            "--quiet",
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        profile_main()

        assert json_out.is_file()
        assert md_out.is_file()

        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["area_name"] == "Smurfville"

        md_text = md_out.read_text(encoding="utf-8")
        assert "Smurfville" in md_text

    def test_cli_failure_on_nonexistent_area(self, monkeypatch):
        """Negative test for CLI with missing area."""
        test_args = [
            "profile_solver.py",
            "--areas", "definitely_does_not_exist.are",
            "--areas-dir", "/tmp",
            "--timeout", "5",
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        with pytest.raises(SystemExit) as exc:
            profile_main()
        assert exc.value.code == 1
