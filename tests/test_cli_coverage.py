"""Comprehensive tests for romutil.cli and edge cases in romutil.models to ensure >95% package coverage."""

import io
from pathlib import Path
from unittest.mock import patch
import pytest

from romutil.cli import cli, main
from romutil.models import (
    AreaData,
    AreaHeader,
    Direction,
    Exit,
    Room,
    merge_areas,
)
from romutil.renderers import RENDERERS
from romutil.renderers.base import BaseRenderer

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "dialects"


class TestCliCoverage:
    """Targeted coverage tests for romutil.cli branches."""

    def test_main_fmt_types(self, tmp_path):
        are_file = FIXTURES_DIR / "rom24.are"
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

    def test_main_directory_input(self, tmp_path):
        circle_dir = FIXTURES_DIR / "circle_world"
        out = str(tmp_path / "dir_test")
        with pytest.raises(SystemExit) as exc:
            main([circle_dir], out, fmt="json")
        assert exc.value.code == 0
        assert (tmp_path / "dir_test.json").exists()

    def test_main_file_stream_input(self, tmp_path):
        are_file = FIXTURES_DIR / "rom24.are"
        content = are_file.read_text(encoding="latin-1")
        stream = io.StringIO(content)
        stream.name = "stream_test.are"
        out = str(tmp_path / "stream_out")

        with pytest.raises(SystemExit) as exc:
            main([stream], out, fmt="json")
        assert exc.value.code == 0
        assert (tmp_path / "stream_out.json").exists()

    def test_main_corrupt_content_graceful(self, tmp_path, caplog):
        corrupt = io.StringIO("BOGUS DATA WITHOUT ROOMS\n")
        corrupt.name = "corrupt.are"
        out = str(tmp_path / "corrupt_out")

        with pytest.raises(SystemExit) as exc:
            main([corrupt], out, fmt="svg")
        assert exc.value.code == 0
        assert "No rooms to plot" in caplog.text

    def test_main_parser_exception_handling(self, tmp_path):
        are_file = FIXTURES_DIR / "rom24.are"
        out = str(tmp_path / "err_out")

        with patch("romutil.cli.Parser.parse", side_effect=ValueError("Simulated parse error")):
            with pytest.raises(SystemExit) as exc:
                main([are_file], out)
            assert exc.value.code == 0

    def test_main_parser_returns_non_area_data_raises(self, tmp_path):
        are_file = FIXTURES_DIR / "rom24.are"
        out = str(tmp_path / "type_err_out")

        with patch("romutil.cli.Parser.parse", return_value="Not an AreaData"):
            with pytest.raises(TypeError, match="Expected AreaData from parser"):
                main([are_file], out)

    def test_main_circle_dir_non_area_data_raises(self, tmp_path):
        circle_dir = FIXTURES_DIR / "circle_world"
        out = str(tmp_path / "circle_type_err")

        with patch("romutil.cli.parse_circlemud_directory", return_value="Not an AreaData"):
            with pytest.raises(TypeError, match="Expected AreaData from parser"):
                main([], out, circle_dir=circle_dir)

    def test_main_fallback_area_header(self, tmp_path):
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

    def test_main_split_levels(self, tmp_path):
        are_file = FIXTURES_DIR / "rom24.are"
        out = str(tmp_path / "split_test")

        # split_levels=True
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, split_levels=True, fmt="svg")
        assert exc.value.code == 0

        # split_levels=False
        with pytest.raises(SystemExit) as exc:
            main([are_file], out, split_levels=False, fmt="svg")
        assert exc.value.code == 0

    def test_main_custom_format_dispatch(self, tmp_path, monkeypatch):
        class MockCustom(BaseRenderer):
            def render(self, *args, **options):
                out = args[2] if len(args) > 2 else args[1]
                p = Path(out)
                p.write_text("custom-out", encoding="utf-8")
                return p

        monkeypatch.setitem(RENDERERS, "txt", MockCustom)
        are_file = FIXTURES_DIR / "rom24.are"
        out = str(tmp_path / "custom_test")

        with pytest.raises(SystemExit) as exc:
            main([are_file], out, fmt="txt")
        assert exc.value.code == 0
        assert (tmp_path / "custom_test.txt").read_text(encoding="utf-8") == "custom-out"

    def test_cli_branches(self, tmp_path, monkeypatch):
        are_file = FIXTURES_DIR / "rom24.are"
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
            import shutil
            shutil.rmtree(default_debug)


class TestModelEdgeCases:
    """Targeted coverage tests for models.py methods."""

    def test_merge_areas_edge_cases(self):
        # Empty areas list
        empty_merged = merge_areas([])
        assert empty_merged.header is not None
        assert empty_merged.header.name == "Composite Area"
        assert len(empty_merged.rooms) == 0

        # Single area with custom title
        r1 = Room(vnum=1, name="R1", description="D")
        hdr = AreaHeader(filename="a.are", name="Orig", builder="B", vnum_min=1, vnum_max=1)
        area = AreaData(header=hdr, rooms=(r1,))

        merged = merge_areas([area], title="Super World")
        assert merged.header is not None
        assert merged.header.name == "Super World"
