"""Tests for repository-wide path hygiene and sample area resolution portability."""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest

from tests.conftest import (
    REPO_ROOT,
    get_sample_areas_dir,
    resolve_sample_areas_dir,
)

# Dynamically constructed to avoid literal git grep matches
TARGET_USER_HOME = "/home/" + "user"
USER_HOME_PATTERN = re.compile(r"/home/[a-zA-Z0-9_-]+/(?:proj|projects|work|git|repo|ROMUtil|QuickMUD|area)")


def get_tracked_files(repo_dir: Path) -> List[Path]:
    """Retrieve all tracked files in the repository using git ls-files."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [repo_dir / f.strip() for f in output.splitlines() if f.strip()]
    except Exception:
        tracked: List[Path] = []
        excluded_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                tracked.append(Path(root) / file)
        return tracked


def scan_file_for_hardcoded_paths(file_path: Path) -> List[Tuple[int, str]]:
    """Scan a file for prohibited developer home directory paths.

    Returns a list of (line_number, line_content) tuples where violations were found.
    """
    violations: List[Tuple[int, str]] = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return violations

    for line_idx, line in enumerate(content.splitlines(), start=1):
        if TARGET_USER_HOME in line or USER_HOME_PATTERN.search(line):
            violations.append((line_idx, line.strip()))

    return violations


class TestRepositoryPathHygiene:
    """Regression tests verifying zero hardcoded developer paths in version control."""

    def test_no_hardcoded_user_paths_in_tracked_files(self):
        """Verify that zero tracked repository files contain hardcoded developer paths."""
        tracked_files = get_tracked_files(REPO_ROOT)
        assert len(tracked_files) > 0, "Expected git tracked files to be discovered"

        all_violations: List[str] = []
        for file_path in tracked_files:
            if not file_path.is_file():
                continue
            violations = scan_file_for_hardcoded_paths(file_path)
            for line_no, text in violations:
                rel_path = file_path.relative_to(REPO_ROOT)
                all_violations.append(f"{rel_path}:{line_no}: {text}")

        assert not all_violations, (
            f"Found {len(all_violations)} hardcoded path violations in tracked files:\n"
            + "\n".join(all_violations)
        )

    def test_makefile_uses_portable_areas_variable(self):
        """Verify Makefile defines AREAS using configurable environment variables without hardcoded paths."""
        makefile = REPO_ROOT / "Makefile"
        assert makefile.exists(), "Makefile must exist at repository root"
        content = makefile.read_text(encoding="utf-8")

        assert TARGET_USER_HOME not in content, "Makefile must not contain hardcoded user home path"
        assert "AREAS ?= $(or $(QUICKMUD_AREA_DIR),$(wildcard ../QuickMUD/area),areas)" in content, (
            "Makefile must use portable configurable AREAS variable definition"
        )

    def test_test_suites_use_shared_area_discovery(self):
        """Verify test modules import SAMPLE_AREAS_DIR from shared conftest instead of hardcoding paths."""
        test_files = [
            REPO_ROOT / "tests" / "test_area_parser.py",
            REPO_ROOT / "tests" / "test_exporter.py",
            REPO_ROOT / "tests" / "test_mapper.py",
            REPO_ROOT / "tests" / "test_typed_models.py",
        ]
        for test_file in test_files:
            assert test_file.exists(), f"{test_file.name} must exist"
            content = test_file.read_text(encoding="utf-8")
            assert TARGET_USER_HOME not in content, f"{test_file.name} contains hardcoded user path"
            assert "from tests.conftest import SAMPLE_AREAS_DIR" in content, (
                f"{test_file.name} should import SAMPLE_AREAS_DIR from tests.conftest"
            )

    def test_scanner_flags_violations_positive_and_negative(self, tmp_path):
        """Positive and negative validation of the path hygiene scanner itself."""
        clean_file = tmp_path / "clean.py"
        clean_file.write_text("AREAS_DIR = os.environ.get('QUICKMUD_AREA_DIR', 'areas')\n")
        assert scan_file_for_hardcoded_paths(clean_file) == []

        violation_file = tmp_path / "dirty.py"
        violation_file.write_text(f"PATH = '{TARGET_USER_HOME}/proj/QuickMUD/area'\n")
        detected = scan_file_for_hardcoded_paths(violation_file)
        assert len(detected) == 1
        assert detected[0][0] == 1
        assert TARGET_USER_HOME in detected[0][1]


class TestAreaPathResolution:
    """Unit tests for portable sample area directory resolution in tests/conftest.py."""

    def test_env_var_override_takes_precedence(self, tmp_path, monkeypatch):
        """QUICKMUD_AREA_DIR environment variable is preferred over other locations."""
        mock_area_dir = tmp_path / "custom_areas"
        mock_area_dir.mkdir()
        monkeypatch.setenv("QUICKMUD_AREA_DIR", str(mock_area_dir))

        resolved = resolve_sample_areas_dir()
        assert resolved == mock_area_dir
        assert get_sample_areas_dir() == str(mock_area_dir)

    def test_env_var_nonexistent_falls_through(self, monkeypatch, tmp_path):
        """Invalid or non-existent QUICKMUD_AREA_DIR paths fall through to next candidates."""
        monkeypatch.setenv("QUICKMUD_AREA_DIR", str(tmp_path / "does_not_exist"))

        fake_root = tmp_path / "fake_repo"
        fake_root.mkdir()
        sibling_area = tmp_path / "QuickMUD" / "area"
        sibling_area.mkdir(parents=True)

        resolved = resolve_sample_areas_dir(custom_root=fake_root)
        assert resolved == sibling_area

    def test_sibling_relative_discovery(self, tmp_path, monkeypatch):
        """Discovers ../QuickMUD/area relative to the repository root."""
        monkeypatch.delenv("QUICKMUD_AREA_DIR", raising=False)
        fake_root = tmp_path / "workspace" / "ROMUtil"
        fake_root.mkdir(parents=True)
        sibling = tmp_path / "workspace" / "QuickMUD" / "area"
        sibling.mkdir(parents=True)

        resolved = resolve_sample_areas_dir(custom_root=fake_root)
        assert resolved == sibling
        assert get_sample_areas_dir(custom_root=fake_root) == str(sibling)

    def test_cwd_sibling_discovery(self, tmp_path, monkeypatch):
        """Discovers ../QuickMUD/area relative to current working directory."""
        monkeypatch.delenv("QUICKMUD_AREA_DIR", raising=False)
        empty_root = tmp_path / "empty_root"
        empty_root.mkdir()
        fake_cwd = tmp_path / "run_dir" / "subdir"
        fake_cwd.mkdir(parents=True)
        sibling = tmp_path / "run_dir" / "QuickMUD" / "area"
        sibling.mkdir(parents=True)

        resolved = resolve_sample_areas_dir(custom_root=empty_root, custom_cwd=fake_cwd)
        assert resolved == sibling

    def test_repository_local_fixtures_discovery(self, tmp_path, monkeypatch):
        """Discovers local test fixtures when external sibling repositories do not exist."""
        monkeypatch.delenv("QUICKMUD_AREA_DIR", raising=False)
        fake_root = tmp_path / "isolated_repo"
        local_fixtures = fake_root / "tests" / "fixtures" / "areas"
        local_fixtures.mkdir(parents=True)

        resolved = resolve_sample_areas_dir(custom_root=fake_root, custom_cwd=fake_root)
        assert resolved == local_fixtures

    def test_fallback_when_no_candidate_exists(self, tmp_path, monkeypatch):
        """Returns None for resolve_sample_areas_dir and a fallback path for get_sample_areas_dir."""
        monkeypatch.delenv("QUICKMUD_AREA_DIR", raising=False)
        isolated_root = tmp_path / "isolated"
        isolated_root.mkdir()

        resolved = resolve_sample_areas_dir(custom_root=isolated_root, custom_cwd=isolated_root)
        assert resolved is None

        dir_str = get_sample_areas_dir(custom_root=isolated_root, custom_cwd=isolated_root)
        assert isinstance(dir_str, str)
        assert dir_str == str((isolated_root.parent / "QuickMUD" / "area").resolve())

    def test_pytest_fixtures_integration(self, sample_areas_path, sample_areas_dir):
        """Verify conftest fixtures return expected types."""
        if sample_areas_path is not None:
            assert isinstance(sample_areas_path, Path)
            assert sample_areas_path.is_dir()
        assert isinstance(sample_areas_dir, str)
