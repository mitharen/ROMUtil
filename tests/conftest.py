"""Shared test configuration, fixtures, and path resolution for ROMUtil."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import pytest

# Repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_sample_areas_dir(
    env_var: str = "QUICKMUD_AREA_DIR",
    custom_root: Optional[Path] = None,
    custom_cwd: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve sample area directory using portable candidate discovery.

    Resolution order:
    1. Environment variable (default QUICKMUD_AREA_DIR).
    2. Sibling directory relative to repository root: ../QuickMUD/area.
    3. Sibling directory relative to current working directory: ../QuickMUD/area.
    4. Repository-local fixtures: tests/fixtures/areas, tests/fixtures, or areas.
    5. Sibling directory relative to git common dir (when running inside a git worktree).

    Returns:
        Resolved Path if found and is a directory, else None.
    """
    # 1. Environment variable
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        p = Path(env_val).expanduser().resolve()
        if p.is_dir():
            return p

    root = custom_root if custom_root is not None else REPO_ROOT
    cwd = custom_cwd if custom_cwd is not None else Path.cwd()

    # 2. Relative sibling repository to repo root
    sibling_root = (root.parent / "QuickMUD" / "area").resolve()
    if sibling_root.is_dir():
        return sibling_root

    # 3. Relative sibling repository to cwd
    sibling_cwd = (cwd / ".." / "QuickMUD" / "area").resolve()
    if sibling_cwd.is_dir():
        return sibling_cwd

    # 4. Repository-local test fixtures
    fixture_candidates = [
        root / "areas",
        root / "tests" / "fixtures" / "areas",
        root / "tests" / "fixtures",
    ]
    for cand in fixture_candidates:
        if cand.is_dir():
            if cand.name == "fixtures" and not any(cand.glob("*.are")):
                continue
            return cand.resolve()

    # 5. Git common directory sibling (supports git worktrees when custom_root is not overridden)
    if custom_root is None:
        try:
            git_common = subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if git_common:
                common_path = Path(git_common)
                if not common_path.is_absolute():
                    common_path = (root / common_path).resolve()
                main_repo = common_path.parent if common_path.name == ".git" else common_path.parent.parent.parent
                worktree_sibling = (main_repo.parent / "QuickMUD" / "area").resolve()
                if worktree_sibling.is_dir():
                    return worktree_sibling
        except Exception:
            pass

    return None


def get_sample_areas_dir(
    env_var: str = "QUICKMUD_AREA_DIR",
    custom_root: Optional[Path] = None,
    custom_cwd: Optional[Path] = None,
) -> str:
    """Return resolved sample area directory path as string, or default fallback path."""
    resolved = resolve_sample_areas_dir(env_var=env_var, custom_root=custom_root, custom_cwd=custom_cwd)
    if resolved is not None:
        return str(resolved)
    root = custom_root if custom_root is not None else REPO_ROOT
    return str((root.parent / "QuickMUD" / "area").resolve())


SAMPLE_AREAS_DIR = get_sample_areas_dir()


@pytest.fixture
def sample_areas_path() -> Optional[Path]:
    """Pytest fixture providing the resolved sample areas Path, or None if unavailable."""
    return resolve_sample_areas_dir()


@pytest.fixture
def sample_areas_dir() -> str:
    """Pytest fixture providing the sample areas directory path string."""
    return SAMPLE_AREAS_DIR


def pytest_configure(config: pytest.Config) -> None:
    """Harmonize coverage thresholds based on active test markers.

    When running filtered test suites (e.g. fast developer feedback via
    '-m "not slow"' or '-m "not integration"'), full solver integration tests
    are omitted, naturally lowering package-wide line coverage. To enable rapid
    developer feedback without spurious coverage failures, cov_fail_under is cleared
    when marker filtering deselects slow or integration tests. Full test suite runs
    (without filtering) retain the strict 95% coverage requirement.
    """
    markexpr = getattr(config.option, "markexpr", "") or ""
    normalized = markexpr.strip().lower()
    if (
        "not slow" in normalized
        or "not integration" in normalized
        or "slow" in normalized
        or "integration" in normalized
    ):
        if hasattr(config.option, "cov_fail_under"):
            config.option.cov_fail_under = None
        cov_plugin = config.pluginmanager.get_plugin("_cov")
        if cov_plugin and hasattr(cov_plugin, "options"):
            cov_plugin.options.cov_fail_under = None
