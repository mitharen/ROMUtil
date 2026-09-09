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
