"""Smoke tests for Dockerfile, .dockerignore, and devcontainer support."""

import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
DEVCONTAINER_PATH = REPO_ROOT / ".devcontainer" / "devcontainer.json"


def test_dockerfile_structure_and_tooling():
    """Verify Dockerfile exists with multi-stage structure, base images, and required tooling."""
    assert DOCKERFILE_PATH.is_file(), f"{DOCKERFILE_PATH} does not exist"
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    # Multi-stage structure: builder and runtime
    from_stages = re.findall(r"^FROM\s+(\S+)\s+[aA][sS]\s+(\S+)", content, re.MULTILINE)
    assert len(from_stages) >= 2, f"Expected at least 2 stages, found {len(from_stages)}"
    stage_names = [name.lower() for _, name in from_stages]
    assert "builder" in stage_names, "Builder stage missing in Dockerfile"
    assert "runtime" in stage_names, "Runtime stage missing in Dockerfile"

    # Python base image
    for base_image, _ in from_stages:
        assert base_image.startswith("python:3.14-slim"), f"Unexpected base image: {base_image}"

    # uv bundling
    assert "ghcr.io/astral-sh/uv" in content, "Dockerfile must bundle uv from ghcr.io/astral-sh/uv"

    # coinor-cbc package in runtime
    assert "coinor-cbc" in content, "Dockerfile must install coinor-cbc"
    assert "rm -rf /var/lib/apt/lists/*" in content, "Dockerfile must clean apt cache"

    # Workdir and volume
    assert "WORKDIR /data" in content, "Runtime stage must set WORKDIR to /data"
    assert "VOLUME" in content and "/data" in content, "Runtime stage must declare volume"

    # Entrypoint and cmd
    assert "ENTRYPOINT" in content and "romutil" in content, "Dockerfile must declare romutil entrypoint"
    assert "CMD" in content and "--help" in content, "Dockerfile must declare default --help CMD"


def test_dockerignore_rules():
    """Verify .dockerignore exists and includes essential exclusions."""
    assert DOCKERIGNORE_PATH.is_file(), f"{DOCKERIGNORE_PATH} does not exist"
    rules = {
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    expected_rules = {".git", ".venv", "__pycache__"}
    assert expected_rules.issubset(rules), f"Missing required dockerignore rules: {expected_rules - rules}"


def test_devcontainer_configuration():
    """Verify .devcontainer/devcontainer.json exists, is valid JSON, and points to Dockerfile."""
    assert DEVCONTAINER_PATH.is_file(), f"{DEVCONTAINER_PATH} does not exist"
    data = json.loads(DEVCONTAINER_PATH.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    assert "ROMUtil" in data.get("name", "")

    build_cfg = data.get("build", {})
    dockerfile_rel = build_cfg.get("dockerfile")
    assert dockerfile_rel is not None, "devcontainer.json missing build.dockerfile"
    resolved_dockerfile = (DEVCONTAINER_PATH.parent / dockerfile_rel).resolve()
    assert resolved_dockerfile == DOCKERFILE_PATH.resolve()

    assert "uv sync" in data.get("postCreateCommand", "")
    vscode_settings = data.get("customizations", {}).get("vscode", {}).get("settings", {})
    assert vscode_settings.get("python.testing.pytestEnabled") is True
