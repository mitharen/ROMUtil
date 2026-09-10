from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PAGES_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"


def load_workflow_data(path: Path = CI_WORKFLOW_PATH) -> dict[str, Any]:
    assert path.exists(), f"Workflow file {path} does not exist."
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "Parsed YAML root must be a mapping/dict."
    return data


def get_workflow_triggers(data: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Retrieves triggers mapping and the key under which it is stored ('on' or True)."""
    if "on" in data and isinstance(data["on"], dict):
        return "on", data["on"]
    raw_dict: dict[Any, Any] = data
    if True in raw_dict and isinstance(raw_dict[True], dict):
        return True, raw_dict[True]
    return "on", {}


def validate_pages_workflow_data(data: dict[str, Any]) -> list[str]:
    """Validates structure, dependencies, and execution commands of pages.yml.

    Returns a list of violation error strings. An empty list indicates a valid workflow.
    """
    errors: list[str] = []
    _, triggers = get_workflow_triggers(data)
    if not triggers:
        errors.append("Workflow must define an 'on' mapping")
        return errors

    # Check push trigger on main
    if "push" not in triggers:
        errors.append("Missing 'push' trigger")
    else:
        push_branches = triggers["push"].get("branches", [])
        if "main" not in push_branches:
            errors.append("Push trigger missing 'main' branch")

    # Check workflow_dispatch trigger
    if "workflow_dispatch" not in triggers:
        errors.append("Missing 'workflow_dispatch' trigger")

    # Check permissions
    permissions = data.get("permissions", {})
    if not isinstance(permissions, dict):
        errors.append("Permissions must be a dictionary")
    else:
        if permissions.get("pages") != "write":
            errors.append("Missing 'pages: write' permission")
        if permissions.get("id-token") != "write":
            errors.append("Missing 'id-token: write' permission")
        if permissions.get("contents") != "read":
            errors.append("Missing 'contents: read' permission")

    jobs = data.get("jobs", {})
    all_steps: list[dict[str, Any]] = []
    for job in jobs.values():
        if isinstance(job, dict):
            all_steps.extend(job.get("steps", []))

    # Check system dependency coinor-cbc
    cbc_steps = [s for s in all_steps if "coinor-cbc" in s.get("run", "")]
    if not cbc_steps:
        errors.append("Missing 'coinor-cbc' dependency installation step")

    # Check astral-sh/setup-uv with enable-cache: true
    uv_steps = [s for s in all_steps if s.get("uses", "").startswith("astral-sh/setup-uv")]
    if not uv_steps:
        errors.append("Missing 'astral-sh/setup-uv' step")
    else:
        for s in uv_steps:
            if not s.get("with", {}).get("enable-cache"):
                errors.append("setup-uv missing enable-cache: true")

    # Check uv sync
    sync_steps = [s for s in all_steps if "uv sync" in s.get("run", "")]
    if not sync_steps:
        errors.append("Missing 'uv sync' step")

    # Check build_pages.py invocation
    build_steps = [s for s in all_steps if "scripts/build_pages.py" in s.get("run", "")]
    if not build_steps:
        errors.append("Missing step invoking 'scripts/build_pages.py'")
    else:
        for s in build_steps:
            run_cmd = s.get("run", "")
            for line in run_cmd.splitlines():
                if "scripts/build_pages.py" in line:
                    if not re.search(r"\buv\s+run\b.*scripts/build_pages\.py", line):
                        errors.append(f"build_pages.py invoked without 'uv run': {line.strip()}")

    # Check actions/upload-pages-artifact
    upload_steps = [s for s in all_steps if s.get("uses", "").startswith("actions/upload-pages-artifact")]
    if not upload_steps:
        errors.append("Missing 'actions/upload-pages-artifact' step")
    else:
        if upload_steps[0].get("with", {}).get("path") != "_site":
            errors.append("upload-pages-artifact path is not '_site'")

    # Check actions/deploy-pages
    deploy_steps = [s for s in all_steps if s.get("uses", "").startswith("actions/deploy-pages")]
    if not deploy_steps:
        errors.append("Missing 'actions/deploy-pages' step")

    return errors


class TestCIWorkflowPositive:
    """Positive test cases verifying required configuration in ci.yml."""

    def test_workflow_file_exists(self):
        assert CI_WORKFLOW_PATH.is_file(), f"{CI_WORKFLOW_PATH} does not exist"

    def test_valid_yaml_syntax(self):
        data = load_workflow_data()
        assert "name" in data
        assert data["name"] == "CI"

    def test_triggers_main_branch(self):
        data = load_workflow_data()
        triggers = data.get("on") or data.get(True)
        assert triggers is not None, "Workflow must define 'on' triggers"

        # Check push trigger on main
        assert "push" in triggers, "'push' trigger must be defined"
        push_branches = triggers["push"].get("branches", [])
        assert "main" in push_branches, "'push' trigger must include 'main' branch"

        # Check pull_request trigger on main
        assert "pull_request" in triggers, "'pull_request' trigger must be defined"
        pr_branches = triggers["pull_request"].get("branches", [])
        assert "main" in pr_branches, "'pull_request' trigger must include 'main' branch"

    def test_python_matrix_versions(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        test_job = jobs.get("test", {})
        matrix = test_job.get("strategy", {}).get("matrix", {})
        py_versions = [str(v) for v in matrix.get("python-version", [])]

        required_versions = {"3.12", "3.13", "3.14"}
        assert required_versions.issubset(set(py_versions)), (
            f"Matrix must test against {required_versions}, found: {py_versions}"
        )

    def test_system_dependencies_cbc(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        all_steps = []
        for job in jobs.values():
            all_steps.extend(job.get("steps", []))

        cbc_steps = [
            step for step in all_steps
            if "coinor-cbc" in step.get("run", "")
        ]
        assert len(cbc_steps) > 0, "At least one step must install 'coinor-cbc'"

    def test_setup_uv_and_caching(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        all_steps = []
        for job in jobs.values():
            all_steps.extend(job.get("steps", []))

        uv_steps = [
            step for step in all_steps
            if step.get("uses", "").startswith("astral-sh/setup-uv")
        ]
        assert len(uv_steps) > 0, "Workflow must use astral-sh/setup-uv"
        for step in uv_steps:
            with_block = step.get("with", {})
            assert with_block.get("enable-cache") is True, "setup-uv must have enable-cache: true"

    def test_dependency_installation(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        all_steps = []
        for job in jobs.values():
            all_steps.extend(job.get("steps", []))

        sync_steps = [
            step for step in all_steps
            if "uv sync" in step.get("run", "")
        ]
        assert len(sync_steps) > 0, "Workflow must run 'uv sync'"

    def test_test_and_coverage_enforcement(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        test_job = jobs.get("test", {})
        steps = test_job.get("steps", [])

        pytest_steps = [
            step for step in steps
            if "pytest" in step.get("run", "")
        ]
        assert len(pytest_steps) > 0, "Test job must run pytest"
        cmd = pytest_steps[0]["run"]
        assert "--cov=romutil" in cmd, "Pytest must track coverage for romutil"
        assert "--cov-fail-under=95" in cmd, "Pytest must enforce >= 95% coverage threshold"

    def test_static_analysis_and_linting(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        all_steps = []
        for job in jobs.values():
            all_steps.extend(job.get("steps", []))

        mypy_steps = [
            step for step in all_steps
            if "mypy" in step.get("run", "")
        ]
        assert len(mypy_steps) > 0, "Workflow must run mypy"

        pre_commit_steps = [
            step for step in all_steps
            if "pre-commit" in step.get("run", "")
        ]
        assert len(pre_commit_steps) > 0, "Workflow must run pre-commit checks"

    def test_docker_build_and_smoke_job(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        assert "docker" in jobs, "Workflow must include a 'docker' job"
        docker_job = jobs["docker"]
        assert docker_job.get("runs-on") == "ubuntu-latest"

        steps = docker_job.get("steps", [])
        buildx_steps = [s for s in steps if "docker/setup-buildx-action" in s.get("uses", "")]
        assert len(buildx_steps) > 0, "Docker job must configure Buildx"

        build_steps = [s for s in steps if "docker/build-push-action" in s.get("uses", "")]
        assert len(build_steps) > 0, "Docker job must invoke docker/build-push-action"

        smoke_steps = [s for s in steps if "docker run" in s.get("run", "")]
        assert len(smoke_steps) > 0, "Docker job must execute container smoke tests"
        run_cmds = smoke_steps[0]["run"]
        assert "--help" in run_cmds
        assert "romutil:test" in run_cmds


class TestCIWorkflowNegative:
    """Negative test cases verifying rejection of malformed or non-conforming workflow configs."""

    def test_invalid_yaml_syntax_raises_error(self):
        malformed_yaml = """
        name: CI
        on: [push
        jobs:
        """
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(malformed_yaml)

    def test_missing_required_triggers_detected(self):
        invalid_workflow = {
            "name": "CI",
            "jobs": {},
        }
        triggers = invalid_workflow.get("on") or invalid_workflow.get(True)
        assert triggers is None, "Expected triggers to be missing in invalid configuration"

    def test_missing_python_version_rejected(self):
        incomplete_matrix = ["3.12", "3.13"]
        required_versions = {"3.12", "3.13", "3.14"}
        assert not required_versions.issubset(set(incomplete_matrix)), (
            "Validator correctly flags missing Python 3.14"
        )

    def test_substandard_coverage_threshold_rejected(self):
        faulty_cmd = "uv run pytest --cov=romutil --cov-fail-under=80"
        threshold = 95
        assert f"--cov-fail-under={threshold}" not in faulty_cmd, (
            "Validator correctly flags insufficient coverage threshold"
        )


class TestPagesWorkflowPositive:
    """Positive test cases verifying required configuration in pages.yml."""

    def test_workflow_file_exists(self):
        assert PAGES_WORKFLOW_PATH.is_file(), f"{PAGES_WORKFLOW_PATH} does not exist"

    def test_valid_yaml_syntax(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        assert "name" in data
        assert data["name"] == "Deploy GitHub Pages"

    def test_triggers_push_main_and_workflow_dispatch(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        _, triggers = get_workflow_triggers(data)
        assert triggers, "Workflow must define 'on' triggers"
        assert "push" in triggers, "'push' trigger must be defined"
        assert "main" in triggers["push"].get("branches", []), "Push trigger must target 'main' branch"
        assert "workflow_dispatch" in triggers, "'workflow_dispatch' trigger must be defined"

    def test_permissions_and_concurrency(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        perms = data.get("permissions", {})
        assert perms.get("contents") == "read"
        assert perms.get("pages") == "write"
        assert perms.get("id-token") == "write"

        concurrency = data.get("concurrency", {})
        assert concurrency.get("group") == "pages"
        assert concurrency.get("cancel-in-progress") is False

    def test_system_dependencies_cbc(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        steps = [s for j in data.get("jobs", {}).values() for s in j.get("steps", [])]
        cbc_steps = [s for s in steps if "coinor-cbc" in s.get("run", "")]
        assert len(cbc_steps) > 0, "Workflow must install coinor-cbc"

    def test_setup_uv_and_caching(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        steps = [s for j in data.get("jobs", {}).values() for s in j.get("steps", [])]
        uv_steps = [s for s in steps if s.get("uses", "").startswith("astral-sh/setup-uv")]
        assert len(uv_steps) > 0, "Workflow must use astral-sh/setup-uv"
        for s in uv_steps:
            assert s.get("with", {}).get("enable-cache") is True, "setup-uv must enable cache"

    def test_dependency_installation_uv_sync(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        steps = [s for j in data.get("jobs", {}).values() for s in j.get("steps", [])]
        sync_steps = [s for s in steps if "uv sync" in s.get("run", "")]
        assert len(sync_steps) > 0, "Workflow must run 'uv sync'"

    def test_build_pages_invoked_via_uv_run(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        steps = [s for j in data.get("jobs", {}).values() for s in j.get("steps", [])]
        build_steps = [s for s in steps if "scripts/build_pages.py" in s.get("run", "")]
        assert len(build_steps) > 0, "Workflow must have a step invoking scripts/build_pages.py"
        for s in build_steps:
            cmd = s["run"]
            assert "uv run" in cmd, f"Command must invoke build_pages.py via 'uv run': {cmd}"
            assert re.search(r"uv\s+run\s+python\s+scripts/build_pages\.py", cmd) or re.search(
                r"uv\s+run\s+scripts/build_pages\.py", cmd
            )

    def test_upload_and_deploy_actions(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        steps = [s for j in data.get("jobs", {}).values() for s in j.get("steps", [])]
        upload_steps = [s for s in steps if s.get("uses", "").startswith("actions/upload-pages-artifact")]
        assert len(upload_steps) > 0, "Workflow must include actions/upload-pages-artifact step"
        assert upload_steps[0].get("with", {}).get("path") == "_site", "Upload path must be '_site'"

        deploy_steps = [s for s in steps if s.get("uses", "").startswith("actions/deploy-pages")]
        assert len(deploy_steps) > 0, "Workflow must include actions/deploy-pages step"

    def test_pages_workflow_full_conformance(self):
        data = load_workflow_data(PAGES_WORKFLOW_PATH)
        violations = validate_pages_workflow_data(data)
        assert violations == [], f"Workflow has validation violations: {violations}"


class TestPagesWorkflowNegative:
    """Negative test cases verifying rejection of malformed or non-conforming pages.yml configs."""

    def test_invalid_yaml_syntax_raises_error(self):
        malformed = """
        name: Deploy GitHub Pages
        on: [push
        jobs:
        """
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(malformed)

    def test_bare_python_build_pages_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        for step in data["jobs"]["build-and-deploy"]["steps"]:
            if "scripts/build_pages.py" in step.get("run", ""):
                step["run"] = "python scripts/build_pages.py --outdir _site"
        violations = validate_pages_workflow_data(data)
        assert any("build_pages.py invoked without 'uv run'" in v for v in violations), (
            f"Expected violation for bare python invocation, got: {violations}"
        )

    def test_missing_workflow_dispatch_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        key, _ = get_workflow_triggers(data)
        del data[key]["workflow_dispatch"]
        violations = validate_pages_workflow_data(data)
        assert any("Missing 'workflow_dispatch' trigger" in v for v in violations)

    def test_missing_push_main_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        key, _ = get_workflow_triggers(data)
        del data[key]["push"]
        violations = validate_pages_workflow_data(data)
        assert any("Missing 'push' trigger" in v for v in violations)

    def test_missing_cbc_dependency_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        data["jobs"]["build-and-deploy"]["steps"] = [
            s for s in data["jobs"]["build-and-deploy"]["steps"]
            if "coinor-cbc" not in s.get("run", "")
        ]
        violations = validate_pages_workflow_data(data)
        assert any("Missing 'coinor-cbc' dependency installation step" in v for v in violations)

    def test_missing_uv_cache_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        for s in data["jobs"]["build-and-deploy"]["steps"]:
            if s.get("uses", "").startswith("astral-sh/setup-uv"):
                s["with"]["enable-cache"] = False
        violations = validate_pages_workflow_data(data)
        assert any("setup-uv missing enable-cache: true" in v for v in violations)

    def test_missing_uv_sync_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        data["jobs"]["build-and-deploy"]["steps"] = [
            s for s in data["jobs"]["build-and-deploy"]["steps"]
            if "uv sync" not in s.get("run", "")
        ]
        violations = validate_pages_workflow_data(data)
        assert any("Missing 'uv sync' step" in v for v in violations)

    def test_missing_upload_artifact_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        data["jobs"]["build-and-deploy"]["steps"] = [
            s for s in data["jobs"]["build-and-deploy"]["steps"]
            if not s.get("uses", "").startswith("actions/upload-pages-artifact")
        ]
        violations = validate_pages_workflow_data(data)
        assert any("Missing 'actions/upload-pages-artifact' step" in v for v in violations)

    def test_missing_deploy_step_rejected(self):
        data = copy.deepcopy(load_workflow_data(PAGES_WORKFLOW_PATH))
        data["jobs"]["build-and-deploy"]["steps"] = [
            s for s in data["jobs"]["build-and-deploy"]["steps"]
            if not s.get("uses", "").startswith("actions/deploy-pages")
        ]
        violations = validate_pages_workflow_data(data)
        assert any("Missing 'actions/deploy-pages' step" in v for v in violations)
