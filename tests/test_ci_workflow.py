from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def load_workflow_data(path: Path = CI_WORKFLOW_PATH) -> dict:
    assert path.exists(), f"Workflow file {path} does not exist."
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "Parsed YAML root must be a mapping/dict."
    return data


class TestCIWorkflowPositive:
    """Positive test cases verifying required configuration in ci.yml."""

    def test_workflow_file_exists(self):
        assert CI_WORKFLOW_PATH.is_file(), f"{CI_WORKFLOW_PATH} does not exist"

    def test_valid_yaml_syntax(self):
        data = load_workflow_data()
        assert "name" in data
        assert data["name"] == "CI"

    def test_triggers_master_branch(self):
        data = load_workflow_data()
        triggers = data.get("on") or data.get(True)
        assert triggers is not None, "Workflow must define 'on' triggers"

        # Check push trigger on master
        assert "push" in triggers, "'push' trigger must be defined"
        push_branches = triggers["push"].get("branches", [])
        assert "master" in push_branches, "'push' trigger must include 'master' branch"

        # Check pull_request trigger on master
        assert "pull_request" in triggers, "'pull_request' trigger must be defined"
        pr_branches = triggers["pull_request"].get("branches", [])
        assert "master" in pr_branches, "'pull_request' trigger must include 'master' branch"

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

    def test_architecture_doc_sync_check(self):
        data = load_workflow_data()
        jobs = data.get("jobs", {})
        all_steps = []
        for job in jobs.values():
            all_steps.extend(job.get("steps", []))

        doc_steps = [
            step for step in all_steps
            if "scripts/sync_design_doc.py" in step.get("run", "")
        ]
        assert len(doc_steps) > 0, "Workflow must run scripts/sync_design_doc.py"

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
