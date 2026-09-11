"""
Unit and integration tests for bulk external corpus validation harness
(scripts/validate_corpus.py and .github/workflows/validate_corpus.yml).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import MagicMock, patch
from typing import Any
import pytest
import yaml

from scripts.validate_corpus import (
    REPOSITORY_REGISTRY,
    FileFailure,
    RepoEntry,
    RepoResult,
    build_summary_dict,
    clone_or_checkout_repo,
    create_argument_parser,
    filter_repos,
    main,
    print_summary_table,
    run_git_cmd,
    validate_repository,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "validate_corpus.yml"
MUD_REPOSITORIES_MD = REPO_ROOT / "docs" / "MUD_REPOSITORIES.md"

VALID_DIALECTS = {
    "rom",
    "merc",
    "envy",
    "diku",
    "circlemud",
    "smaug",
    "anatolia",
    "ackmud",
}


def load_workflow_data() -> dict[str, Any]:
    assert WORKFLOW_PATH.exists(), f"Workflow file {WORKFLOW_PATH} does not exist."
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "Parsed workflow YAML root must be a mapping/dict."
    return data


def get_workflow_triggers(data: dict[str, Any]) -> dict[str, Any]:
    if "on" in data and isinstance(data["on"], dict):
        return data["on"]
    raw_dict: dict[Any, Any] = data
    if True in raw_dict and isinstance(raw_dict[True], dict):
        return raw_dict[True]
    return {}


def validate_corpus_workflow_data(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    triggers = get_workflow_triggers(data)
    if not triggers:
        errors.append("Workflow must define triggers")
        return errors

    if "workflow_dispatch" not in triggers:
        errors.append("Missing 'workflow_dispatch' trigger")
    else:
        dispatch_inputs = triggers["workflow_dispatch"].get("inputs", {})
        for req_input in ("repo", "dialect", "limit"):
            if req_input not in dispatch_inputs:
                errors.append(f"workflow_dispatch missing input '{req_input}'")

    jobs = data.get("jobs", {})
    if "validate" not in jobs:
        errors.append("Workflow must define a 'validate' job")
        return errors

    validate_job = jobs["validate"]
    if validate_job.get("runs-on") != "ubuntu-latest":
        errors.append("Validate job must run on ubuntu-latest")

    steps = validate_job.get("steps", [])

    checkout_steps = [s for s in steps if s.get("uses", "").startswith("actions/checkout")]
    if not checkout_steps:
        errors.append("Missing 'actions/checkout' step")

    uv_steps = [s for s in steps if s.get("uses", "").startswith("astral-sh/setup-uv")]
    if not uv_steps:
        errors.append("Missing 'astral-sh/setup-uv' step")
    else:
        for s in uv_steps:
            if not s.get("with", {}).get("enable-cache"):
                errors.append("setup-uv step missing enable-cache: true")

    sync_steps = [s for s in steps if "uv sync" in s.get("run", "")]
    if not sync_steps:
        errors.append("Missing 'uv sync' step")

    run_steps = [s for s in steps if "scripts/validate_corpus.py" in s.get("run", "")]
    if not run_steps:
        errors.append("Missing step executing 'scripts/validate_corpus.py'")
    else:
        for s in run_steps:
            cmd = s.get("run", "")
            if not re.search(r"\buv\s+run\s+python\s+scripts/validate_corpus\.py\b", cmd):
                errors.append(f"validate_corpus.py not invoked via 'uv run python': {cmd}")
            if "--summary-json" not in cmd:
                errors.append("validate_corpus.py invocation missing '--summary-json'")

    upload_steps = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
    if not upload_steps:
        errors.append("Missing 'actions/upload-artifact' step")
    else:
        for s in upload_steps:
            if "corpus_summary.json" not in str(s.get("with", {}).get("path", "")):
                errors.append("upload-artifact step missing path 'corpus_summary.json'")

    return errors


# ==============================================================================
# 1. Workflow Validation Tests (Positive & Negative)
# ==============================================================================

class TestCorpusWorkflowPositive:
    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.is_file(), f"{WORKFLOW_PATH} does not exist"

    def test_valid_yaml_syntax(self):
        data = load_workflow_data()
        assert data.get("name") == "Validate External Corpus"

    def test_triggers_workflow_dispatch_inputs(self):
        data = load_workflow_data()
        triggers = get_workflow_triggers(data)
        assert "workflow_dispatch" in triggers
        inputs = triggers["workflow_dispatch"].get("inputs", {})
        assert "repo" in inputs
        assert inputs["repo"].get("default") == "all"
        assert "dialect" in inputs
        assert inputs["dialect"].get("default") == "all"
        assert "limit" in inputs
        assert inputs["limit"].get("default") == "5"

    def test_job_environment_and_setup(self):
        data = load_workflow_data()
        job = data["jobs"]["validate"]
        assert job.get("runs-on") == "ubuntu-latest"

        steps = job.get("steps", [])
        uses = [s.get("uses", "") for s in steps]
        assert any(u.startswith("actions/checkout@v4") for u in uses)

        uv_step = next(s for s in steps if s.get("uses", "").startswith("astral-sh/setup-uv"))
        assert uv_step.get("with", {}).get("enable-cache") is True

        assert any("uv sync" in s.get("run", "") for s in steps)

    def test_corpus_validation_command_and_artifact(self):
        data = load_workflow_data()
        job = data["jobs"]["validate"]
        steps = job.get("steps", [])

        val_step = next(s for s in steps if "scripts/validate_corpus.py" in s.get("run", ""))
        run_cmd = val_step["run"]
        assert "uv run python scripts/validate_corpus.py" in run_cmd
        assert '--repo "${{ inputs.repo }}"' in run_cmd
        assert '--dialect "${{ inputs.dialect }}"' in run_cmd
        assert '--limit "${{ inputs.limit }}"' in run_cmd
        assert '--summary-json "corpus_summary.json"' in run_cmd

        upload_step = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
        assert upload_step.get("if") == "always()"
        assert upload_step.get("with", {}).get("path") == "corpus_summary.json"

    def test_workflow_full_conformance(self):
        data = load_workflow_data()
        violations = validate_corpus_workflow_data(data)
        assert violations == [], f"Workflow has validation violations: {violations}"


class TestCorpusWorkflowNegative:
    def test_invalid_yaml_syntax_raises(self):
        malformed = """
        name: Validate External Corpus
        on: [workflow_dispatch
        """
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(malformed)

    def test_missing_dispatch_trigger_rejected(self):
        data = copy.deepcopy(load_workflow_data())
        key = "on" if "on" in data else True
        data[key] = {"push": {"branches": ["main"]}}
        violations = validate_corpus_workflow_data(data)
        assert any("Missing 'workflow_dispatch' trigger" in v for v in violations)

    def test_missing_dispatch_input_rejected(self):
        data = copy.deepcopy(load_workflow_data())
        key = "on" if "on" in data else True
        del data[key]["workflow_dispatch"]["inputs"]["limit"]
        violations = validate_corpus_workflow_data(data)
        assert any("missing input 'limit'" in v for v in violations)

    def test_bare_python_invocation_rejected(self):
        data = copy.deepcopy(load_workflow_data())
        for step in data["jobs"]["validate"]["steps"]:
            if "scripts/validate_corpus.py" in step.get("run", ""):
                step["run"] = "python scripts/validate_corpus.py --summary-json corpus_summary.json"
        violations = validate_corpus_workflow_data(data)
        assert any("not invoked via 'uv run python'" in v for v in violations)

    def test_missing_upload_artifact_rejected(self):
        data = copy.deepcopy(load_workflow_data())
        data["jobs"]["validate"]["steps"] = [
            s for s in data["jobs"]["validate"]["steps"]
            if not s.get("uses", "").startswith("actions/upload-artifact")
        ]
        violations = validate_corpus_workflow_data(data)
        assert any("Missing 'actions/upload-artifact' step" in v for v in violations)

    def test_missing_uv_cache_rejected(self):
        data = copy.deepcopy(load_workflow_data())
        for s in data["jobs"]["validate"]["steps"]:
            if s.get("uses", "").startswith("astral-sh/setup-uv"):
                s["with"]["enable-cache"] = False
        violations = validate_corpus_workflow_data(data)
        assert any("setup-uv step missing enable-cache: true" in v for v in violations)


# ==============================================================================
# 2. Registry Structure & Metadata Integrity Tests
# ==============================================================================

class TestRegistryStructureIntegrity:
    def test_registry_contains_all_15_repositories(self):
        assert len(REPOSITORY_REGISTRY) == 15, (
            f"Expected 15 authoritative repositories in registry, found {len(REPOSITORY_REGISTRY)}"
        )

    def test_registry_names_and_slugs_are_unique(self):
        names = [r.name for r in REPOSITORY_REGISTRY]
        slugs = [r.slug for r in REPOSITORY_REGISTRY]
        assert len(names) == len(set(names)), "Duplicate repository name found in registry"
        assert len(slugs) == len(set(slugs)), "Duplicate repository slug found in registry"

    def test_registry_urls_valid_github(self):
        for r in REPOSITORY_REGISTRY:
            assert r.url == f"https://github.com/{r.slug}", (
                f"Repository URL mismatch for {r.name}: {r.url} != https://github.com/{r.slug}"
            )

    def test_registry_commit_shas_are_valid_40_hex(self):
        for r in REPOSITORY_REGISTRY:
            assert len(r.commit) == 40, f"SHA length != 40 for {r.name}: {r.commit}"
            assert all(c in "0123456789abcdef" for c in r.commit.lower()), (
                f"SHA contains non-hex characters for {r.name}: {r.commit}"
            )

    def test_registry_area_paths_normalized(self):
        for r in REPOSITORY_REGISTRY:
            assert not r.area_path.startswith("/"), f"Area path has leading slash for {r.name}: {r.area_path}"
            assert not r.area_path.endswith("/"), f"Area path has trailing slash for {r.name}: {r.area_path}"
            assert len(r.area_path) > 0, f"Area path is empty for {r.name}"

    def test_registry_dialects_recognized(self):
        for r in REPOSITORY_REGISTRY:
            assert r.dialect in VALID_DIALECTS, (
                f"Unrecognized dialect '{r.dialect}' in repository {r.name}. Valid: {VALID_DIALECTS}"
            )

    def test_registry_mirrors_mud_repositories_markdown(self):
        """Cross-check registry against authoritative table in docs/MUD_REPOSITORIES.md."""
        assert MUD_REPOSITORIES_MD.exists(), f"{MUD_REPOSITORIES_MD} must exist"
        md_text = MUD_REPOSITORIES_MD.read_text(encoding="utf-8")

        # Parse rows of the table in Section 3
        # Format: | **Repository Name** | Codebase Dialect | GitHub URL | Branch | Target Commit SHA | Area Folder Path | ...
        row_pattern = re.compile(
            r"^\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]+)\s*\|\s*(https://github\.com/[^|\s]+)\s*\|\s*([^|]+)\s*\|\s*([0-9a-f]{40})\s*\|\s*([^|]+)\s*\|",
            re.MULTILINE,
        )

        md_entries: dict[str, dict[str, str]] = {}
        for match in row_pattern.finditer(md_text):
            name, dialect_raw, url, branch, commit, area_path = (
                match.group(1).strip(),
                match.group(2).strip(),
                match.group(3).strip(),
                match.group(4).strip(),
                match.group(5).strip(),
                match.group(6).strip().strip("/"),
            )
            # Skip DikuMUD III since it uses VME format (.zon) rather than .are/.wld
            if "DikuMUD III" in name:
                continue
            md_entries[name] = {
                "url": url,
                "branch": branch,
                "commit": commit,
                "area_path": area_path,
            }

        assert len(md_entries) == 15, f"Expected 15 markdown table entries, found {len(md_entries)}"

        reg_by_name = {r.name: r for r in REPOSITORY_REGISTRY}
        for name, expected in md_entries.items():
            assert name in reg_by_name, f"Repository '{name}' in MUD_REPOSITORIES.md not in REPOSITORY_REGISTRY"
            reg_entry = reg_by_name[name]
            assert reg_entry.url == expected["url"], f"URL mismatch for {name}"
            assert reg_entry.branch == expected["branch"], f"Branch mismatch for {name}"
            assert reg_entry.commit == expected["commit"], f"Commit SHA mismatch for {name}"
            assert reg_entry.area_path == expected["area_path"], f"Area path mismatch for {name}"


# ==============================================================================
# 3. CLI Parser & Filter Tests (Positive & Negative)
# ==============================================================================

class TestCLIParser:
    def test_default_cli_arguments(self):
        parser = create_argument_parser()
        args = parser.parse_args([])
        assert args.repo == "all"
        assert args.dialect == "all"
        assert args.limit is None
        assert args.cache_dir == Path(".corpus_cache")
        assert args.summary_json is None
        assert args.dry_run is False
        assert args.verbose is False

    def test_custom_cli_arguments(self):
        parser = create_argument_parser()
        args = parser.parse_args([
            "--repo", "QuickMUD",
            "--dialect", "rom",
            "--limit", "10",
            "--cache-dir", "/tmp/cache",
            "--summary-json", "summary.json",
            "--dry-run",
            "--verbose",
        ])
        assert args.repo == "QuickMUD"
        assert args.dialect == "rom"
        assert args.limit == 10
        assert args.cache_dir == Path("/tmp/cache")
        assert args.summary_json == Path("summary.json")
        assert args.dry_run is True
        assert args.verbose is True

    def test_negative_limit_rejected_in_main(self):
        ret = main(["--limit", "-1"])
        assert ret == 2

    def test_zero_limit_rejected_in_main(self):
        ret = main(["--limit", "0"])
        assert ret == 2

    def test_invalid_flag_rejected(self):
        parser = create_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--non-existent-flag"])

    def test_help_flag_displays_usage(self, capsys):
        parser = create_argument_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "ROMUtil Bulk External Corpus Validation Harness" in captured.out


class TestRepositoryFiltering:
    def test_filter_all_returns_all_entries(self):
        result = filter_repos(repo_target="all", dialect_target="all")
        assert len(result) == 15

    def test_filter_by_exact_name(self):
        result = filter_repos(repo_target="QuickMUD")
        assert len(result) == 1
        assert result[0].name == "QuickMUD"

    def test_filter_by_slug(self):
        result = filter_repos(repo_target="avinson/rom24-quickmud")
        assert len(result) == 1
        assert result[0].name == "QuickMUD"

    def test_filter_by_slug_leaf(self):
        result = filter_repos(repo_target="rom24-quickmud")
        assert len(result) == 1
        assert result[0].name == "QuickMUD"

    def test_filter_by_dialect(self):
        merc_repos = filter_repos(dialect_target="merc")
        assert len(merc_repos) == 2
        assert {r.name for r in merc_repos} == {"Merc 2.1", "Merc 2.2"}

        circlemud_repos = filter_repos(dialect_target="circlemud")
        assert len(circlemud_repos) == 2
        assert {r.name for r in circlemud_repos} == {"CircleMUD 3.1", "tbaMUD"}

        ack_repos = filter_repos(dialect_target="ackmud")
        assert len(ack_repos) == 2
        assert {r.name for r in ack_repos} == {"AckFUSS", "AckMUD Classic"}

    def test_filter_case_insensitive(self):
        result_lower = filter_repos(repo_target="quickmud")
        assert len(result_lower) == 1
        assert result_lower[0].name == "QuickMUD"

        result_upper = filter_repos(dialect_target="MERC")
        assert len(result_upper) == 2

    def test_filter_combination_match(self):
        result = filter_repos(repo_target="QuickMUD", dialect_target="rom")
        assert len(result) == 1
        assert result[0].name == "QuickMUD"

    def test_filter_combination_mismatch(self):
        result = filter_repos(repo_target="QuickMUD", dialect_target="merc")
        assert len(result) == 0

    def test_filter_unmatched_returns_empty(self):
        result = filter_repos(repo_target="unknown_repository")
        assert result == []


# ==============================================================================
# 4. Mocked Execution & Summary JSON Output Tests
# ==============================================================================

SAMPLE_ARE_CONTENT = """#AREA
sample.are~
Sample Area~
{ 1 10 } Tester  Sample Area~
100 199

#ROOMS
#100
First Room~
A test room.~
0 0 0
D0
North exit description~
~
0 0 101
S
#101
Second Room~
Another test room.~
0 0 0
D2
South exit description~
~
0 0 100
S
#0

#$
"""

SAMPLE_WLD_CONTENT = """#3001
Temple of Midgaard~
You are inside the temple.
~
30 0 0
D0
North exit~
~
0 -1 3002
S
#3002
Temple Hallway~
A quiet hallway.
~
30 0 0
D2
South exit~
~
0 -1 3001
S
#$
"""

CORRUPTED_ARE_CONTENT = """#AREA
broken.are~
Broken~
{ 1 10 } Tester Broken~
100 199

#ROOMS
#100
Room Missing Tildes and Fields
"""


class TestCorpusValidationExecutionMocked:
    def test_validate_repository_success_with_are_file(self, tmp_path):
        repo_dir = tmp_path / "mock_repo"
        area_dir = repo_dir / "area"
        area_dir.mkdir(parents=True)
        (area_dir / "test.are").write_text(SAMPLE_ARE_CONTENT, encoding="latin-1")

        entry = RepoEntry(
            name="Mock ROM",
            slug="mock/rom",
            url="https://github.com/mock/rom",
            branch="master",
            commit="1234567890123456789012345678901234567890",
            area_path="area",
            dialect="rom",
        )

        with patch("scripts.validate_corpus.clone_or_checkout_repo", return_value=repo_dir):
            result = validate_repository(entry, cache_dir=tmp_path / "cache", verbose=True)

        assert result.status == "success"
        assert result.files_scanned == 1
        assert result.files_parsed == 1
        assert result.files_failed == 0
        assert result.rooms == 2
        assert result.exits == 2
        assert len(result.failures) == 0

    def test_validate_repository_success_with_wld_file(self, tmp_path):
        repo_dir = tmp_path / "mock_circlemud"
        area_dir = repo_dir / "lib" / "world" / "wld"
        area_dir.mkdir(parents=True)
        (area_dir / "30.wld").write_text(SAMPLE_WLD_CONTENT, encoding="latin-1")

        entry = RepoEntry(
            name="Mock Circle",
            slug="mock/circle",
            url="https://github.com/mock/circle",
            branch="master",
            commit="abcdefabcdefabcdefabcdefabcdefabcdefabcd",
            area_path="lib/world/wld",
            dialect="circlemud",
        )

        with patch("scripts.validate_corpus.clone_or_checkout_repo", return_value=repo_dir):
            result = validate_repository(entry, cache_dir=tmp_path / "cache")

        assert result.status == "success"
        assert result.files_scanned == 1
        assert result.files_parsed == 1
        assert result.rooms == 2
        assert result.exits == 2

    def test_limit_enforced(self, tmp_path):
        repo_dir = tmp_path / "mock_repo"
        area_dir = repo_dir / "area"
        area_dir.mkdir(parents=True)
        for i in range(5):
            (area_dir / f"zone_{i}.are").write_text(SAMPLE_ARE_CONTENT, encoding="latin-1")

        entry = RepoEntry(
            name="Mock Limit",
            slug="mock/limit",
            url="https://github.com/mock/limit",
            branch="master",
            commit="0000000000000000000000000000000000000000",
            area_path="area",
            dialect="rom",
        )

        with patch("scripts.validate_corpus.clone_or_checkout_repo", return_value=repo_dir):
            result = validate_repository(entry, cache_dir=tmp_path / "cache", limit=2)

        assert result.files_scanned == 2
        assert result.files_parsed == 2

    def test_build_summary_dict_schema(self):
        r1 = RepoResult(
            name="Repo 1",
            slug="org/repo1",
            dialect="rom",
            commit="1111111111111111111111111111111111111111",
            area_path="area",
            status="success",
            files_scanned=2,
            files_parsed=2,
            files_failed=0,
            rooms=50,
            exits=120,
        )
        r2 = RepoResult(
            name="Repo 2",
            slug="org/repo2",
            dialect="merc",
            commit="2222222222222222222222222222222222222222",
            area_path="area",
            status="failed",
            files_scanned=2,
            files_parsed=1,
            files_failed=1,
            rooms=30,
            exits=60,
            failures=[FileFailure(file="area/bad.are", error="SyntaxError", traceback="tb")],
        )

        summary = build_summary_dict([r1, r2], dry_run=False)

        assert summary["dry_run"] is False
        assert "timestamp" in summary
        agg = summary["summary"]
        assert agg["repositories_processed"] == 2
        assert agg["total_files_scanned"] == 4
        assert agg["total_files_parsed"] == 3
        assert agg["total_files_failed"] == 1
        assert agg["total_rooms"] == 80
        assert agg["total_exits"] == 180
        assert agg["overall_success_rate"] == 0.75

        assert "Repo 1" in summary["repositories"]
        assert summary["repositories"]["Repo 1"]["status"] == "success"
        assert summary["repositories"]["Repo 2"]["status"] == "failed"
        assert len(summary["repositories"]["Repo 2"]["failures"]) == 1

    def test_main_dry_run_execution(self, tmp_path):
        summary_file = tmp_path / "dry_summary.json"
        ret = main([
            "--repo", "QuickMUD",
            "--dialect", "rom",
            "--limit", "3",
            "--dry-run",
            "--summary-json", str(summary_file),
        ])
        assert ret == 0
        assert summary_file.is_file()
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        assert data["dry_run"] is True
        assert data["summary"]["repositories_processed"] == 1
        assert "QuickMUD" in data["repositories"]

    def test_print_summary_table(self, capsys):
        res = [
            RepoResult(
                name="QuickMUD",
                slug="avinson/rom24-quickmud",
                dialect="rom",
                commit="364c26f1b124e238156e3d11b4e72a8992c66b74",
                area_path="area",
                status="success",
                files_scanned=5,
                files_parsed=5,
                files_failed=0,
                rooms=500,
                exits=1200,
            )
        ]
        print_summary_table(res)
        out = capsys.readouterr().out
        assert "ROMUtil Bulk External Corpus Validation Summary" in out
        assert "QuickMUD" in out
        assert "SUCCESS" in out
        assert "500" in out


# ==============================================================================
# 5. Error Handling & Edge Case Tests
# ==============================================================================

class TestCorpusValidationErrorHandling:
    def test_corrupt_file_captured_gracefully(self, tmp_path):
        repo_dir = tmp_path / "mock_corrupt"
        area_dir = repo_dir / "area"
        area_dir.mkdir(parents=True)
        (area_dir / "good.are").write_text(SAMPLE_ARE_CONTENT, encoding="latin-1")
        (area_dir / "bad.are").write_text(CORRUPTED_ARE_CONTENT, encoding="latin-1")

        entry = RepoEntry(
            name="Corrupt Test",
            slug="mock/corrupt",
            url="https://github.com/mock/corrupt",
            branch="master",
            commit="1111111111111111111111111111111111111111",
            area_path="area",
            dialect="rom",
        )

        with patch("scripts.validate_corpus.clone_or_checkout_repo", return_value=repo_dir):
            result = validate_repository(entry, cache_dir=tmp_path / "cache", verbose=True)

        assert result.status == "failed"
        assert result.files_scanned == 2
        assert result.files_parsed == 1
        assert result.files_failed == 1
        assert len(result.failures) == 1
        assert result.failures[0].file == "area/bad.are"
        assert len(result.failures[0].traceback) > 0

    def test_missing_area_path_in_cloned_repo(self, tmp_path):
        repo_dir = tmp_path / "mock_empty_repo"
        repo_dir.mkdir(parents=True)

        entry = RepoEntry(
            name="Missing Area",
            slug="mock/missing",
            url="https://github.com/mock/missing",
            branch="master",
            commit="2222222222222222222222222222222222222222",
            area_path="non_existent_folder",
            dialect="rom",
        )

        with patch("scripts.validate_corpus.clone_or_checkout_repo", return_value=repo_dir):
            result = validate_repository(entry, cache_dir=tmp_path / "cache")

        assert result.status == "error"
        assert "does not exist" in (result.error_message or "")

    def test_git_missing_raises_runtime_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("No git")):
            with pytest.raises(RuntimeError, match="git executable not found in system PATH"):
                run_git_cmd(["git", "status"])

    def test_git_called_process_error_raises_runtime_error(self):
        mock_err = subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "clone"],
            stderr="Could not resolve host: github.com\n",
        )
        with patch("subprocess.run", side_effect=mock_err):
            with pytest.raises(RuntimeError, match="Could not resolve host"):
                run_git_cmd(["git", "clone", "bad_url"])

    def test_clone_or_checkout_repo_cached_commit(self, tmp_path):
        cache_dir = tmp_path / "cache"
        repo_dest = cache_dir / "test_slug"
        repo_dest.mkdir(parents=True)
        (repo_dest / ".git").mkdir()

        entry = RepoEntry(
            name="Test",
            slug="test/slug",
            url="https://github.com/test/slug",
            branch="master",
            commit="abc1234567890123456789012345678901234567",
            area_path="area",
            dialect="rom",
        )

        with patch("scripts.validate_corpus.run_git_cmd", return_value="abc1234567890123456789012345678901234567"):
            dest = clone_or_checkout_repo(entry, cache_dir=cache_dir, verbose=True)
            assert dest == repo_dest

    def test_clone_or_checkout_repo_fetch_fallback(self, tmp_path):
        cache_dir = tmp_path / "cache"
        repo_dest = cache_dir / "test_slug"
        repo_dest.mkdir(parents=True)
        (repo_dest / ".git").mkdir()

        entry = RepoEntry(
            name="Test",
            slug="test/slug",
            url="https://github.com/test/slug",
            branch="master",
            commit="abc1234567890123456789012345678901234567",
            area_path="area",
            dialect="rom",
        )

        # Initial rev-parse returns old commit, checkout fails, shallow fetch succeeds
        call_count = 0
        def mock_git(cmd, cwd=None):
            nonlocal call_count
            call_count += 1
            if cmd == ["git", "rev-parse", "HEAD"]:
                return "old_commit_0000000000000000000000000000000"
            if cmd[1] == "checkout" and call_count == 2:
                raise RuntimeError("commit not found locally")
            return ""

        with patch("scripts.validate_corpus.run_git_cmd", side_effect=mock_git):
            dest = clone_or_checkout_repo(entry, cache_dir=cache_dir)
            assert dest == repo_dest

    def test_clone_new_repo_shallow(self, tmp_path):
        cache_dir = tmp_path / "cache"
        entry = RepoEntry(
            name="New",
            slug="new/repo",
            url="https://github.com/new/repo",
            branch="master",
            commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            area_path="area",
            dialect="rom",
        )

        def mock_git(cmd, cwd=None):
            if cmd == ["git", "rev-parse", "HEAD"]:
                return "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
            return ""

        with patch("scripts.validate_corpus.run_git_cmd", side_effect=mock_git):
            dest = clone_or_checkout_repo(entry, cache_dir=cache_dir, verbose=True)
            assert dest == cache_dir / "new_repo"

    def test_main_unmatched_filter_exits_2(self, capsys):
        ret = main(["--repo", "non_existent_mud_repo"])
        assert ret == 2
        captured = capsys.readouterr()
        assert "No repositories matched filter" in captured.err

    def test_main_parse_failure_exits_1(self, tmp_path):
        repo_dir = tmp_path / "mock_fail"
        area_dir = repo_dir / "area"
        area_dir.mkdir(parents=True)
        (area_dir / "corrupt.are").write_text(CORRUPTED_ARE_CONTENT, encoding="latin-1")

        with patch("scripts.validate_corpus.clone_or_checkout_repo", return_value=repo_dir):
            ret = main(["--repo", "QuickMUD", "--cache-dir", str(tmp_path / "cache"), "--limit", "1"])
            assert ret == 1

    def test_main_clone_error_exits_2(self, tmp_path):
        with patch("scripts.validate_corpus.clone_or_checkout_repo", side_effect=RuntimeError("Network down")):
            ret = main(["--repo", "QuickMUD", "--cache-dir", str(tmp_path / "cache")])
            assert ret == 2


# ==============================================================================
# 6. Path Hygiene Tests for Newly Added Modules
# ==============================================================================

class TestCorpusHarnessPathHygiene:
    def test_no_hardcoded_user_paths_in_harness(self):
        prohibited = "/home/" + "user"
        for path in (
            REPO_ROOT / "scripts" / "validate_corpus.py",
            REPO_ROOT / "tests" / "test_validate_corpus.py",
            REPO_ROOT / ".github" / "workflows" / "validate_corpus.yml",
        ):
            assert path.exists(), f"{path} must exist"
            text = path.read_text(encoding="utf-8")
            assert prohibited not in text, f"Found hardcoded user path in {path.name}"
