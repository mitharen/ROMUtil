"""
test_mud_repositories.py - Test suite for MUD area repository catalog verification.

Validates that docs/MUD_REPOSITORIES.md adheres to structural invariants:
- Well-formed Markdown syntax, table structures, and Mermaid diagrams.
- Schema conformity for all cataloged repositories (URLs, commit SHAs, paths, licenses).
- Baseline repository inclusion (QuickMUD / ROM 2.4b6 with >= 50 areas).
- Multi-dialect representation across at least 5 distinct MUD lineages.
- Link integrity for all internal file references and external HTTP(S) URLs.
- Comprehensive negative test cases exercising validation rules.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CATALOG_PATH = DOCS_DIR / "MUD_REPOSITORIES.md"


# ---------------------------------------------------------------------------
# Registry Parsing and Schema Validation Helpers
# ---------------------------------------------------------------------------

GITHUB_URL_PATTERN = re.compile(r"^https://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)$")
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FOLDER_PATH_PATTERN = re.compile(r"^/[a-zA-Z0-9_./-]+/$")


def parse_repository_registry(markdown_text: str) -> list[dict[str, str]]:
    """Extracts and parses the repository registry table from markdown text.

    Returns a list of dicts with standardized field keys.
    """
    lines = markdown_text.splitlines()
    table_lines: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "Repository Name" in stripped and "GitHub URL" in stripped:
            in_table = True
            table_lines.append(stripped)
            continue
        if in_table:
            if stripped.startswith("|"):
                table_lines.append(stripped)
            else:
                break

    if len(table_lines) < 3:
        raise ValueError("Repository registry table not found or contains no data rows")

    # Header and separator
    raw_headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(raw_headers):
            raise ValueError(
                f"Row column count ({len(cells)}) does not match header count ({len(raw_headers)}): {line}"
            )
        row_dict = dict(zip(raw_headers, cells))
        rows.append(row_dict)

    return rows


def validate_repository_entry(entry: dict[str, str]) -> None:
    """Validates schema conformance for an individual cataloged repository entry."""
    required_keys = [
        "Repository Name",
        "Codebase Dialect",
        "GitHub URL",
        "Branch",
        "Target Commit SHA",
        "Area Folder Path",
        "Area File Count",
        "Sample Iconic Zones",
        "Distribution License",
    ]
    for key in required_keys:
        if key not in entry or not entry[key].strip():
            raise ValueError(f"Missing or empty required field: {key}")

    # Strip markdown bolding from repository name if present
    repo_name = re.sub(r"^\*\*(.+)\*\*$", r"\1", entry["Repository Name"]).strip()
    if not repo_name:
        raise ValueError("Repository Name cannot be empty")

    # URL validation
    url = entry["GitHub URL"].strip("`").strip()
    match = GITHUB_URL_PATTERN.match(url)
    if not match:
        raise ValueError(f"Invalid GitHub URL format: '{url}' (expected https://github.com/<owner>/<repo>)")

    # Commit SHA validation (must be full 40-character hex)
    commit = entry["Target Commit SHA"].strip("`").strip()
    if not COMMIT_SHA_PATTERN.match(commit):
        raise ValueError(f"Invalid Target Commit SHA: '{commit}' (must be 40-character hexadecimal string)")

    # Branch validation
    branch = entry["Branch"].strip("`").strip()
    if not branch or any(c in branch for c in " \t\n\r"):
        raise ValueError(f"Invalid default branch name: '{branch}'")

    # Area Folder Path validation
    path = entry["Area Folder Path"].strip("`").strip()
    if not FOLDER_PATH_PATTERN.match(path):
        raise ValueError(f"Invalid Area Folder Path: '{path}' (must start and end with '/')")

    # Area File Count validation
    try:
        count = int(entry["Area File Count"].strip())
        if count < 1:
            raise ValueError()
    except ValueError:
        raise ValueError(f"Invalid Area File Count: '{entry['Area File Count']}' (must be integer >= 1)")

    # Sample Iconic Zones validation
    zones = entry["Sample Iconic Zones"].strip()
    if len(zones) < 3:
        raise ValueError(f"Sample Iconic Zones too short or empty: '{zones}'")

    # Distribution License validation
    license_str = entry["Distribution License"].strip()
    if len(license_str) < 3:
        raise ValueError(f"Distribution License too short or empty: '{license_str}'")


def extract_markdown_urls(markdown_text: str) -> list[str]:
    """Finds all explicit http/https URLs in markdown text."""
    url_pattern = re.compile(r"https?://[^\s)\]`'\"<>]+")
    urls = []
    for match in url_pattern.finditer(markdown_text):
        raw_url = match.group(0).rstrip(".,;")
        urls.append(raw_url)
    return urls


def extract_internal_links(markdown_text: str) -> list[tuple[str, str]]:
    """Extracts relative file markdown links [text](target), skipping external URLs and anchors."""
    link_pattern = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^)#]+)(?:#[^)]*)?\)")
    internal_links = []
    for m in link_pattern.finditer(markdown_text):
        target = m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        internal_links.append((m.group(1), target))
    return internal_links


# ---------------------------------------------------------------------------
# Positive Test Suite: Verified Catalog Invariants
# ---------------------------------------------------------------------------

class TestMudRepositoriesCatalogPositive:
    """Positive verification tests for docs/MUD_REPOSITORIES.md."""

    def test_catalog_file_exists(self):
        """Validates that docs/MUD_REPOSITORIES.md exists and is substantive."""
        assert CATALOG_PATH.exists(), f"Missing catalog file at {CATALOG_PATH}"
        content = CATALOG_PATH.read_text(encoding="utf-8")
        assert len(content) > 2000, "Catalog file is unexpectedly brief"

    def test_required_headings_present(self):
        """Verifies that all required functional sections exist in the catalog."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        required_headings = [
            "# Public MUD Area Repository Catalog & Dialect Research",
            "## 1. Executive Summary & Dialect Lineage",
            "## 2. Baseline Repository: QuickMUD (ROM 2.4b6)",
            "## 3. Verified Public MUD Repository Registry",
            "## 4. Codebase Dialect Breakdown & Format Specifications",
            "## 5. Comparative Syntax Matrix",
        ]
        for heading in required_headings:
            assert heading in content, f"Missing required heading: '{heading}'"

    def test_registry_table_schema_and_entries(self):
        """Parses and validates schema conformance for all entries in the registry table."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        entries = parse_repository_registry(content)

        # Must have at least 5 distinct repositories per acceptance criteria (we catalog 16)
        assert len(entries) >= 5, f"Expected at least 5 repositories, found {len(entries)}"

        dialects = set()
        repo_names = set()
        repo_urls = set()

        for entry in entries:
            validate_repository_entry(entry)
            repo_name = re.sub(r"^\*\*(.+)\*\*$", r"\1", entry["Repository Name"]).strip()
            dialects.add(entry["Codebase Dialect"])
            repo_names.add(repo_name)
            repo_urls.add(entry["GitHub URL"])

        # Distinct dialect requirement
        assert len(dialects) >= 5, f"Expected at least 5 distinct dialects, found {len(dialects)}"
        assert len(repo_names) == len(entries), "Duplicate repository names detected in registry"
        assert len(repo_urls) == len(entries), "Duplicate GitHub URLs detected in registry"

    def test_baseline_quickmud_entry(self):
        """Verifies that baseline QuickMUD / ROM 2.4b6 is cataloged with authentic specs."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        entries = parse_repository_registry(content)
        quickmud_entries = [e for e in entries if "quickmud" in e["GitHub URL"].lower()]
        assert len(quickmud_entries) >= 1, "QuickMUD baseline repository not found in registry"

        qm = quickmud_entries[0]
        assert qm["GitHub URL"] == "https://github.com/avinson/rom24-quickmud"
        assert qm["Branch"] == "master"
        assert qm["Target Commit SHA"] == "364c26f1b124e238156e3d11b4e72a8992c66b74"
        assert qm["Area Folder Path"] == "/area/"
        assert int(qm["Area File Count"]) >= 50
        assert "midgaard.are" in qm["Sample Iconic Zones"]
        assert "school.are" in qm["Sample Iconic Zones"]

    def test_major_mud_families_represented(self):
        """Verifies coverage across all major MUD families requested in Task 12."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        entries = parse_repository_registry(content)
        all_dialects = " ".join(e["Codebase Dialect"].lower() for e in entries)

        assert "rom" in all_dialects, "ROM 2.4 family not represented"
        assert "merc" in all_dialects, "Merc family not represented"
        assert "envy" in all_dialects, "Envy family not represented"
        assert "diku" in all_dialects, "DikuMUD family not represented"
        assert "circle" in all_dialects or "tba" in all_dialects, "CircleMUD family not represented"
        assert "smaug" in all_dialects, "SMAUG family not represented"
        assert "anatolia" in all_dialects, "ANATOLIA family not represented"
        assert "ack" in all_dialects, "ACK!MUD family not represented"

    def test_url_integrity_and_formats(self):
        """Validates all URLs in the document are well-formed and valid HTTP(S)."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        urls = extract_markdown_urls(content)
        assert len(urls) >= 15, "Expected at least 15 URLs in catalog"

        for url in urls:
            parsed = urlparse(url)
            assert parsed.scheme in ("http", "https"), f"Invalid URL scheme in {url}"
            assert parsed.netloc, f"Missing network location / hostname in {url}"
            assert not any(c in url for c in " \t\n\r<>\"'"), f"URL contains illegal characters: {url}"

    def test_internal_links_resolve(self):
        """Validates that all relative markdown links point to existing files."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        internal_links = extract_internal_links(content)
        assert len(internal_links) >= 1, "Expected at least one relative internal link"

        for text, target in internal_links:
            resolved = (CATALOG_PATH.parent / target).resolve()
            assert resolved.exists(), f"Broken relative file link [{text}]({target}) -> {resolved}"

    def test_mermaid_diagram_syntax(self):
        """Validates that mermaid diagrams have valid structure."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        diagrams = re.findall(r"```mermaid\n(.*?)\n```", content, re.DOTALL)
        assert len(diagrams) >= 1, "Expected at least 1 mermaid diagram"
        for d in diagrams:
            lines = [l.strip() for l in d.strip().splitlines() if l.strip()]
            assert any(lines[0].startswith(prefix) for prefix in ("flowchart", "graph")), (
                f"Diagram missing graph/flowchart declaration: {lines[0]}"
            )

    def test_no_backlog_tasks_or_legal_essays_in_public_catalog(self):
        """Verifies that public catalog documentation does not contain internal coordinator task descriptions or legal essays."""
        content = CATALOG_PATH.read_text(encoding="utf-8")
        assert "## 6. Licensing & Distribution Permissions Matrix" not in content
        assert "## 7. Downstream Compatibility Roadmap" not in content
        assert "Decomposed Backlog Tasks" not in content
        assert "Task 8a" not in content
        assert "Task 8h" not in content


# ---------------------------------------------------------------------------
# Negative Test Suite: Error Handling and Schema Rejection
# ---------------------------------------------------------------------------

class TestMudRepositoriesCatalogNegative:
    """Negative tests for repository table parser and validation rules."""

    def test_parse_registry_missing_table(self):
        """Rejects markdown without a repository registry table."""
        with pytest.raises(ValueError, match="Repository registry table not found"):
            parse_repository_registry("# Just a header\n\nNo table here.")

    def test_parse_registry_mismatched_columns(self):
        """Rejects markdown table with mismatched column count."""
        malformed_table = (
            "| Repository Name | GitHub URL |\n"
            "| :--- | :--- |\n"
            "| Repo1 | https://github.com/a/b | Extra Column |\n"
        )
        with pytest.raises(ValueError, match="Row column count"):
            parse_repository_registry(malformed_table)

    def test_validate_entry_missing_field(self):
        """Rejects an entry missing a required field."""
        entry = {
            "Repository Name": "QuickMUD",
            # Missing Codebase Dialect
            "GitHub URL": "https://github.com/avinson/rom24-quickmud",
            "Branch": "master",
            "Target Commit SHA": "364c26f1b124e238156e3d11b4e72a8992c66b74",
            "Area Folder Path": "/area/",
            "Area File Count": "53",
            "Sample Iconic Zones": "midgaard.are",
            "Distribution License": "ROM",
        }
        with pytest.raises(ValueError, match="Missing or empty required field: Codebase Dialect"):
            validate_repository_entry(entry)

    def test_validate_entry_invalid_github_url(self):
        """Rejects invalid or non-GitHub URLs."""
        invalid_urls = [
            "http://github.com/avinson/rom24-quickmud",  # http
            "https://gitlab.com/avinson/rom24-quickmud",  # gitlab
            "https://github.com/single_token",  # missing repo
            "not_a_url",
            "https://github.com/user/repo/extra/path",
        ]
        base_entry = {
            "Repository Name": "QuickMUD",
            "Codebase Dialect": "ROM 2.4",
            "GitHub URL": "",
            "Branch": "master",
            "Target Commit SHA": "364c26f1b124e238156e3d11b4e72a8992c66b74",
            "Area Folder Path": "/area/",
            "Area File Count": "53",
            "Sample Iconic Zones": "midgaard.are",
            "Distribution License": "ROM",
        }
        for u in invalid_urls:
            base_entry["GitHub URL"] = u
            with pytest.raises(ValueError, match="Invalid GitHub URL format"):
                validate_repository_entry(base_entry)

    def test_validate_entry_invalid_commit_sha(self):
        """Rejects non-40-hex commit SHAs."""
        invalid_shas = [
            "364c26f",  # 7-char short SHA
            "364c26f1b124e238156e3d11b4e72a8992c66b7G",  # non-hex character G
            "364c26f1b124e238156e3d11b4e72a8992c66b74aaaa",  # too long
            "HEAD",
        ]
        base_entry = {
            "Repository Name": "QuickMUD",
            "Codebase Dialect": "ROM 2.4",
            "GitHub URL": "https://github.com/avinson/rom24-quickmud",
            "Branch": "master",
            "Target Commit SHA": "",
            "Area Folder Path": "/area/",
            "Area File Count": "53",
            "Sample Iconic Zones": "midgaard.are",
            "Distribution License": "ROM",
        }
        # Test empty string check
        base_entry["Target Commit SHA"] = ""
        with pytest.raises(ValueError, match="Missing or empty required field: Target Commit SHA"):
            validate_repository_entry(base_entry)

        # Test invalid values check
        for sha in invalid_shas:
            base_entry["Target Commit SHA"] = sha
            with pytest.raises(ValueError, match="Invalid Target Commit SHA"):
                validate_repository_entry(base_entry)

    def test_validate_entry_invalid_area_path(self):
        """Rejects area folder paths that do not start and end with /."""
        invalid_paths = [
            "area/",  # missing leading slash
            "/area",  # missing trailing slash
            "area",  # relative
        ]
        base_entry = {
            "Repository Name": "QuickMUD",
            "Codebase Dialect": "ROM 2.4",
            "GitHub URL": "https://github.com/avinson/rom24-quickmud",
            "Branch": "master",
            "Target Commit SHA": "364c26f1b124e238156e3d11b4e72a8992c66b74",
            "Area Folder Path": "",
            "Area File Count": "53",
            "Sample Iconic Zones": "midgaard.are",
            "Distribution License": "ROM",
        }
        # Test empty string check
        base_entry["Area Folder Path"] = ""
        with pytest.raises(ValueError, match="Missing or empty required field: Area Folder Path"):
            validate_repository_entry(base_entry)

        # Test invalid values check
        for path in invalid_paths:
            base_entry["Area Folder Path"] = path
            with pytest.raises(ValueError, match="Invalid Area Folder Path"):
                validate_repository_entry(base_entry)

    def test_validate_entry_invalid_file_count(self):
        """Rejects non-integer or negative/zero file counts."""
        invalid_counts = ["0", "-5", "fifty"]
        base_entry = {
            "Repository Name": "QuickMUD",
            "Codebase Dialect": "ROM 2.4",
            "GitHub URL": "https://github.com/avinson/rom24-quickmud",
            "Branch": "master",
            "Target Commit SHA": "364c26f1b124e238156e3d11b4e72a8992c66b74",
            "Area Folder Path": "/area/",
            "Area File Count": "",
            "Sample Iconic Zones": "midgaard.are",
            "Distribution License": "ROM",
        }
        # Test empty string check
        base_entry["Area File Count"] = ""
        with pytest.raises(ValueError, match="Missing or empty required field: Area File Count"):
            validate_repository_entry(base_entry)

        # Test invalid values check
        for count in invalid_counts:
            base_entry["Area File Count"] = count
            with pytest.raises(ValueError, match="Invalid Area File Count"):
                validate_repository_entry(base_entry)

    def test_validate_entry_invalid_branch(self):
        """Rejects whitespace-containing or empty branches."""
        base_entry = {
            "Repository Name": "QuickMUD",
            "Codebase Dialect": "ROM 2.4",
            "GitHub URL": "https://github.com/avinson/rom24-quickmud",
            "Branch": "master branch",
            "Target Commit SHA": "364c26f1b124e238156e3d11b4e72a8992c66b74",
            "Area Folder Path": "/area/",
            "Area File Count": "53",
            "Sample Iconic Zones": "midgaard.are",
            "Distribution License": "ROM",
        }
        with pytest.raises(ValueError, match="Invalid default branch name"):
            validate_repository_entry(base_entry)

    def test_validate_entry_empty_zones_or_license(self):
        """Rejects entries with empty zones or licenses."""
        base_entry = {
            "Repository Name": "QuickMUD",
            "Codebase Dialect": "ROM 2.4",
            "GitHub URL": "https://github.com/avinson/rom24-quickmud",
            "Branch": "master",
            "Target Commit SHA": "364c26f1b124e238156e3d11b4e72a8992c66b74",
            "Area Folder Path": "/area/",
            "Area File Count": "53",
            "Sample Iconic Zones": "",
            "Distribution License": "ROM",
        }
        with pytest.raises(ValueError, match="Missing or empty required field: Sample Iconic Zones"):
            validate_repository_entry(base_entry)

        base_entry["Sample Iconic Zones"] = "midgaard.are"
        base_entry["Distribution License"] = ""
        with pytest.raises(ValueError, match="Missing or empty required field: Distribution License"):
            validate_repository_entry(base_entry)

    def test_extract_internal_links_broken_link_detection(self, tmp_path):
        """Verifies that broken internal relative links can be detected."""
        dummy_md = tmp_path / "test.md"
        dummy_md.write_text("See [nonexistent](does_not_exist.py) file.", encoding="utf-8")
        links = extract_internal_links(dummy_md.read_text(encoding="utf-8"))
        assert len(links) == 1
        resolved = (tmp_path / links[0][1]).resolve()
        assert not resolved.exists()
