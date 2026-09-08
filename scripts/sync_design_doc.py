#!/usr/bin/env python3
"""
sync_design_doc.py - Evaluates code changes against DESIGN.md and updates it as needed.
Runs as a git pre-commit hook to guarantee documentation stays in sync with code.
"""

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = REPO_ROOT / "DESIGN.md"
PACKAGE_DIR = REPO_ROOT / "romutil"

MODULE_DESCRIPTIONS = {
    "__init__.py": "Package public API exports",
    "models.py": "Direction, Room, and Exit domain models",
    "parser.py": "PLY Lexer & LALR Parser with resilient encoding",
    "graph.py": "Corridor collapsing, restoration, and mfas",
    "solver.py": "Pyomo MILP optimization and overlap detection",
    "plotter.py": "Oblique isometric SVG rendering engine",
    "exporter.py": "Interactive JSON and standalone HTML map export",
    "cli.py": "Modernized CLI (pathlib.Path) & entry point",
}

def generate_package_tree():
    """Generates a text directory tree for romutil/ and package structure."""
    lines = ["ROMUtil/", "├── romutil/                     # Core Python package"]
    files = sorted([f.name for f in PACKAGE_DIR.glob("*.py")])
    for i, fname in enumerate(files):
        prefix = "│   └── " if i == len(files) - 1 else "│   ├── "
        desc = MODULE_DESCRIPTIONS.get(fname, f"{fname} module")
        lines.append(f"{prefix}{fname:<24} # {desc}")

    lines.extend([
        "├── pyproject.toml               # PEP 621 package metadata & script definitions",
        "├── uv.lock                      # Pinned dependency lockfile managed by uv",
        "└── tests/                       # Comprehensive unit and integration test suite",
    ])
    return "\n".join(lines)

def validate_links(content):
    """Finds and validates all file links inside DESIGN.md."""
    broken = []
    # Match both file:/// absolute links and relative file paths in markdown links
    link_pattern = re.compile(r'\[([^\]]+)\]\((?:file://)?([^)#]+)(?:#[^)]*)?\)')
    for match in link_pattern.finditer(content):
        target = match.group(2)
        if target.startswith("http://") or target.startswith("https://"):
            continue
        # Check relative or absolute path
        path = Path(target)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            broken.append((match.group(1), target))
    return broken

def verify_modules_documented(content):
    """Verifies that all Python modules in romutil/ are documented in DESIGN.md."""
    missing = []
    for py_file in PACKAGE_DIR.glob("*.py"):
        if py_file.name not in content:
            missing.append(py_file.name)
    return missing

def sync_design_doc():
    if not DESIGN_DOC.exists():
        print(f"[ERROR] {DESIGN_DOC} does not exist!", file=sys.stderr)
        return 1

    content = DESIGN_DOC.read_text(encoding="utf-8")
    original_content = content
    has_errors = False

    # 1. Check for missing module documentation
    missing_modules = verify_modules_documented(content)
    if missing_modules:
        print(f"[ERROR] The following modules in romutil/ are not mentioned in DESIGN.md:", file=sys.stderr)
        for m in missing_modules:
            print(f"  - {m}", file=sys.stderr)
        has_errors = True

    # 2. Check for broken file links
    broken_links = validate_links(content)
    if broken_links:
        print(f"[ERROR] Broken file links found in DESIGN.md:", file=sys.stderr)
        for text, target in broken_links:
            print(f"  - [{text}] -> {target}", file=sys.stderr)
        has_errors = True

    # 3. Update the package tree block in Section 3
    tree_pattern = re.compile(r'(```text\n)(ROMUtil/.*?\n)(```)', re.DOTALL)
    new_tree = generate_package_tree()

    match = tree_pattern.search(content)
    if match:
        old_tree = match.group(2).strip()
        if old_tree != new_tree.strip():
            content = content[:match.start(2)] + new_tree + "\n" + content[match.end(2):]

    if content != original_content:
        DESIGN_DOC.write_text(content, encoding="utf-8")
        print("[INFO] DESIGN.md was automatically updated to reflect current package structure.")
        print("[INFO] Please review and stage the updated DESIGN.md ('git add DESIGN.md').")
        return 1

    if has_errors:
        return 1

    print("[OK] DESIGN.md is up to date and all links are valid.")
    return 0

if __name__ == "__main__":
    sys.exit(sync_design_doc())
