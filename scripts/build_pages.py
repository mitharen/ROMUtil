#!/usr/bin/env python3
"""
build_pages.py - Compiles standalone interactive map viewers and generates
a static site index for GitHub Pages deployment.
"""

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_pages")

# Curated authentic showcase areas bundled in tests/fixtures/areas/
SHOWCASE_AREAS = (
    "arachnos.are",
    "chapel.are",
    "midgaard.are",
    "school.are",
    "shire.are",
    "smurf.are",
)

# Curated composite showcase clusters (category, name/slug, constituent filenames)
COMPOSITE_SHOWCASE_AREAS = (
    (
        "ROM 2.4 / Composite",
        "midgaard_metropolitan",
        ("midgaard.are", "hood.are", "grave.are", "mobfact.are"),
    ),
)


def resolve_candidate_areas() -> list[tuple[str, str, Path | list[Path], bool]]:
    """Discover curated showcase areas and composite clusters for page generation."""
    candidates: list[tuple[str, str, Path | list[Path], bool]] = []

    # Check local fixture areas or external QuickMUD area directory
    sample_dirs = [
        REPO_ROOT / "tests" / "fixtures" / "areas",
    ]
    quickmud_env = os.environ.get("QUICKMUD_AREA_DIR", "").strip()
    if quickmud_env:
        sample_dirs.insert(0, Path(quickmud_env))
    sample_dirs.append(REPO_ROOT.parent / "QuickMUD" / "area")

    found_names = set()
    for sdir in sample_dirs:
        if sdir.is_dir():
            for name in SHOWCASE_AREAS:
                if name in found_names:
                    continue
                area_file = sdir / name
                if area_file.is_file():
                    candidates.append(("ROM 2.4 / QuickMUD", area_file.stem, area_file, False))
                    found_names.add(name)
            if candidates:
                break

    # Discover composite showcase areas
    for cat, comp_name, filenames in COMPOSITE_SHOWCASE_AREAS:
        # Check if all constituent files exist within a single directory first
        found_cluster: list[Path] | None = None
        for sdir in sample_dirs:
            if sdir.is_dir():
                paths = [sdir / fname for fname in filenames]
                if all(p.is_file() for p in paths):
                    found_cluster = paths
                    break
        if found_cluster is not None:
            candidates.append((cat, comp_name, found_cluster, False))
        else:
            resolved_files: list[Path] = []
            for fname in filenames:
                for sdir in sample_dirs:
                    if sdir.is_dir():
                        candidate_file = sdir / fname
                        if candidate_file.is_file():
                            resolved_files.append(candidate_file)
                            break
            if len(resolved_files) == len(filenames):
                candidates.append((cat, comp_name, resolved_files, False))

    candidates.sort(key=lambda c: c[1])
    return candidates


def render_area(
    name: str,
    source_path: Path | Sequence[Path],
    is_dir: bool,
    outdir: Path,
    solver_timeout: int = 60,
) -> None:
    """Invokes romutil to render HTML viewer and SVG vector map in a single pass."""
    outbase = outdir / name

    cmd = [
        sys.executable, "-m", "romutil.cli",
    ]
    if is_dir:
        cmd.extend(["--circle-dir", str(source_path)])
    elif isinstance(source_path, Sequence) and not isinstance(source_path, (str, bytes, Path)):
        cmd.extend(str(p) for p in source_path)
    else:
        cmd.append(str(source_path))

    cmd.extend([
        "-outbase", str(outbase),
        "--format", "html,svg",
        "--solver-timeout", str(solver_timeout),
    ])

    log.info(f"Rendering map artifacts for {name}...")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        log.error(f"Failed to render map artifacts for {name}: {res.stderr}")
        raise RuntimeError(f"Failed to render map artifacts for {name}: {res.stderr}")


def generate_index_html(rendered_areas, outdir):
    """Generates a minimal, clinical index.html landing page."""
    rows = []
    for category, name in rendered_areas:
        html_file = f"{name}.html"
        svg_file = f"{name}0.svg" if (outdir / f"{name}0.svg").exists() else f"{name}.svg"

        html_link = f'<a href="{html_file}">Interactive Viewer</a>' if (outdir / html_file).exists() else "N/A"
        svg_link = f'<a href="{svg_file}">SVG Map</a>' if (outdir / svg_file).exists() else "N/A"

        rows.append(
            f"<tr><td><code>{name}</code></td><td>{category}</td><td>{html_link}</td><td>{svg_link}</td></tr>"
        )

    rows_html = "\n".join(rows)

    index_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROMUtil Maps</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 2rem auto;
      max-width: 800px;
      padding: 0 1rem;
      color: #24292f;
      line-height: 1.5;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    p {{ color: #57606a; margin-top: 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1.5rem;
    }}
    th, td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid #d0d7de;
    }}
    th {{
      background-color: #f6f8fa;
      font-weight: 600;
    }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 0.9em;
      background: #f6f8fa;
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>ROMUtil Map Directory</h1>
  <p>Static map visualizer builds generated via Coin-OR CBC Mixed-Integer Linear Programming.</p>
  <table>
    <thead>
      <tr>
        <th>Area</th>
        <th>Source</th>
        <th>Web Viewer</th>
        <th>Vector SVG</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    (outdir / "index.html").write_text(index_content, encoding="utf-8")
    log.info(f"Generated index.html with {len(rendered_areas)} area maps.")


def main():
    parser = argparse.ArgumentParser(description="Build GitHub Pages site with map viewers")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "_site", help="Output directory")
    parser.add_argument("--solver-timeout", type=int, default=60, help="CBC solver timeout in seconds (default: 60)")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    candidates = resolve_candidate_areas()
    log.info(f"Found {len(candidates)} candidate areas for site generation.")

    rendered = []
    for category, name, source_path, is_dir in candidates:
        render_area(name, source_path, is_dir, args.outdir, solver_timeout=args.solver_timeout)
        rendered.append((category, name))

    generate_index_html(rendered, args.outdir)
    log.info(f"Pages build complete in {args.outdir}")


if __name__ == "__main__":
    main()
