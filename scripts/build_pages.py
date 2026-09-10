#!/usr/bin/env python3
"""
build_pages.py - Compiles standalone interactive map viewers and generates
a static site index for GitHub Pages deployment.
"""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_pages")


def resolve_candidate_areas():
    """Discover available sample and fixture areas for page generation."""
    candidates = []

    # 1. Check local fixture areas or external QuickMUD area directory
    sample_dirs = [
        REPO_ROOT / "tests" / "fixtures" / "areas",
    ]
    quickmud_env = os.environ.get("QUICKMUD_AREA_DIR", "").strip()
    if quickmud_env:
        sample_dirs.insert(0, Path(quickmud_env))
    sample_dirs.append(REPO_ROOT.parent / "QuickMUD" / "area")

    for sdir in sample_dirs:
        if sdir.is_dir():
            for name in ("school.are", "smurf.are"):
                area_file = sdir / name
                if area_file.is_file():
                    candidates.append(("quickmud", area_file.stem, area_file, False))
            if candidates:
                break

    # 2. Add local repository dialect test fixtures
    dialects_dir = REPO_ROOT / "tests" / "fixtures" / "dialects"
    if dialects_dir.is_dir():
        for are_file in sorted(dialects_dir.glob("*.are")):
            candidates.append(("fixture", are_file.stem, are_file, False))
        circle_world = dialects_dir / "circle_world"
        if circle_world.is_dir():
            candidates.append(("fixture", "circle_world", circle_world, True))

    return candidates


def render_area(name, source_path, is_dir, outdir):
    """Invokes romutil to render HTML viewer and SVG vector map."""
    outbase = outdir / name

    # Render HTML viewer
    cmd_html = [
        sys.executable, "-m", "romutil.cli",
        str(source_path),
        "-outbase", str(outbase),
        "--format", "html",
    ]
    if is_dir:
        cmd_html = [
            sys.executable, "-m", "romutil.cli",
            "--circle-dir", str(source_path),
            "-outbase", str(outbase),
            "--format", "html",
        ]

    log.info(f"Rendering HTML viewer for {name}...")
    res = subprocess.run(cmd_html, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        log.error(f"Failed to render HTML for {name}: {res.stderr}")
        raise RuntimeError(f"Failed to render HTML for {name}: {res.stderr}")

    # Render SVG map
    cmd_svg = [
        sys.executable, "-m", "romutil.cli",
        str(source_path),
        "-outbase", str(outbase),
        "--format", "svg",
    ]
    if is_dir:
        cmd_svg = [
            sys.executable, "-m", "romutil.cli",
            "--circle-dir", str(source_path),
            "-outbase", str(outbase),
            "--format", "svg",
        ]

    log.info(f"Rendering SVG map for {name}...")
    res_svg = subprocess.run(cmd_svg, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res_svg.returncode != 0:
        log.error(f"Failed to render SVG for {name}: {res_svg.stderr}")
        raise RuntimeError(f"Failed to render SVG for {name}: {res_svg.stderr}")


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
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    candidates = resolve_candidate_areas()
    log.info(f"Found {len(candidates)} candidate areas for site generation.")

    rendered = []
    for category, name, source_path, is_dir in candidates:
        render_area(name, source_path, is_dir, args.outdir)
        rendered.append((category, name))

    generate_index_html(rendered, args.outdir)
    log.info(f"Pages build complete in {args.outdir}")


if __name__ == "__main__":
    main()
