# ROMUtil

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/coverage-98%25-brightgreen.svg" alt="Coverage" />
  <img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker Ready" />
  <img src="https://img.shields.io/badge/web%20viewer-zero%20dependencies-success.svg" alt="Zero External Dependencies" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License" />
</p>

**ROMUtil** is an automated spatial layout engine and multi-target visualization pipeline for text-based area files (`.are`) from **ROM (Rivers of MUD / DikuMUD)** codebases.

MUD areas were authored by world-builders over decades using directional room exits rather than Cartesian coordinates. Consequently, they contain non-Euclidean loops, straight hallway corridors, cyclic mazes, multi-floor elevation changes, and unlinked boundary exits. ROMUtil transforms these arbitrary directed graphs into consistent 3D geometric coordinates `(x, y, z)` using graph reduction algorithms and **Mixed-Integer Linear Programming (MILP)**, rendering both vector-graphic maps and standalone interactive web applications.

<p align="center">
  <img src="docs/assets/pipeline_diagram.svg" alt="ROMUtil Architecture & Spatial Pipeline" width="100%" />
</p>

---

## Visual Gallery & Interactive Previews

ROMUtil produces publication-quality 3D vector graphics and standalone interactive web applications directly from vintage MUD area files.

### Interactive Web Viewers

Exported via `--format html`, each viewer is a standalone, single-file HTML/JS application with **zero external CDN dependencies or network requests**:

| Area Map | Description | Interactive Web Viewer | Vector Map Source |
| :--- | :--- | :--- | :--- |
| **MUD School** (`school.are`) | Multi-level training academy featuring arenas, cages, and vertical stairways. | [**Launch Interactive Viewer**](./docs/assets/school.html) | [`school.svg`](./docs/assets/school.svg) |
| **Smurf Village** (`smurf.are`) | Sprawling forest village with winding paths, river crossings, and cottages. | [**Launch Interactive Viewer**](./docs/assets/smurf.html) | [`smurf.svg`](./docs/assets/smurf.svg) |
| **Wizard's Spire** (`demo_tower.are`) | Multi-floor vertical tower demonstrating Z-axis layout and elevation transitions. | [**Launch Interactive Viewer**](./docs/assets/demo_tower.html) | [`demo_tower.svg`](./docs/assets/demo_tower.svg) |

### 3D Isometric Vector Maps

#### MUD School (`school.are`)
Full oblique isometric projection (59 rooms, 178 exits) with continuous HSL elevation coloring and interactive room inspection:

<p align="center">
  <img src="docs/assets/school.svg" alt="MUD School 3D Isometric Map" width="85%" />
</p>

#### Multi-Plane Elevation Slices (`--split-levels`)
Individual elevation planes exported independently to eliminate vertical occlusion across stacked floors:

<p align="center">
  <img src="docs/assets/school_z0.svg" alt="MUD School Elevation Plane Z=0" width="45%" />
  &nbsp;&nbsp;
  <img src="docs/assets/school_z1.svg" alt="MUD School Elevation Plane Z=1" width="45%" />
</p>

- Ground Floor Level ($Z=0$): [`docs/assets/school_z0.svg`](./docs/assets/school_z0.svg)
- Upper Academy Level ($Z=1$): [`docs/assets/school_z1.svg`](./docs/assets/school_z1.svg)

#### Smurf Village (`smurf.are`)
Horizontal landscape layout showcasing corridor condensation and planar layout optimization:

<p align="center">
  <img src="docs/assets/smurf.svg" alt="Smurf Village 3D Isometric Map" width="85%" />
</p>

---

## Features

- **3D Isometric SVG Maps**: Oblique isometric vector visualization with interactive mouseover popups showing room names, descriptions, and exits.
- **Z-Axis Elevation Layering**:
  - Embedded SVG JavaScript controls to toggle individual floor levels on/off, eliminating vertical visual occlusion.
  - Multi-plane export (`--split-levels`) generating separate single-floor SVGs (`<outbase>_z{z}.svg`).
  - Continuous HSL color gradients across arbitrary vertical depths ($Z \ge 10$).
- **Interactive Web Map Viewer (`--format html`)**:
  - Exports a single, self-contained HTML/JS web application requiring **zero external dependencies**, network access, CDNs, or build steps.
  - Smooth pan and cursor-centered zoom.
  - Real-time search by room name or VNUM with animated auto-centering.
  - Room detail inspection sidebar with jump-to-exit navigation.
  - Client-side Breadth-First Search (BFS) shortest-path pathfinder with visual route highlighting and turn-by-turn cardinal directions.
  - Radar minimap and floor-level filtering.
- **Structured Data Export (`--format json`)**:
  - Complete room database export with normalized bounding boxes, solved coordinates, and directed exit metadata for downstream tools.
- **Resilient Parsing**:
  - Python PLY (`ply.lex` and `ply.yacc`) lexer/parser with Latin-1 fallback for vintage 8-bit text encodings and tolerance for non-spatial sections (`#SHOPS`, `#RESETS`, `#MOBILES`).

---

## Quick Start

### Prerequisites
- **Python 3.10+** (tested through Python 3.14)
- **[uv](https://docs.astral.sh/uv/)** (recommended) or standard `pip`
- **Coin-OR CBC Solver** (`cbc` executable must be on your `PATH`):
  ```bash
  # Debian / Ubuntu:
  sudo apt-get install coinor-cbc

  # macOS (Homebrew):
  brew install cbc
  ```

### Installation
Clone the repository and sync dependencies using `uv`:
```bash
git clone git@github.com:mitharen/ROMUtil.git
cd ROMUtil
uv sync
```

### Docker Execution

Run ROMUtil with pre-configured Python 3.14 and Coin-OR CBC solvers without local environment setup:

```bash
# Build the container image:
docker build -t romutil .

# Run map generation mounting local area files:
docker run -v $(pwd)/area:/data romutil /data/midgaard.are -outbase /data/midgaard
```

### Basic Usage

Run the CLI on any ROM area file (`.are`):

```bash
# 1. Generate standard isometric SVG map:
uv run romutil path/to/area.are

# 2. Export standalone interactive HTML viewer:
uv run romutil path/to/area.are --format html -outbase my_map

# 3. Export separate SVG maps for each elevation plane:
uv run romutil path/to/area.are --split-levels

# 4. Export structured JSON room database:
uv run romutil path/to/area.are --format json -outbase area_data
```

### Batch Map Generation
The project includes a [`Makefile`](./Makefile) to compile SVG maps across an entire MUD area directory:
```bash
# Build all maps from a QuickMUD area directory:
make all AREAS=/path/to/QuickMUD/area

# Or compile a single area:
make school.svg AREAS=/path/to/QuickMUD/area
```

---

## Command-Line Options

```text
usage: romutil [-h] [-outbase OUTBASE] [--split-levels] [-d] [-f {svg,json,html}] areas [areas ...]

ROM MUD Area Mapper and 3D Visualizer

positional arguments:
  areas                 .ARE files for parsing

options:
  -h, --help            show this help message and exit
  -outbase OUTBASE      output base filename (default: input area name)
  --split-levels        Output separate SVGs for each distinct elevation plane
  -d, --debug           Show debug logging
  -f, --format {svg,json,html}
                        Output format: svg (default), json, or html
```

---

## Architecture & System Design

For a comprehensive explanation of the mathematical optimization, graph algorithms, and system pipeline, see [`DESIGN.md`](./DESIGN.md):
- **Problem Space**: Non-Euclidean geometry, opposing paths of unequal length, and boundary rooms.
- **Topological Condensation (`romutil/graph.py`)**: Collapsing straight degree-2 hallway corridors into composite edges to minimize MILP variable counts.
- **Mathematical Layout Formulation (`romutil/solver.py`)**: Pyomo MILP formulation with big-$M$ cut relaxations and lazy constraint generation for line crossing prevention.
- **Isometric Projection Geometry (`romutil/plotter.py`)**: The $(x, y, z) \to (X', Y')$ oblique projection formula and elevation layer grouping.
- **Web Export Engine (`romutil/exporter.py`)**: JSON schema specification and self-contained viewer application architecture.

---

## Testing & Quality Assurance

ROMUtil maintains a strict code quality and test coverage standard:
- **Run the test suite**:
  ```bash
  uv run pytest
  ```
- **Coverage Enforcement**: Total test coverage across `romutil/` must remain at or above **95%** (enforced by `pytest --cov-fail-under=95`).
- **Pre-commit Hooks**:
  ```bash
  uv run pre-commit install
  ```
  Runs whitespace fixers, validates configuration, enforces documentation synchronization via [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py), and runs the test suite on every commit.

---

## Contributing & Agent Guidelines

For developer and AI agent policies, workspace isolation rules, and design documentation maintenance, refer to [`AGENTS.md`](./AGENTS.md).
