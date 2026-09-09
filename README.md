# ROMUtil

ROMUtil generates spatial coordinate layouts and map visualizations from text-based MUD area files. It formulates room positioning as a Mixed-Integer Linear Program (MILP) solved via Coin-OR CBC, resolving directed exits, non-Euclidean loops, and vertical elevations into consistent 3D geometric coordinates `(x, y, z)`.

Supported output formats include 3D isometric vector graphics (SVG), standalone interactive web viewers (HTML/JS), and structured room database exports (JSON).

## Prerequisites

- **Python 3.10+**
- **[Coin-OR CBC](https://github.com/coin-or/Cbc)** solver (`cbc` executable must be on `PATH`):
  - Debian / Ubuntu: `sudo apt-get install coinor-cbc`
  - macOS: `brew install cbc`
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`

## Installation

```bash
git clone git@github.com:mitharen/ROMUtil.git
cd ROMUtil
uv sync
```

Alternatively, install in editable mode with pip:
```bash
pip install -e .
```

### Docker

A containerized environment bundling Python and CBC is available:

```bash
docker build -t romutil .
docker run -v $(pwd)/area:/data romutil /data/midgaard.are -outbase /data/midgaard
```

## Usage

```bash
# Generate isometric SVG map
romutil area.are

# Export separate SVGs per elevation plane (<outbase>_z0.svg, <outbase>_z1.svg, ...)
romutil area.are --split-levels

# Export standalone interactive HTML viewer
romutil area.are --format html -outbase my_map

# Export structured JSON room database
romutil area.are --format json -outbase area_data

# Ingest CircleMUD / tbaMUD split world directory (.wld and .zon files)
romutil path/to/world_dir -outbase circlemud_map
romutil --circle-dir path/to/world_dir -outbase circlemud_map

# Configure solver timeout limit in seconds (default: 300)
romutil area.are --solver-timeout 60
```

### Command-Line Reference

```text
usage: romutil [-h] [-outbase OUTBASE] [--split-levels] [-d]
               [-f {svg,json,html}] [--solver-timeout SOLVER_TIMEOUT]
               [--circle-dir CIRCLE_DIR]
               [areas ...]

positional arguments:
  areas                 .ARE files or directories for parsing

options:
  -h, --help            show this help message and exit
  -outbase OUTBASE      output base filename (default: input area name)
  --split-levels        Output separate SVGs for each distinct elevation plane
  -d, --debug           Show debug logging
  -f, --format {svg,json,html}
                        Output format: svg (default), json, or html
  --solver-timeout SOLVER_TIMEOUT
                        CBC solver timeout limit in seconds
  --circle-dir CIRCLE_DIR
                        CircleMUD split world directory containing .wld and .zon files
```

### Batch Map Generation

A [`Makefile`](./Makefile) is provided to batch-compile maps across an area directory:

```bash
make all AREAS=/path/to/QuickMUD/area
```

Area directory resolution order:
1. `make all AREAS=/path/to/area`
2. `QUICKMUD_AREA_DIR` environment variable
3. Sibling repository fallback: `../QuickMUD/area`
4. Default local directory: `areas`

## Supported Dialects

ROMUtil parses and normalizes area formats across major DikuMUD derivations:

| Dialect | Status | Format Notes |
| :--- | :--- | :--- |
| **ROM 2.4 / QuickMUD** | Full | Canonical baseline; `#AREA`, `#ROOMS`, standard 3-field doors |
| **Merc 2.1 / 2.2** | Full | Single-line `#AREA`, piped bitmasks (`4\|8\|1024`), legacy 5-field doors |
| **Envy 1.0 / 2.0** | Full | Key-value `#AREADATA ... End`, `#ROOMDATA`, non-modeled section stripping |
| **CircleMUD 3.x / tbaMUD** | Full | Split-file directory parsing (`*.wld` rooms + `*.zon` headers), 6-field room lines |

Detailed format specifications and verified repository references are cataloged in [`docs/MUD_REPOSITORIES.md`](./docs/MUD_REPOSITORIES.md).

## Documentation

- [`DESIGN.md`](./DESIGN.md): System architecture, mathematical layout formulation, graph reduction, and projection geometry.
- [`docs/SOLVER_PROFILING.md`](./docs/SOLVER_PROFILING.md): MILP solver benchmarks, computational bottlenecks, and optimization analysis.
- [`docs/MUD_REPOSITORIES.md`](./docs/MUD_REPOSITORIES.md): Public MUD area repository catalog and dialect grammar comparison.
- [`AGENTS.md`](./AGENTS.md): Development policies, branching model, and documentation invariants.

## Testing

Run the test suite:
```bash
uv run pytest
```

All tests must pass and coverage across `romutil/` must remain $\ge 95\%$ (enforced by `pytest --cov-fail-under=95`).
