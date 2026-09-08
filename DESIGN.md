# ROMUtil Architecture & Design Guide

**ROMUtil** is an automated pipeline that parses text-based area files (`.are`) from **ROM (Rivers of MUD / DikuMUD)** codebases, calculates consistent 3D geometric coordinates `(x, y, z)` for rooms, resolves non-Euclidean contradictions and spatial collisions, and generates interactive isometric SVG map visualizations.

---

## 1. Problem Space & Challenges

In text-based MUDs, areas were authored manually by human builders over decades. Unlike tile-based maps, MUD areas are arbitrary directed graphs with cardinal exits:
- **Non-Euclidean Topologies**: Moving East and then West does not guarantee you return to the same room. Loops may have unequal opposing path lengths (e.g., 3 rooms North, 2 rooms East, 2 rooms South, 2 rooms West).
- **Corridors & Mazes**: Long straight hallways cause high solver complexity, while mazes contain deliberate cycles and one-way drops.
- **Verticality (Z-Axis)**: Areas have multiple elevation levels connected by `Up` and `Down` exits.
- **Zone Boundaries**: Exits often point to rooms outside the file (`VNUM`s not in the area) or unresolved destinations (`dst == -1`).

ROMUtil solves this layout challenge by combining **graph algorithms** with **Mixed-Integer Linear Programming (MILP)** via Coin-OR CBC and Pyomo.

---

## 2. System Architecture & Pipeline

The project is structured as a modular Python package ([`romutil/`](./romutil)) with a unified console CLI entry point:

```mermaid
flowchart TD
    A[".are Area File(s)"] --> B["romutil/parser.py (PLY Lexer & Parser)"]
    B --> C["romutil/models.py (Room Database & Exit Graph)"]
    C --> D["romutil/graph.py (Corridor Collapse)"]
    D --> E["romutil/solver.py: non_euler() (Cut Minimization)"]
    E --> F["romutil/solver.py: solve() (MILP 3D Coordinate Solver)"]
    F -->|Iterative Collision Resolution| F
    F --> G["romutil/graph.py: restore_rooms() (Corridor Expansion)"]
    G --> H["romutil/plotter.py: Plotter (Isometric SVG Renderer)"]
    G --> J["romutil/exporter.py: JSON & Standalone HTML Exporter"]
    H --> I[".svg Interactive Map"]
    J --> K[".json Room Database / .html Web Viewer"]

    subgraph CLI Entry Points
        CLI1["romutil CLI (uv run romutil)"] --> B
    end
```

The pipeline executes in five distinct phases:
1. **Lexing & Parsing**: Reads `.are` files with Latin-1 fallback into structured Python objects.
2. **Corridor Reduction**: Collapses straight hallways to reduce graph and solver complexity.
3. **Eulerian / Cut Optimization**: Detects geometric inconsistencies and marks minimal exit cuts.
4. **Iterative MILP Placement**: Assigns integer coordinates `(x, y, z)` while lazily preventing exit line collisions.
5. **Restoration & Isometric Plotting**: Re-expands hallways and renders an SVG with interactive mouseover popups.

---

## 3. Package & Module Structure

```text
ROMUtil/
├── romutil/                     # Core Python package
│   ├── __init__.py              # Package public API exports
│   ├── cli.py                   # Modernized CLI (pathlib.Path) & entry point
│   ├── exporter.py              # Interactive JSON and standalone HTML map export
│   ├── graph.py                 # Corridor collapsing, restoration, and mfas
│   ├── models.py                # Direction, Room, and Exit domain models
│   ├── parser.py                # PLY Lexer & LALR Parser with resilient encoding
│   ├── plotter.py               # Oblique isometric SVG rendering engine
│   └── solver.py                # Pyomo MILP optimization and overlap detection
├── pyproject.toml               # PEP 621 package metadata & script definitions
├── uv.lock                      # Pinned dependency lockfile managed by uv
└── tests/                       # Comprehensive unit and integration test suite
```

---

## 4. Component Architecture

### 4.1. Parsing Engine — [`romutil/parser.py`](./romutil/parser.py)

The parsing engine ingests text-based ROM area files and constructs structured in-memory area representations using Python PLY (`ply.lex` and `ply.yacc`).

- **[`Lexer`](./romutil/parser.py)**:
  - Employs dedicated lexer states (`INITIAL`, `string`, `line`, `optional`) to parse mixed-format data.
  - Recognizes tilde-terminated multiline text strings (`~`), cardinal door tokens (`D0` through `D5`), and section headers (`#AREA`, `#ROOMS`, etc.).
- **[`Parser`](./romutil/parser.py)**:
  - An LALR(1) grammar that extracts room topology (VNUMs, titles, descriptions, flags, sector types) and directional exits (destination VNUMs, door keywords, lock flags).
  - Tolerates non-spatial sections (`#SHOPS`, `#RESETS`, `#MOBILES`, `#SPECIALS`) to ensure grammar compatibility across diverse MUD codebases.
  - Reads files using `latin-1` decoding with replacement fallback to handle vintage MUD files containing non-UTF-8 character data.

---

### 4.2. Domain Models & Graph Representation — [`romutil/models.py`](./romutil/models.py)

The domain models define the spatial and topological primitives used throughout the graph layout and rendering pipeline:

- **[`Direction`](./romutil/models.py)**:
  An enumeration representing the 6 degrees of spatial movement (`North`, `East`, `Up`, `South`, `West`, `Down`).
  - Defines opposing directional symmetry via `invert() = (dir + 3) % 6`.
- **[`Exit`](./romutil/models.py)**:
  Represents a directed spatial edge between `src` and `dst` with an associated direction and distance span.
  - Implements symmetrical equality (`__eq__`) and hash invariance so opposing exits (`A -> B East` and `B -> A West`) map to the same logical edge.
  - Tracks whether an exit is strictly `one_way` (lacking a reciprocal return path).
- **[`Room`](./romutil/models.py)**:
  The mutable room node in the spatial graph. Stores integer coordinates `(x, y, z)`, connected exits, and a `fixups` queue of collapsed hallway nodes.
  - Flags synthetic boundary rooms (`room.dummy = True`) created for unresolved or out-of-area destinations.
- **`AreaData`**:
  Top-level container holding the parsed area header, room index, and section definitions.

---

### 4.3. Graph Simplification — [`romutil/graph.py`](./romutil/graph.py)

Before invoking the mathematical solver, the graph is simplified to minimize variables:

1. **Hallway Condensation**:
   Rooms with exactly two opposite exits (e.g., East and West) are straight corridors. `graph()` trims the intermediate room, updates neighbor exits to span the combined distance, and registers the collapsed room in `parent.fixups`.
2. **External & Unresolved Exit Handling**:
   - Exits pointing to `-1` (incomplete rooms) are assigned a synthetic VNUM `max(rdb.keys()) + 1`.
   - Exits leading outside the area file create lightweight `dummy` rooms (`room.dummy = True`) so boundaries can still be routed without crashing.
3. **Connected Components**:
   Uses `networkx.connected_components()` to split disconnected areas into independent subgraphs, solving and plotting each component separately.

---

### 4.4. Mathematical Layout Optimization — [`romutil/solver.py`](./romutil/solver.py)

The layout is formulated as a Mixed-Integer Linear Program (MILP) using **Pyomo** and solved with the **Coin-OR CBC** solver:

#### Variables:
- `m.x[r]`, `m.y[r]`, `m.z[r]`: Integer coordinates for each room `r`.
- `m.cut[e]`: Binary variable indicating if an exit `e` is "cut" (relaxed from geometric constraints).
- `m.l_max[e]`: Positive integer measuring the maximum rendered length of exit `e`.
- `m.one_ways`: Variables bounding Manhattan distance between one-way endpoints.

#### Constraints:
1. **Relative Distance Constraints**:
   For an exit from room $u$ to room $v$ in direction East:
   $$x_v - x_u + M \cdot \text{cut}_e \ge l_{\min}(e)$$
   $$x_v - x_u - M \cdot \text{cut}_e \le l_{\max}(e)$$
   *(where $M$ is a big-M upper bound equal to the sum of all exit lengths).*
2. **Axis Alignment**:
   Non-cardinal axes are constrained to match (e.g., East exits enforce $y_u = y_v$ and $z_u = z_v$ unless cut).
3. **Cut Relaxation**:
   If an area contains contradictory cycles (e.g., a maze or non-Euclidean loop), `cut[e] = 1` disables the strict geometric distance requirement for that edge.

#### Objective Function:
$$\min \left( M^2 \sum \text{cut}_e + \sum l_{\max}(e) + \sum \text{dist}_{\text{one-way}} \right)$$
- Heavy penalty ($M^2$) prevents cutting exits unless mathematically unavoidable.
- Minimizes overall exit lengths to keep rooms compact.
- Keeps one-way endpoints clustered near each other.

#### Lazy Collision Avoidance:
To avoid adding $O(E^2)$ crossing constraints up front:
1. The solver finds an initial coordinate assignment.
2. An overlap detector scans pairs of non-incident exits using bounding-box checks with `tqdm` progress tracking.
3. When two exit lines intersect, disjunctive spatial separation constraints are added using binary relation variables (`relation[Direction]`):
   $$\sum_{d} \text{relation}_d \ge 1$$
4. Generates an intermediate `progress.svg` snapshot after each solver iteration.
5. The solver iterates until no crossings remain or constraints converge.

---

### 4.5. Reconstruction & Rendering — [`romutil/plotter.py`](./romutil/plotter.py)

1. **[`restore_rooms()`](./romutil/graph.py#L12-L29)**:
   Traverses `fixups` on surviving rooms and calculates exact coordinates for previously collapsed corridor rooms.
2. **Isometric Projection**:
   Converts 3D coordinates $(x, y, z)$ into 2D SVG canvas points using an oblique lift factor ($\text{lift} = 0.15$):
   $$X' = 2 + x + \text{lift} \cdot z$$
   $$Y' = 2 + \text{lift} \cdot z_{\max} + (y_{\max} - y) - \text{lift} \cdot z$$
3. **SVG Generation (`svgwrite`) & Multi-Layer Elevation**:
   - **Elevation Grouping**: Elements are grouped by Z-coordinate into `<g id="elevation-{z}" class="elevation-layer" data-z="{z}">` tags for every unique elevation plane.
   - **Interactive Layer Controls**: Embedded `<style>` and JavaScript within the SVG `<defs>` provide clickable toggle buttons (`<g id="elevation-controls">`) with visual active/inactive states, allowing users to toggle individual floor levels on/off to prevent vertical visual occlusion.
   - **Continuous HSL Color Gradient**: Maps elevation levels across a continuous HSL color gradient (`hsl(hue, 75%, 50%)`), providing distinct visual differentiation across arbitrary vertical depths ($Z \ge 10$).
   - Bidirectional exits are drawn as black lines; one-way exits as red lines.
   - External exits are rendered as stub arrows pointing off-map.
   - Interactive `<set>` triggers display floating tooltips on mouseover showing room names, full descriptions, and exit directions.

---

### 4.6. Web & JSON Export Engine — [`romutil/exporter.py`](./romutil/exporter.py)

Exports complete solved area databases into portable structured formats and interactive standalone web viewers:

1. **Structured JSON Map (`build_area_json`, `export_json`)**:
   - Exports the entire room graph with solved integer 3D coordinates `(x, y, z)` and normalized area bounds.
   - Preserves 100% of room definitions, descriptions, and directional connections.
   - Computes directed `one_way` exit properties and directional step distances.
   - Conforms to the standard schema:
     ```json
     {
       "area": { "name": "mud school", "file": "school.are" },
       "bounds": { "min_x": 0, "max_x": 5, "min_y": 0, "max_y": 7, "min_z": 0, "max_z": 4 },
       "rooms": [
         {
           "vnum": 3700,
           "name": "Entrance to Mud School",
           "desc": "...",
           "coords": { "x": 3, "y": 3, "z": 4 },
           "exits": [
             { "direction": "north", "dst": 3757, "distance": 1, "one_way": false }
           ]
         }
       ]
     }
     ```

2. **Standalone HTML Viewer (`generate_html_viewer`, `export_html`)**:
   - A single, self-contained HTML/JS web application requiring **zero external network requests**, CDNs, or Node.js dependencies.
   - Features smooth drag-to-pan and cursor-centered wheel zoom.
   - Interactive search bar with instant VNUM / name filtering, result counts, and animated auto-centering.
   - Floating hover tooltips and rich room detail sidebar inspector with jump-to-exit navigation.
   - Embedded shortest-path Breadth-First Search (BFS) pathfinder with visual route highlighting and turn-by-turn navigation instructions.
   - Elevation floor filter (`Floor Z`) with dynamic color mapping.
   - Embedded interactive radar minimap canvas for orientation and rapid viewport panning.

---

### 4.7. Execution Contract & CLI — [`romutil/cli.py`](./romutil/cli.py)

- Registered console script entry point: `romutil` (via `uv run romutil`).
- Multi-format output support (`--format` / `-f`): `svg` (default), `json`, and `html`.
- Elevation plane splitting (`--split-levels`): Generates separate SVG files for each distinct elevation level (`<outbase>_z{z}.svg`).
- Usage:
  ```bash
  # Generate standard isometric SVG map
  uv run romutil <area.are> [-outbase <name>] [-d]

  # Export split-elevation SVGs
  uv run romutil <area.are> --split-levels

  # Export complete room database as JSON
  uv run romutil <area.are> --format json

  # Export self-contained interactive web viewer
  uv run romutil <area.are> --format html
  ```

---

## 5. Testing & Verification Workflow

The development workflow is standardized using **`uv`**, **`pytest`**, and **`pre-commit`**:

### Running Tests
```bash
uv run pytest
```
Total test coverage across `romutil/` is enforced automatically at or above **95%** on every test execution (`--cov-fail-under=95`).

### Pre-commit Hooks
Automated pre-commit hooks verify code quality before permitting commits:
- Strips trailing whitespace and fixes end-of-file formatting.
- Validates YAML configuration files and blocks large file additions.
- Runs [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py) to guarantee `DESIGN.md` remains synchronized with repository structure and valid links.
- Executes `uv run pytest` to ensure all tests pass and coverage remains $\ge 95\%$.
