"""Command-line interface and entrypoints for ROMUtil."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

from romutil.layout import (
    ChainedSimplifier,
    HallwayCompactor,
    HierarchicalLayoutAssembler,
    HierarchicalLayoutSolver,
    HierarchicalTreeSolver,
    SpqrTreeBuilder,
    WarpSeverer,
)
from romutil.models import AreaData, AreaHeader, merge_areas
from romutil.parser import Parser, parse_circlemud_directory
from romutil.renderers import RENDERERS, get_renderer
from romutil.types import LayoutOptions, RenderOptions, WorldGraph

logging.basicConfig()
log = logging.getLogger("Mapper")


def main(
    area_files: Sequence[Any],
    outbase: str | None = None,
    split_levels: bool = False,
    fmt: str | Sequence[str] = "svg",
    circle_dir: Path | str | None = None,
    solver_timeout: int | None = None,
    hierarchical: bool = True,
) -> WorldGraph:
    """Parse areas into a unified WorldGraph and optionally render without invoking the solver engine."""
    if isinstance(fmt, str):
        format_list = [f.strip().lower() for f in fmt.split(",") if f.strip()]
    elif isinstance(fmt, (list, tuple, set)):
        format_list = [str(f).strip().lower() for f in fmt if str(f).strip()]
    else:
        raise TypeError(f"Expected format string or sequence of formats, got {type(fmt).__name__}")

    if not format_list:
        raise ValueError("No formats specified.")

    for f in format_list:
        if f not in RENDERERS:
            supported = ", ".join(sorted(RENDERERS.keys()))
            raise ValueError(f"Unsupported format '{f}'. Supported formats: {supported}")

    # Deduplicate preserving order
    seen: set[str] = set()
    formats: list[str] = []
    for f in format_list:
        if f not in seen:
            seen.add(f)
            formats.append(f)

    parsed_areas: list[AreaData] = []
    first_file_name = "area.are"

    if circle_dir is not None:
        first_file_name = str(circle_dir)
        area = parse_circlemud_directory(circle_dir)
        if not isinstance(area, AreaData):
            raise TypeError(f"Expected AreaData from parser, got {type(area).__name__}")
        parsed_areas.append(area)

    for area_file in area_files:
        if isinstance(area_file, (str, Path)) and Path(area_file).is_dir():
            first_file_name = str(area_file)
            area = parse_circlemud_directory(area_file)
            if not isinstance(area, AreaData):
                raise TypeError(f"Expected AreaData from parser, got {type(area).__name__}")
            parsed_areas.append(area)
            continue

        if hasattr(area_file, "read"):
            content = area_file.read()
            first_file_name = getattr(area_file, "name", "area.are")
        else:
            first_file_name = str(area_file)
            with open(area_file, "r", encoding="latin-1", errors="replace") as fp:
                content = fp.read()

        parser = Parser()
        try:
            area = parser.parse(content)
        except Exception as e:
            log.error(e)
            continue

        if not isinstance(area, AreaData):
            raise TypeError(f"Expected AreaData from parser, got {type(area).__name__}")
        parsed_areas.append(area)

    if not parsed_areas or not any(a.rooms for a in parsed_areas):
        log.error("No rooms to plot.")
        sys.exit(0)

    if len(parsed_areas) == 1:
        composite_area = parsed_areas[0]
    else:
        composite_area = merge_areas(parsed_areas)

    area_meta = composite_area.header
    if area_meta is None and composite_area.rooms:
        room_vnums = [r.vnum for r in composite_area.rooms]
        v_min = min(room_vnums)
        v_max = max(room_vnums)
        area_meta = AreaHeader(
            filename=Path(first_file_name).name,
            name=Path(first_file_name).stem,
            builder="",
            vnum_min=v_min,
            vnum_max=v_max,
        )
        composite_area = AreaData(
            header=area_meta,
            rooms=composite_area.rooms,
            mobiles=composite_area.mobiles,
            objects=composite_area.objects,
            resets=composite_area.resets,
            shops=composite_area.shops,
            specials=composite_area.specials,
            helps=composite_area.helps,
            socials=composite_area.socials,
            reset_message=composite_area.reset_message,
            flag=composite_area.flag,
        )

    graph = WorldGraph.from_area_data(composite_area)
    log.info("Parsed %d rooms and %d exits into WorldGraph", len(graph.rooms), len(graph.exits))

    if outbase:
        clean_base = outbase
        for ext in (".html", ".svg", ".json"):
            if clean_base.lower().endswith(ext):
                clean_base = clean_base[:-len(ext)]
                break

        if hierarchical:
            solver = HierarchicalLayoutSolver(
                simplifier=ChainedSimplifier([HallwayCompactor(), WarpSeverer()]),
                tree_builder=SpqrTreeBuilder(),
                tree_solver=HierarchicalTreeSolver(),
                assembler=HierarchicalLayoutAssembler(),
            )
        else:
            solver = HierarchicalLayoutSolver()

        layout_opts = LayoutOptions(timeout_seconds=solver_timeout or 60)
        layout = solver.solve(graph, layout_opts)

        render_opts = RenderOptions(split_levels=split_levels)
        for fmt_name in formats:
            renderer = get_renderer(fmt_name)
            out_file = outbase if outbase.endswith(f".{fmt_name}") else f"{clean_base}.{fmt_name}"
            renderer.render(graph, layout, out_file, options=render_opts)
            log.info("Exported %s to %s", fmt_name.upper(), out_file)

    sys.exit(0)


def cli() -> None:
    """CLI argument parser entrypoint."""
    parser = argparse.ArgumentParser(description="ROM MUD Area Parser and Topology Builder")
    parser.add_argument("areas", nargs="*", type=Path, help=".ARE files or directories for parsing")
    parser.add_argument(
        "--circle-dir",
        type=Path,
        default=None,
        help="CircleMUD split world directory containing .wld and .zon files",
    )
    parser.add_argument("-outbase", help="output base name")
    parser.add_argument(
        "--split-levels",
        action="store_true",
        help="Output separate SVGs for each distinct elevation plane",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    parser.add_argument(
        "-f",
        "--format",
        default="svg",
        help="Output format(s), comma-separated: svg (default), json, html",
    )
    parser.add_argument(
        "--solver-timeout",
        type=int,
        default=None,
        help="Solver timeout limit in seconds",
    )
    parser.add_argument(
        "--hierarchical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hierarchical decomposition (default: True)",
    )
    args = parser.parse_args()

    if not args.areas and not args.circle_dir:
        parser.error("At least one area file or --circle-dir must be provided.")

    raw_fmt = args.format
    format_list = [f.strip().lower() for f in raw_fmt.split(",") if f.strip()]
    if not format_list:
        parser.error("No format specified in --format argument.")
    for f in format_list:
        if f not in RENDERERS:
            supported = ", ".join(sorted(RENDERERS.keys()))
            parser.error(f"Unsupported format '{f}'. Supported formats: {supported}")

    if args.debug:
        log.setLevel(logging.DEBUG)

    outbase = args.outbase
    if not outbase:
        if args.areas:
            outbase = str(args.areas[0].with_suffix(""))
        elif args.circle_dir:
            outbase = str(args.circle_dir / args.circle_dir.name)

    main(
        args.areas,
        outbase,
        split_levels=args.split_levels,
        fmt=format_list,
        circle_dir=args.circle_dir,
        solver_timeout=args.solver_timeout,
        hierarchical=args.hierarchical,
    )


if __name__ == "__main__":
    cli()
