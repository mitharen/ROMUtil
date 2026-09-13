import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import networkx as nx
import pyomo.common.config

from romutil.models import Room, Exit, AreaData, AreaHeader, merge_areas
from romutil.parser import Parser, parse_circlemud_directory
from romutil.graph import graph, solve_layout
from romutil.renderers import RENDERERS, get_renderer, render_map

logging.basicConfig()
log = logging.getLogger('Mapper')

def main(
    area_files: Sequence[Any],
    outbase: str,
    split_levels: bool = False,
    fmt: str | Sequence[str] = "svg",
    circle_dir: Path | str | None = None,
    solver_timeout: int | None = None,
) -> None:
    if isinstance(fmt, str):
        format_list = [f.strip().lower() for f in fmt.split(',') if f.strip()]
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

    # Deduplicate while preserving order
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

        # Support both open file objects and Path / string paths
        if hasattr(area_file, 'read'):
            content = area_file.read()
            first_file_name = getattr(area_file, 'name', 'area.are')
        else:
            first_file_name = str(area_file)
            with open(area_file, 'r', encoding='latin-1', errors='replace') as fp:
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
        log.error('No rooms to plot.')
        sys.exit(0)

    if len(parsed_areas) == 1:
        composite_area = parsed_areas[0]
    else:
        composite_area = merge_areas(parsed_areas)

    area_meta = composite_area.header
    rdb = {r.vnum: Room(r) for r in composite_area.rooms}

    # Provide fallback area metadata if none was in the file
    if area_meta is None and rdb:
        room_vnums = list(rdb.keys())
        v_min = min(room_vnums)
        v_max = max(room_vnums)
        area_meta = AreaHeader(
            filename=Path(first_file_name).name,
            name=Path(first_file_name).stem,
            builder="",
            vnum_min=v_min,
            vnum_max=v_max,
        )

    edges = [
        (e.src, e.dst)
        for r in rdb.values()
        for e in r.exits
        if e.src in rdb.keys() and e.dst in rdb.keys()
    ]
    g = nx.DiGraph()
    g.add_nodes_from(rdb.keys())
    g.add_edges_from(edges)

    connected_comps = list(nx.connected_components(g.to_undirected()))
    connected_comps.sort(key=lambda c: min(c) if c else 0)

    all_solved_rooms: dict[int, Room] = {}
    all_exits: list[Exit] = []
    offset_x = 0

    if solver_timeout is None:
        log.info(f"Dynamic solver timeout scaling enabled across {len(connected_comps)} component(s)")
    else:
        log.info(f"Using user-specified fixed solver timeout of {solver_timeout}s across components")

    for i, sub_graph in enumerate(connected_comps):
        sub_rdb = {node: rdb[node] for node in sub_graph}
        solved_sub, solved_exits = solve_layout(sub_rdb, area_meta, solver_timeout=solver_timeout)
        non_dummy = [r for r in solved_sub.values() if not getattr(r, 'dummy', False)]
        if non_dummy:
            if offset_x > 0:
                for r in non_dummy:
                    if r.x is not None:
                        r.x += offset_x
            valid_x = [r.x for r in non_dummy if r.x is not None]
            if valid_x:
                max_x = max(valid_x)
                offset_x = max_x + 3
        all_solved_rooms.update({r.vnum: r for r in non_dummy})
        all_exits.extend(solved_exits)

    # Clean base name (strip known renderer extensions if present)
    clean_base = outbase
    for ext in ('.html', '.svg', '.json'):
        if clean_base.lower().endswith(ext):
            clean_base = clean_base[:-len(ext)]
            break

    for fmt_name in formats:
        renderer = get_renderer(fmt_name)
        if fmt_name == 'html':
            out_file = outbase if outbase.endswith('.html') else f'{clean_base}.html'
            renderer.render(all_solved_rooms, out_file, header=area_meta)
            log.info(f'Exported HTML viewer to {out_file}')
        elif fmt_name == 'json':
            out_file = outbase if outbase.endswith('.json') else f'{clean_base}.json'
            renderer.render(all_solved_rooms, out_file, header=area_meta)
            log.info(f'Exported JSON map to {out_file}')
        elif fmt_name == 'svg':
            out_file = outbase if outbase.endswith('.svg') else f'{clean_base}.svg'
            renderer.render(
                all_solved_rooms,
                out_file,
                header=area_meta,
                exits=all_exits,
                split_levels=split_levels,
                outbase=clean_base,
            )
            if not split_levels:
                out_file_0 = Path(f'{clean_base}0.svg')
                try:
                    shutil.copyfile(out_file, out_file_0)
                except Exception:
                    pass
            log.info(f'Exported SVG map to {out_file}')
        else:
            out_file = outbase if outbase.endswith(f'.{fmt_name}') else f'{clean_base}.{fmt_name}'
            renderer.render(all_solved_rooms, out_file, header=area_meta)
            log.info(f'Exported {fmt_name.upper()} to {out_file}')

    sys.exit(0)

def cli():
    parser = argparse.ArgumentParser(description='ROM MUD Area Mapper and 3D Visualizer')
    parser.add_argument('areas', nargs='*', type=Path, help='.ARE files or directories for parsing')
    parser.add_argument(
        '--circle-dir',
        type=Path,
        default=None,
        help='CircleMUD split world directory containing .wld and .zon files'
    )
    parser.add_argument('-outbase', help='output base name')
    parser.add_argument('--split-levels', action='store_true', help='Output separate SVGs for each distinct elevation plane')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug info')
    parser.add_argument(
        '-f', '--format',
        default='svg',
        help='Output format(s), comma-separated: svg (default), json, html (e.g. -f html,svg)',
    )
    parser.add_argument(
        '--solver-timeout',
        type=int,
        default=None,
        help='CBC solver timeout limit in seconds (default: dynamic room-scaled timeout)',
    )
    args = parser.parse_args()

    if args.areas and str(args.areas[0]).lower() == "map" and not args.areas[0].exists():
        args.areas = args.areas[1:]

    if not args.areas and not args.circle_dir:
        parser.error('At least one area file or --circle-dir must be provided.')

    # Validate format list in CLI
    raw_fmt = args.format
    format_list = [f.strip().lower() for f in raw_fmt.split(',') if f.strip()]
    if not format_list:
        parser.error("No format specified in --format argument.")
    for f in format_list:
        if f not in RENDERERS:
            supported = ", ".join(sorted(RENDERERS.keys()))
            parser.error(f"Unsupported format '{f}'. Supported formats: {supported}")

    pyomo.common.config.logger.setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)

    outbase = args.outbase
    if not outbase:
        if args.areas:
            outbase = str(args.areas[0].with_suffix(''))
        elif args.circle_dir:
            outbase = str(args.circle_dir / args.circle_dir.name)

    main(
        args.areas,
        outbase,
        split_levels=args.split_levels,
        fmt=format_list,
        circle_dir=args.circle_dir,
        solver_timeout=args.solver_timeout,
    )

if __name__ == "__main__":
    cli()
