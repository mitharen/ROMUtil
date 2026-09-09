import argparse
import logging
import os
from pathlib import Path
import sys

import networkx as nx
import pyomo.common.config

from romutil.models import Room, AreaData, AreaHeader
from romutil.parser import Parser, parse_circlemud_directory
from romutil.graph import graph, solve_layout
from romutil.exporter import build_area_json, export_json, export_html

logging.basicConfig()
log = logging.getLogger('Mapper')

def main(area_files, outbase, split_levels=False, fmt="svg", circle_dir=None):
    rdb = {}
    area_meta = None
    first_file_name = "area.are"

    if circle_dir is not None:
        first_file_name = str(circle_dir)
        area = parse_circlemud_directory(circle_dir)
        if not isinstance(area, AreaData):
            raise TypeError(f"Expected AreaData from parser, got {type(area).__name__}")
        if area.header:
            area_meta = area.header
        rooms = list(area.rooms)
        rdb.update({r.vnum: Room(r) for r in rooms})

    for area_file in area_files:
        if isinstance(area_file, (str, Path)) and Path(area_file).is_dir():
            first_file_name = str(area_file)
            area = parse_circlemud_directory(area_file)
            if not isinstance(area, AreaData):
                raise TypeError(f"Expected AreaData from parser, got {type(area).__name__}")
            if area.header:
                area_meta = area.header
            rooms = list(area.rooms)
            rdb.update({r.vnum: Room(r) for r in rooms})
            continue

        # Support both open file objects and Path / string paths
        if hasattr(area_file, 'read'):
            content = area_file.read()
            first_file_name = getattr(area_file, 'name', 'area.are')
        else:
            first_file_name = str(area_file)
            with open(area_file, 'r', encoding='latin-1', errors='replace') as f:
                content = f.read()

        parser = Parser()
        try:
            area = parser.parse(content)
        except Exception as e:
            log.error(e)
            continue

        if not isinstance(area, AreaData):
            raise TypeError(f"Expected AreaData from parser, got {type(area).__name__}")

        area_meta = area.header
        rooms = list(area.rooms)
        rdb.update({r.vnum: Room(r) for r in rooms})

    if not rdb:
        log.error('No rooms to plot.')
        sys.exit(0)

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

    if fmt == 'svg':
        for i, sub_graph in enumerate(connected_comps):
            sub_rdb = {node: rdb[node] for node in sub_graph}
            if not split_levels:
                graph(sub_rdb, f'{outbase}{i}.svg', area_meta, split_levels=False)
            else:
                base_name = outbase if len(connected_comps) == 1 else f'{outbase}{i}'
                graph(sub_rdb, f'{base_name}.svg', area_meta, split_levels=True, outbase=base_name)
    elif fmt in ('json', 'html'):
        all_solved_rooms = {}
        offset_x = 0
        for i, sub_graph in enumerate(connected_comps):
            sub_rdb = {node: rdb[node] for node in sub_graph}
            solved_sub, _ = solve_layout(sub_rdb, area_meta)
            non_dummy = [r for r in solved_sub.values() if not r.dummy]
            if non_dummy:
                if offset_x > 0:
                    for r in non_dummy:
                        if r.x is not None:
                            r.x += offset_x
                max_x = max(r.x for r in non_dummy if r.x is not None)
                offset_x = max_x + 3
            all_solved_rooms.update({r.vnum: r for r in non_dummy})

        # Provide fallback area metadata if none was in the file
        if area_meta is None:
            area_meta = AreaHeader(
                filename=Path(first_file_name).name,
                name=Path(first_file_name).stem,
                builder="",
                vnum_min=0,
                vnum_max=0,
            )

        data = build_area_json(all_solved_rooms, area_meta)

        if fmt == 'json':
            json_path = outbase if outbase.endswith('.json') else f'{outbase}.json'
            export_json(data, json_path)
            log.info(f'Exported JSON map to {json_path}')
        elif fmt == 'html':
            html_path = outbase if outbase.endswith('.html') else f'{outbase}.html'
            export_html(data, html_path)
            log.info(f'Exported HTML viewer to {html_path}')

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
        choices=['svg', 'json', 'html'],
        default='svg',
        help='Output format: svg (default), json, or html'
    )
    args = parser.parse_args()

    if not args.areas and not args.circle_dir:
        parser.error('At least one area file or --circle-dir must be provided.')

    pyomo.common.config.logger.setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)

    outbase = args.outbase
    if not outbase:
        if args.areas:
            outbase = str(args.areas[0].with_suffix(''))
        elif args.circle_dir:
            outbase = str(args.circle_dir / args.circle_dir.name)

    main(args.areas, outbase, split_levels=args.split_levels, fmt=args.format, circle_dir=args.circle_dir)

if __name__ == '__main__':
    cli()
