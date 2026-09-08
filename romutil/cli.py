import argparse
import logging
import os
from pathlib import Path
import sys

import networkx as nx
import pyomo.common.config

from romutil.models import Room, AreaData, AreaHeader
from romutil.parser import Parser
from romutil.graph import graph, solve_layout
from romutil.exporter import build_area_json, export_json, export_html

logging.basicConfig()
log = logging.getLogger('Mapper')

def main(area_files, outbase, split_levels=False, fmt="svg"):
    rdb = {}
    area_meta = None
    first_file_name = "area.are"

    for area_file in area_files:
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
    parser.add_argument('areas', nargs='+', type=Path, help='.ARE files for parsing')
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

    pyomo.common.config.logger.setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)

    outbase = args.outbase
    if not outbase:
        outbase = str(args.areas[0].with_suffix(''))

    main(args.areas, outbase, split_levels=args.split_levels, fmt=args.format)

if __name__ == '__main__':
    cli()
