import argparse
import logging
import os
from pathlib import Path
import sys

import networkx as nx
import pyomo.common.config

from romutil.models import Room, AreaData
from romutil.parser import Parser
from romutil.graph import graph

logging.basicConfig()
log = logging.getLogger('Mapper')

def main(area_files, outbase, split_levels=False):
    rdb = {}
    area_meta = None

    for area_file in area_files:
        # Support both open file objects and Path / string paths
        if hasattr(area_file, 'read'):
            content = area_file.read()
        else:
            with open(area_file, 'r', encoding='latin-1', errors='replace') as f:
                content = f.read()

        parser = Parser()
        try:
            area = parser.parse(content)
        except Exception as e:
            log.error(e)
            continue

        if isinstance(area, AreaData):
            area_meta = area.header
            rooms = list(area.rooms)
            rdb.update({r.vnum: Room(r) for r in rooms})
        else:
            rooms = []
            for section in area:
                if not section:
                    continue
                if section[0] == '#ROOMS':
                    rooms = section[1]
                elif section[0] == '#AREA':
                    area_meta = section[1]

            rdb.update({(r.vnum if hasattr(r, 'vnum') else r[0]): Room(r) for r in rooms})

    if not rdb:
        log.error('No rooms to plot.')
        sys.exit(0)

    edges = [
        (e.src, e.dst)
        for r in rdb.values()
        for e in r.exits
        if e.src in rdb.keys() and e.dst in rdb.keys()
    ]
    g = nx.DiGraph(edges)

    components = list(nx.connected_components(g.to_undirected()))
    for i, sub_graph in enumerate(components):
        sub_rdb = {node: rdb[node] for node in sub_graph}
        if not split_levels:
            graph(sub_rdb, f'{outbase}{i}.svg', area_meta, split_levels=False)
        else:
            base_name = outbase if len(components) == 1 else f'{outbase}{i}'
            graph(sub_rdb, f'{base_name}.svg', area_meta, split_levels=True, outbase=base_name)

    sys.exit(0)

def cli():
    parser = argparse.ArgumentParser(description='ROM MUD Area Mapper and 3D Visualizer')
    parser.add_argument('areas', nargs='+', type=Path, help='.ARE files for parsing')
    parser.add_argument('-outbase', help='output base name')
    parser.add_argument('--split-levels', action='store_true', help='Output separate SVGs for each distinct elevation plane')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug info')
    args = parser.parse_args()

    pyomo.common.config.logger.setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)

    outbase = args.outbase
    if not outbase:
        outbase = str(args.areas[0].with_suffix(''))

    main(args.areas, outbase, split_levels=args.split_levels)

if __name__ == '__main__':
    cli()
