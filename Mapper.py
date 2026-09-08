#!/usr/bin/env python
"""
Backward-compatible wrapper for romutil package.
"""
from romutil.models import Direction, direction_matrix, Room, Exit
from romutil.plotter import Plotter
from romutil.solver import non_euler, solve
from romutil.graph import restore_rooms, mfas, graph
from romutil.cli import main, cli, log

if __name__ == '__main__':
    cli()
