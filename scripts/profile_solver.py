#!/usr/bin/env python3
"""
scripts/profile_solver.py - Standalone profiling utility for ROMUtil's layout solver.

Instruments and quantifies execution time, memory/variable footprints, and solver
termination states across every phase of the room layout pipeline:
  1. Parsing and graph construction
  2. 2-Degree corridor condensation
  3. Boundary dummy room allocation
  4. Non-Eulerian cut minimization (MIP)
  5. Layout Pyomo MILP formulation
  6. Initial CBC relaxation solve
  7. Iterative collision detection (O(E^2) candidate scanning)
  8. Collision-resolution CBC solves (tracking binary variable counts & solver termination)
  9. Coordinate restoration & normalization

Can be run as a standalone CLI or imported for testing and automated benchmarks.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import itertools
import json
import logging
from math import sqrt
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pyomo.opt
from pyomo.environ import (
    Binary,
    Boolean,
    ConcreteModel,
    ConstraintList,
    Integers,
    NonNegativeIntegers,
    Objective,
    Param,
    PositiveIntegers,
    RangeSet,
    Set,
    SolverFactory,
    Var,
    VarList,
)

from romutil.models import AreaData, AreaHeader, Direction, Exit, ExitDef, Room, RoomDef
from romutil.parser import Parser, parse_file

log = logging.getLogger("SolverProfiler")

DEFAULT_CANDIDATE_DIRS = [
    os.environ.get("QUICKMUD_AREA_DIR", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../QuickMUD/area")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../QuickMUD/area")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../areas")),
]


@dataclass
class IterationMetrics:
    """Metrics for a single collision-detection & resolution iteration."""
    iteration: int
    overlap_detection_time_sec: float
    candidates_checked: int
    overlaps_found: int
    constraints_added: int
    binary_vars_added: int
    total_binary_vars: int
    total_constraints: int
    solve_time_sec: float
    termination_condition: str
    solver_status: str


@dataclass
class StageTimings:
    """Granular timing breakdown for the layout pipeline."""
    parsing_sec: float = 0.0
    graph_construction_sec: float = 0.0
    condensation_sec: float = 0.0
    dummy_allocation_sec: float = 0.0
    non_euler_formulation_sec: float = 0.0
    non_euler_solve_sec: float = 0.0
    layout_formulation_sec: float = 0.0
    initial_solve_sec: float = 0.0
    overlap_detection_total_sec: float = 0.0
    collision_solves_total_sec: float = 0.0
    coordinate_restoration_sec: float = 0.0
    total_pipeline_sec: float = 0.0


@dataclass
class SolverProfileResult:
    """Complete profiling and benchmarking results for an area layout solve."""
    area_name: str
    area_path: str
    original_rooms: int
    original_exits: int
    condensed_rooms: int
    active_rooms: int
    dummy_rooms: int
    model_exits: int
    one_way_exits: int
    initial_candidate_pairs: int
    initial_binary_vars: int
    initial_constraints: int
    initial_termination_condition: str
    timings: StageTimings = field(default_factory=StageTimings)
    iterations: List[IterationMetrics] = field(default_factory=list)
    total_collision_iterations: int = 0
    final_termination_condition: str = "UNKNOWN"
    success: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to JSON-serializable dictionary."""
        return asdict(self)


def find_area_file(filename_or_path: Union[str, Path], search_dir: Optional[Union[str, Path]] = None) -> Path:
    """Locate an area file given a name or explicit path."""
    p = Path(filename_or_path)
    if p.is_file():
        return p.resolve()

    candidate_dirs = []
    if search_dir:
        candidate_dirs.append(Path(search_dir))
    for d in DEFAULT_CANDIDATE_DIRS:
        if d:
            candidate_dirs.append(Path(d))

    for d in candidate_dirs:
        cand = d / p.name
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(f"Area file '{filename_or_path}' not found in candidate paths.")


def profile_area(
    area_path: Union[str, Path],
    cbc_sec_limit: int = 300,
    max_iterations: Optional[int] = None,
    non_euler_sec_limit: int = 20,
    search_dir: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> SolverProfileResult:
    """
    Instruments every stage of layout computation for an area file and returns
    structured performance metrics.
    """
    resolved_path = find_area_file(area_path, search_dir=search_dir)
    pipeline_t0 = time.perf_counter()

    timings = StageTimings()

    # Stage 1: Parsing
    t0 = time.perf_counter()
    area_data = parse_file(str(resolved_path))
    timings.parsing_sec = time.perf_counter() - t0

    area_header = area_data.header
    area_name = area_header.name if area_header and area_header.name else resolved_path.stem

    # Stage 2: Graph construction
    t0 = time.perf_counter()
    rdb: Dict[int, Room] = {r.vnum: Room(r) for r in area_data.rooms}
    original_rooms_count = len(rdb)
    original_exits_count = sum(len(r.exits) for r in rdb.values())
    original_exits_backup = {vnum: copy.deepcopy(r.exits) for vnum, r in rdb.items()}
    timings.graph_construction_sec = time.perf_counter() - t0

    # Stage 3: 2-Degree corridor condensation
    t0 = time.perf_counter()
    condensed_rooms_count = 0
    for vnum, r in list(rdb.items()):
        if len(r.exits) == 2:
            if not all(e.dst in rdb.keys() for e in r.exits):
                continue
            if not all([e in rdb[e.dst].exits for e in r.exits]):
                continue
            if r.exits[0].direction == r.exits[1].direction.invert():
                rdb[r.exits[0].dst].replace_exit(vnum, r.exits[1].dst, r.exits[1].distance)
                rdb[r.exits[1].dst].replace_exit(vnum, r.exits[0].dst, r.exits[0].distance)
                rdb[r.exits[0].dst].fixups.append((r, r.exits[0].direction.invert(), r.exits[0].distance))
                del rdb[vnum]
                condensed_rooms_count += 1
    timings.condensation_sec = time.perf_counter() - t0

    # Stage 4: Dummy room allocation
    t0 = time.perf_counter()
    exits: List[Exit] = list(set([e for r in rdb.values() for e in r.exits]))
    dummy_rooms_count = 0
    for e in exits:
        if e.dst == -1:
            e.dst = max(rdb.keys()) + 1
        if e.dst not in rdb:
            rdb[e.dst] = Room(RoomDef(vnum=e.dst, name="", description="", exits=()))
            rdb[e.dst].exits.append(e)
            rdb[e.dst].dummy = True
            dummy_rooms_count += 1
    timings.dummy_allocation_sec = time.perf_counter() - t0

    active_rooms_count = len(rdb) - dummy_rooms_count

    if not exits:
        timings.total_pipeline_sec = time.perf_counter() - pipeline_t0
        return SolverProfileResult(
            area_name=area_name,
            area_path=str(resolved_path),
            original_rooms=original_rooms_count,
            original_exits=original_exits_count,
            condensed_rooms=condensed_rooms_count,
            active_rooms=active_rooms_count,
            dummy_rooms=dummy_rooms_count,
            model_exits=0,
            one_way_exits=0,
            initial_candidate_pairs=0,
            initial_binary_vars=0,
            initial_constraints=0,
            initial_termination_condition="trivial_empty",
            timings=timings,
            iterations=[],
            total_collision_iterations=0,
            final_termination_condition="trivial_empty",
            success=True,
        )

    # Filter self-loops
    clean_exits = [e for e in exits if e.src != e.dst]

    # Stage 5: Non-Eulerian formulation and solve
    t0 = time.perf_counter()
    m_euler = ConcreteModel()
    m_euler.Rooms = Set(initialize=rdb.keys())
    m_euler.Exits = RangeSet(0, len(clean_exits) - 1)
    m_euler.Directions = RangeSet(0, Direction.mod.value - 1)
    m_euler.M = Param(initialize=sum([e.distance for e in clean_exits]))
    m_euler.x = Var(m_euler.Rooms, within=Integers, bounds=(0, m_euler.M))
    m_euler.y = Var(m_euler.Rooms, within=Integers, bounds=(0, m_euler.M))
    m_euler.z = Var(m_euler.Rooms, within=Integers, bounds=(0, m_euler.M))
    m_euler.cut = Var(m_euler.Exits, within=Binary)
    m_euler.relative_pos = ConstraintList()

    one_way_exits = [e for e in clean_exits if e not in rdb[e.dst].exits]

    for i, e in enumerate(clean_exits):
        if e in one_way_exits:
            continue
        if e.dst != e.src:
            if e.direction not in (Direction.east, Direction.west):
                m_euler.relative_pos.add(m_euler.x[e.src] - m_euler.x[e.dst] + m_euler.M * m_euler.cut[i] >= 0)
                m_euler.relative_pos.add(m_euler.x[e.dst] - m_euler.x[e.src] + m_euler.M * m_euler.cut[i] >= 0)
            if e.direction not in (Direction.north, Direction.south):
                m_euler.relative_pos.add(m_euler.y[e.src] - m_euler.y[e.dst] + m_euler.M * m_euler.cut[i] >= 0)
                m_euler.relative_pos.add(m_euler.y[e.dst] - m_euler.y[e.src] + m_euler.M * m_euler.cut[i] >= 0)
            if e.direction not in (Direction.up, Direction.down):
                m_euler.relative_pos.add(m_euler.z[e.src] - m_euler.z[e.dst] + m_euler.M * m_euler.cut[i] >= 0)
                m_euler.relative_pos.add(m_euler.z[e.dst] - m_euler.z[e.src] + m_euler.M * m_euler.cut[i] >= 0)

            if e.direction == Direction.east:
                m_euler.relative_pos.add(m_euler.x[e.dst] - m_euler.x[e.src] + m_euler.M * m_euler.cut[i] >= 1)
            elif e.direction == Direction.north:
                m_euler.relative_pos.add(m_euler.y[e.dst] - m_euler.y[e.src] + m_euler.M * m_euler.cut[i] >= 1)
            elif e.direction == Direction.up:
                m_euler.relative_pos.add(m_euler.z[e.dst] - m_euler.z[e.src] + m_euler.M * m_euler.cut[i] >= 1)
            elif e.direction == Direction.west:
                m_euler.relative_pos.add(m_euler.x[e.src] - m_euler.x[e.dst] + m_euler.M * m_euler.cut[i] >= 1)
            elif e.direction == Direction.south:
                m_euler.relative_pos.add(m_euler.y[e.src] - m_euler.y[e.dst] + m_euler.M * m_euler.cut[i] >= 1)
            elif e.direction == Direction.down:
                m_euler.relative_pos.add(m_euler.z[e.src] - m_euler.z[e.dst] + m_euler.M * m_euler.cut[i] >= 1)

    m_euler.obj = Objective(expr=sum(m_euler.cut[i] for i in range(len(clean_exits))))
    timings.non_euler_formulation_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    euler_solver = SolverFactory("cbc")
    euler_solver.options["sec"] = non_euler_sec_limit
    euler_solver.solve(m_euler, tee=False)
    timings.non_euler_solve_sec = time.perf_counter() - t0

    for i in range(len(clean_exits)):
        if m_euler.cut[i].value:
            clean_exits[i].one_way = True

    # Handle disconnected rooms by adding a self/dummy exit
    for room in rdb.values():
        if not len(room.exits):
            e_stub = Exit(ExitDef(direction=0, dst_vnum=room.vnum), source=room.vnum)
            clean_exits.append(e_stub)

    # Stage 6: Pyomo Layout MILP formulation
    t0 = time.perf_counter()
    m = ConcreteModel()
    m.Rooms = Set(initialize=rdb.keys())
    m.Exits = RangeSet(0, len(clean_exits) - 1)
    m.Directions = RangeSet(0, Direction.mod.value - 1)
    m.M = Param(initialize=sum([e.distance for e in clean_exits]))
    m.d_min = Param(initialize=1)
    m.l_min = Param(m.Exits, initialize=lambda m, x: clean_exits[x].distance)

    m.x = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.y = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.z = Var(m.Rooms, within=Integers, bounds=(0, m.M))
    m.cut = Var(m.Exits, within=Binary)
    m.l_max = Var(m.Exits, within=PositiveIntegers)
    m.one_ways = VarList(within=NonNegativeIntegers, bounds=(0, m.M))

    m.relative_pos = ConstraintList()
    m.one_way_pos = ConstraintList()
    m.crossings = ConstraintList()

    one_ways_terms = []
    for e in clean_exits:
        if e not in rdb[e.dst].exits:
            e.one_way = True

    for i, x in enumerate(clean_exits):
        if x.one_way:
            x_off = m.d_min if x.direction == Direction.east else -m.d_min if x.direction == Direction.west else 0
            y_off = m.d_min if x.direction == Direction.north else -m.d_min if x.direction == Direction.south else 0
            z_off = m.d_min if x.direction == Direction.up else -m.d_min if x.direction == Direction.down else 0
            X = m.one_ways.add()
            Y = m.one_ways.add()
            Z = m.one_ways.add()
            x_diff = m.x[x.dst] - m.x[x.src] - x_off
            m.one_way_pos.add(x_diff <= X)
            m.one_way_pos.add(-x_diff <= X)
            y_diff = m.y[x.dst] - m.y[x.src] - y_off
            m.one_way_pos.add(y_diff <= Y)
            m.one_way_pos.add(-y_diff <= Y)
            z_diff = m.z[x.dst] - m.z[x.src] - z_off
            m.one_way_pos.add(z_diff <= Z)
            m.one_way_pos.add(-z_diff <= Z)
            one_ways_terms.append(X + Y + Z)
            continue

        if x.dst != x.src:
            if x.direction not in (Direction.east, Direction.west):
                m.relative_pos.add(m.x[x.src] + m.M * m.cut[i] >= m.x[x.dst])
                m.relative_pos.add(m.x[x.dst] + m.M * m.cut[i] >= m.x[x.src])
            if x.direction not in (Direction.north, Direction.south):
                m.relative_pos.add(m.y[x.src] + m.M * m.cut[i] >= m.y[x.dst])
                m.relative_pos.add(m.y[x.dst] + m.M * m.cut[i] >= m.y[x.src])
            if x.direction not in (Direction.up, Direction.down):
                m.relative_pos.add(m.z[x.src] + m.M * m.cut[i] >= m.z[x.dst])
                m.relative_pos.add(m.z[x.dst] + m.M * m.cut[i] >= m.z[x.src])

            if x.direction == Direction.east:
                m.relative_pos.add(m.x[x.dst] - m.x[x.src] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.x[x.dst] - m.x[x.src] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.north:
                m.relative_pos.add(m.y[x.dst] - m.y[x.src] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.y[x.dst] - m.y[x.src] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.up:
                m.relative_pos.add(m.z[x.dst] - m.z[x.src] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.z[x.dst] - m.z[x.src] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.west:
                m.relative_pos.add(m.x[x.src] - m.x[x.dst] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.x[x.src] - m.x[x.dst] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.south:
                m.relative_pos.add(m.y[x.src] - m.y[x.dst] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.y[x.src] - m.y[x.dst] - m.M * m.cut[i] <= m.l_max[i])
            elif x.direction == Direction.down:
                m.relative_pos.add(m.z[x.src] - m.z[x.dst] + m.M * m.cut[i] >= m.l_min[i])
                m.relative_pos.add(m.z[x.src] - m.z[x.dst] - m.M * m.cut[i] <= m.l_max[i])

    m.obj = Objective(
        expr=m.M * m.M * (sum(m.cut[i] for i in range(len(clean_exits))))
        + sum([m.l_max[e] for e in m.Exits])
        + sum([way for way in one_ways_terms])
    )
    timings.layout_formulation_sec = time.perf_counter() - t0

    # Initial variable counts
    initial_binary_vars = len(clean_exits)  # m.cut
    initial_constraints = len(m.relative_pos) + len(m.one_way_pos)

    # Initial overlap candidates computation (O(E^2))
    all_pairs = list(itertools.combinations(range(len(clean_exits)), 2))
    non_incidents = [
        pair
        for pair in all_pairs
        if clean_exits[pair[0]].src not in clean_exits[pair[1]]
        and clean_exits[pair[0]].dst not in clean_exits[pair[1]]
    ]
    non_incidents = [
        pair
        for pair in non_incidents
        if not clean_exits[pair[0]].one_way and not clean_exits[pair[1]].one_way
    ]
    initial_candidate_pairs_count = len(non_incidents)

    solver = SolverFactory("cbc", tee=False)
    solver.options["sec"] = cbc_sec_limit

    iterations_log: List[IterationMetrics] = []
    iteration_count = 0
    relations_count = 0
    current_total_binary = initial_binary_vars
    last_term_cond = "UNKNOWN"

    while True:
        solve_t0 = time.perf_counter()
        result = solver.solve(m, tee=False)
        solve_duration = time.perf_counter() - solve_t0
        term_cond = str(result.solver.termination_condition)
        solver_status = str(result.solver.status)
        last_term_cond = term_cond

        if iteration_count == 0:
            timings.initial_solve_sec = solve_duration
            initial_term_cond = term_cond
        else:
            timings.collision_solves_total_sec += solve_duration

        if result.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
            break

        # Check if max_iterations reached
        if max_iterations is not None and iteration_count >= max_iterations:
            break

        # Update room coordinates
        for vnum, room in list(rdb.items()):
            room.x = m.x[vnum].value if m.x[vnum].value is not None else 0
            room.y = m.y[vnum].value if m.y[vnum].value is not None else 0
            room.z = m.z[vnum].value if m.z[vnum].value is not None else 0

        # Scan for overlaps among non_incidents
        detect_t0 = time.perf_counter()
        added_constraints = 0
        overlaps_found = 0
        binary_vars_added = 0
        to_remove = []

        batch_target = int(sqrt(len(non_incidents))) if len(non_incidents) > 0 else 0

        for left, right in non_incidents:
            ex = clean_exits[left]
            nx = clean_exits[right]

            if m.cut[left].value or m.cut[right].value:
                continue
            if None in [
                m.x[ex.src].value, m.x[ex.dst].value, m.x[nx.src].value, m.x[nx.dst].value,
                m.y[ex.src].value, m.y[ex.dst].value, m.y[nx.src].value, m.y[nx.dst].value,
                m.z[ex.src].value, m.z[ex.dst].value, m.z[nx.src].value, m.z[nx.dst].value,
            ]:
                continue

            # 3D AABB overlap check
            if (
                max(m.x[ex.src].value, m.x[ex.dst].value) < min(m.x[nx.src].value, m.x[nx.dst].value)
                or min(m.x[ex.src].value, m.x[ex.dst].value) > max(m.x[nx.src].value, m.x[nx.dst].value)
                or max(m.y[ex.src].value, m.y[ex.dst].value) < min(m.y[nx.src].value, m.y[nx.dst].value)
                or min(m.y[ex.src].value, m.y[ex.dst].value) > max(m.y[nx.src].value, m.y[nx.dst].value)
                or max(m.z[ex.src].value, m.z[ex.dst].value) < min(m.z[nx.src].value, m.z[nx.dst].value)
                or min(m.z[ex.src].value, m.z[ex.dst].value) > max(m.z[nx.src].value, m.z[nx.dst].value)
            ):
                continue

            # Overlap detected!
            overlaps_found += 1
            rel_var = Var(m.Directions, within=Boolean)
            m.add_component(f"relation_{relations_count}", rel_var)
            relations_count += 1
            binary_vars_added += Direction.mod.value
            current_total_binary += Direction.mod.value

            m.crossings.add(sum([rel_var[j] for j in range(Direction.mod.value)]) >= 1)

            prod = list(itertools.product((ex.src, ex.dst), (nx.src, nx.dst)))
            for p, q in prod:
                m.crossings.add(m.x[p] - m.x[q] + m.M * ((1 - rel_var[Direction.east]) + m.cut[left] + m.cut[right]) >= m.d_min)
                m.crossings.add(m.y[p] - m.y[q] + m.M * ((1 - rel_var[Direction.north]) + m.cut[left] + m.cut[right]) >= m.d_min)
                m.crossings.add(m.z[p] - m.z[q] + m.M * ((1 - rel_var[Direction.up]) + m.cut[left] + m.cut[right]) >= m.d_min)
                m.crossings.add(m.x[q] - m.x[p] + m.M * ((1 - rel_var[Direction.west]) + m.cut[left] + m.cut[right]) >= m.d_min)
                m.crossings.add(m.y[q] - m.y[p] + m.M * ((1 - rel_var[Direction.south]) + m.cut[left] + m.cut[right]) >= m.d_min)
                m.crossings.add(m.z[q] - m.z[p] + m.M * ((1 - rel_var[Direction.down]) + m.cut[left] + m.cut[right]) >= m.d_min)

            to_remove.append((left, right))
            added_constraints += 1
            if added_constraints == batch_target:
                break

        for rem in to_remove:
            non_incidents.remove(rem)

        detect_duration = time.perf_counter() - detect_t0
        timings.overlap_detection_total_sec += detect_duration

        iter_metric = IterationMetrics(
            iteration=iteration_count,
            overlap_detection_time_sec=detect_duration,
            candidates_checked=len(non_incidents) + len(to_remove),
            overlaps_found=overlaps_found,
            constraints_added=added_constraints * 25,  # 1 + 24
            binary_vars_added=binary_vars_added,
            total_binary_vars=current_total_binary,
            total_constraints=initial_constraints + len(m.crossings),
            solve_time_sec=solve_duration,
            termination_condition=term_cond,
            solver_status=solver_status,
        )
        iterations_log.append(iter_metric)

        iteration_count += 1

        if not added_constraints:
            break

    # Stage 7: Coordinate restoration & normalization
    t0 = time.perf_counter()
    def _restore(r_node: Room) -> List[Room]:
        restored = []
        for r_sub, d_sub, dist in r_node.fixups:
            r_sub.x, r_sub.y, r_sub.z = r_node.x, r_node.y, r_node.z
            if d_sub == Direction.north:
                r_sub.y += dist
            elif d_sub == Direction.east:
                r_sub.x += dist
            elif d_sub == Direction.south:
                r_sub.y -= dist
            elif d_sub == Direction.west:
                r_sub.x -= dist
            elif d_sub == Direction.up:
                r_sub.z += dist
            elif d_sub == Direction.down:
                r_sub.z -= dist
            restored.append(r_sub)
            restored += _restore(r_sub)
        return restored

    for vnum, room in list(rdb.items()):
        for r_rest in _restore(room):
            rdb[r_rest.vnum] = r_rest

    valid_rooms = [r for r in rdb.values() if r.x is not None and r.y is not None and r.z is not None]
    if valid_rooms:
        x_min = min(r.x for r in valid_rooms)
        y_min = min(r.y for r in valid_rooms)
        z_min = min(r.z for r in valid_rooms)
        for r in valid_rooms:
            r.x -= x_min
            r.y -= y_min
            r.z -= z_min

    for vnum, orig_ex in original_exits_backup.items():
        if vnum in rdb:
            rdb[vnum].exits = orig_ex

    timings.coordinate_restoration_sec = time.perf_counter() - t0
    timings.total_pipeline_sec = time.perf_counter() - pipeline_t0

    success = (last_term_cond == str(pyomo.opt.TerminationCondition.optimal))

    return SolverProfileResult(
        area_name=area_name,
        area_path=str(resolved_path),
        original_rooms=original_rooms_count,
        original_exits=original_exits_count,
        condensed_rooms=condensed_rooms_count,
        active_rooms=active_rooms_count,
        dummy_rooms=dummy_rooms_count,
        model_exits=len(clean_exits),
        one_way_exits=len([e for e in clean_exits if e.one_way]),
        initial_candidate_pairs=initial_candidate_pairs_count,
        initial_binary_vars=initial_binary_vars,
        initial_constraints=initial_constraints,
        initial_termination_condition=iterations_log[0].termination_condition if iterations_log else "unknown",
        timings=timings,
        iterations=iterations_log,
        total_collision_iterations=len(iterations_log),
        final_termination_condition=last_term_cond,
        success=success,
    )


def format_profile_markdown_table(results: List[SolverProfileResult]) -> str:
    """Format multiple area profile results into an evergreen markdown table."""
    lines = [
        "| Area | Scale (Rooms / Exits) | Condensation (Rooms / Exits) | Dummy Rooms | Initial CBC Solve | Overlap Scan ($O(E^2)$) | Collision CBC Solves | Total Time | Termination |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        scale = f"{r.original_rooms} / {r.original_exits}"
        cond = f"-{r.condensed_rooms} -> {r.active_rooms} / {r.model_exits}"
        dummies = f"{r.dummy_rooms}"
        t_init = f"{r.timings.initial_solve_sec:.3f}s"
        t_detect = f"{r.timings.overlap_detection_total_sec:.3f}s ({r.initial_candidate_pairs} pairs)"
        t_col = f"{r.timings.collision_solves_total_sec:.3f}s ({r.total_collision_iterations} iters)"
        t_total = f"{r.timings.total_pipeline_sec:.3f}s"
        term = r.final_termination_condition
        lines.append(f"| `{r.area_name}` | {scale} | {cond} | {dummies} | {t_init} | {t_detect} | {t_col} | {t_total} | `{term}` |")
    return "\n".join(lines)


def format_stage_breakdown_table(results: List[SolverProfileResult]) -> str:
    """Format detailed timing breakdown per stage across profiled areas."""
    lines = [
        "| Area | Parsing | Graph & Condensation | Non-Euler MIP | Initial Layout MIP | Overlap Detection | Collision Solves | Restoration | Total Wall Clock |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        t = r.timings
        lines.append(
            f"| `{r.area_name}` | {t.parsing_sec * 1000:.1f}ms | {(t.graph_construction_sec + t.condensation_sec + t.dummy_allocation_sec) * 1000:.1f}ms | "
            f"{t.non_euler_solve_sec:.3f}s | {t.initial_solve_sec:.3f}s | {t.overlap_detection_total_sec:.3f}s | "
            f"{t.collision_solves_total_sec:.3f}s | {t.coordinate_restoration_sec * 1000:.1f}ms | {t.total_pipeline_sec:.3f}s |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ROMUtil Solver Pipeline Profiler & Benchmark")
    parser.add_argument("--areas", nargs="+", default=["smurf.are", "school.are", "midgaard.are"], help="Area files to profile")
    parser.add_argument("--areas-dir", type=Path, default=None, help="Directory containing .ARE files")
    parser.add_argument("--timeout", type=int, default=300, help="CBC solver timeout in seconds (default: 300)")
    parser.add_argument("--max-iterations", type=int, default=None, help="Limit collision-resolution iterations")
    parser.add_argument("--json-out", type=Path, default=None, help="Path to write JSON profile metrics")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Path to write Markdown report")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose stdout logging")
    args = parser.parse_args()

    results: List[SolverProfileResult] = []
    print(f"Profiling solver across {len(args.areas)} areas (timeout={args.timeout}s)...")

    for area_name in args.areas:
        print(f"\n--- Profiling {area_name} ---")
        try:
            res = profile_area(
                area_name,
                cbc_sec_limit=args.timeout,
                max_iterations=args.max_iterations,
                search_dir=args.areas_dir,
                quiet=args.quiet,
            )
            results.append(res)
            print(f"Finished {area_name}: total {res.timings.total_pipeline_sec:.2f}s (status={res.final_termination_condition})")
            print(f"  Rooms: {res.original_rooms} -> {res.active_rooms} (+{res.dummy_rooms} dummies), Exits: {res.model_exits}")
            print(f"  Initial CBC: {res.timings.initial_solve_sec:.3f}s | Overlap Detection: {res.timings.overlap_detection_total_sec:.3f}s | Collision Solves: {res.timings.collision_solves_total_sec:.3f}s")
        except Exception as e:
            print(f"Error profiling {area_name}: {e}", file=sys.stderr)

    if not results:
        print("No areas successfully profiled.", file=sys.stderr)
        sys.exit(1)

    table_md = format_profile_markdown_table(results)
    breakdown_md = format_stage_breakdown_table(results)

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY TABLE:")
    print("=" * 80)
    print(table_md)
    print("\n" + breakdown_md)

    if args.json_out:
        data = [r.to_dict() for r in results]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nSaved structured JSON metrics to {args.json_out}")

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        content = f"# Solver Profiling Report\n\n## Overview\n\n{table_md}\n\n## Stage Breakdown\n\n{breakdown_md}\n"
        args.markdown_out.write_text(content, encoding="utf-8")
        print(f"Saved Markdown report to {args.markdown_out}")


if __name__ == "__main__":
    main()
