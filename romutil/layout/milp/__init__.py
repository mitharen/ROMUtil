"""MILP solvers and modular pipeline stages for layout optimization."""

from __future__ import annotations

from romutil.layout.milp.clearance import (
    BoundaryExitClearanceBox,
    ChildSubClearanceBox,
    ClearanceConstraintBuilder,
    ClearanceEntity,
    ClearanceViolation,
    CorridorClearanceBox,
    RoomClearanceBox,
    detect_cluster_collisions,
)
from romutil.layout.milp.cluster import (
    ClusterSolvePhase,
    HierarchicalClusterSolver,
    solve_cluster,
    solve_cluster_with_cutting_planes,
)
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.debug import (
    render_cluster_solve_svg,
    render_debug_solve_tree_index,
)
from romutil.layout.milp.formulation import (
    ClusterFormulation,
    calculate_dynamic_bounds,
    formulate_cluster_model,
)
from romutil.layout.milp.inventory import (
    ActiveEdge,
    ClusterInventory,
    GatewayCutHandler,
    InterChildBridgeHandler,
    SplitPairHandler,
    derive_child_sub_boxes,
    inventory_cluster,
)
from romutil.layout.milp.normalization import (
    normalize_cluster_solution,
    pack_world_root_components,
)
from romutil.layout.milp.solver import (
    SolvedClusterExtraction,
    solve_and_extract,
)
from romutil.layout.milp.tree import HierarchicalTreeSolver
from romutil.types import LayoutSolverError

__all__ = [
    "ActiveEdge",
    "BoundaryExitClearanceBox",
    "ChildSubClearanceBox",
    "ClearanceConstraintBuilder",
    "ClearanceEntity",
    "ClearanceViolation",
    "ClusterFormulation",
    "ClusterInventory",
    "ClusterSolvePhase",
    "CorridorClearanceBox",
    "GatewayCutHandler",
    "HierarchicalClusterSolver",
    "HierarchicalTreeSolver",
    "InterChildBridgeHandler",
    "LayoutSolverError",
    "RoomClearanceBox",
    "SolvedClusterExtraction",
    "SplitPairHandler",
    "add_disjunctive_box_separation",
    "calculate_dynamic_bounds",
    "derive_child_sub_boxes",
    "detect_cluster_collisions",
    "formulate_cluster_model",
    "inventory_cluster",
    "normalize_cluster_solution",
    "pack_world_root_components",
    "render_cluster_solve_svg",
    "render_debug_solve_tree_index",
    "solve_and_extract",
    "solve_cluster",
    "solve_cluster_with_cutting_planes",
]
