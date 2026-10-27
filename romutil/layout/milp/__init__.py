"""MILP solvers and modular pipeline stages for layout optimization."""

from __future__ import annotations

from romutil.layout.milp.clearance import (
    ClearanceConstraintBuilder,
    ClearanceViolation,
    apply_clearance_constraints,
    detect_cluster_collisions,
)
from romutil.layout.milp.cluster import (
    HierarchicalClusterSolver,
    solve_cluster,
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
    "ClearanceConstraintBuilder",
    "ClearanceViolation",
    "ClusterFormulation",
    "ClusterInventory",
    "GatewayCutHandler",
    "HierarchicalClusterSolver",
    "HierarchicalTreeSolver",
    "InterChildBridgeHandler",
    "LayoutSolverError",
    "SolvedClusterExtraction",
    "SplitPairHandler",
    "add_disjunctive_box_separation",
    "apply_clearance_constraints",
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
]
