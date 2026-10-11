"""MILP solvers and modular pipeline stages for layout optimization."""

from __future__ import annotations

from romutil.layout.milp.clearance import apply_clearance_constraints
from romutil.layout.milp.cluster import (
    ActiveEdge,
    HierarchicalClusterSolver,
    HierarchicalTreeSolver,
    solve_cluster,
)
from romutil.layout.milp.constraints import add_disjunctive_box_separation
from romutil.layout.milp.formulation import (
    ClusterFormulation,
    formulate_cluster_model,
)
from romutil.layout.milp.inventory import (
    ClusterInventory,
    derive_child_sub_boxes,
    inventory_cluster,
)
from romutil.layout.milp.normalization import normalize_cluster_solution
from romutil.layout.milp.solver import (
    SolvedClusterExtraction,
    solve_and_extract,
)
from romutil.types import LayoutSolverError

__all__ = [
    "ActiveEdge",
    "ClusterFormulation",
    "ClusterInventory",
    "HierarchicalClusterSolver",
    "HierarchicalTreeSolver",
    "LayoutSolverError",
    "SolvedClusterExtraction",
    "add_disjunctive_box_separation",
    "apply_clearance_constraints",
    "derive_child_sub_boxes",
    "formulate_cluster_model",
    "inventory_cluster",
    "normalize_cluster_solution",
    "solve_and_extract",
    "solve_cluster",
]
