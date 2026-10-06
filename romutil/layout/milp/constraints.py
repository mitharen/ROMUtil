"""Modular MILP constraint builders for 3D layout optimization."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pyomo.environ as pyo


def add_disjunctive_box_separation(
    model: pyo.ConcreteModel,
    pairs: Sequence[tuple[Any, Any]],
    get_box_a: Callable[[Any], tuple[Any, Any, Any, Any, Any, Any]],
    get_box_b: Callable[[Any], tuple[Any, Any, Any, Any, Any, Any]],
    prefix: str,
    m_big: int,
) -> None:
    """Enforce pairwise 3D axis-aligned disjunctive separation between bounding boxes.

    For each pair (item_a, item_b), creates 6 binary decision variables representing
    the 6 directional non-overlap half-spaces (right, left, ahead/north, behind/south,
    up, down). At least one direction must be active:
        b_r + b_l + b_a + b_b + b_u + b_d >= 1

    Directional Big-M constraints enforce coordinate separation when the respective
    binary indicator is 1:
        Right:  A_min_x - B_max_x >= 1 - 2M(1 - b_r)
        Left:   B_min_x - A_max_x >= 1 - 2M(1 - b_l)
        Ahead:  A_min_y - B_max_y >= 1 - 2M(1 - b_a)
        Behind: B_min_y - A_max_y >= 1 - 2M(1 - b_b)
        Up:     A_min_z - B_max_z >= 1 - 2M(1 - b_u)
        Down:   B_min_z - A_max_z >= 1 - 2M(1 - b_d)

    Args:
        model: Pyomo ConcreteModel instance to which variables and constraints are attached.
        pairs: Sequence of 2-tuples representing the items to separate. If empty, the function
            returns immediately without attaching empty variables or constraints.
        get_box_a: Callable mapping item_a to (min_x, max_x, min_y, max_y, min_z, max_z).
        get_box_b: Callable mapping item_b to (min_x, max_x, min_y, max_y, min_z, max_z).
        prefix: Naming prefix for the decision variables and constraint lists on the model.
        m_big: Big-M relaxation constant, typically twice the coordinate bounding domain.
    """
    if not pairs:
        return

    pair_list = list(pairs)
    clean_prefix = prefix.rstrip("_") if prefix else ""

    var_r = pyo.Var(pair_list, domain=pyo.Binary)
    var_l = pyo.Var(pair_list, domain=pyo.Binary)
    var_a = pyo.Var(pair_list, domain=pyo.Binary)
    var_b = pyo.Var(pair_list, domain=pyo.Binary)
    var_u = pyo.Var(pair_list, domain=pyo.Binary)
    var_d = pyo.Var(pair_list, domain=pyo.Binary)

    sep_sum = pyo.ConstraintList()
    sep_r = pyo.ConstraintList()
    sep_l = pyo.ConstraintList()
    sep_a = pyo.ConstraintList()
    sep_b = pyo.ConstraintList()
    sep_u = pyo.ConstraintList()
    sep_d = pyo.ConstraintList()

    if clean_prefix:
        setattr(model, f"{clean_prefix}_br", var_r)
        setattr(model, f"{clean_prefix}_bl", var_l)
        setattr(model, f"{clean_prefix}_ba", var_a)
        setattr(model, f"{clean_prefix}_bb", var_b)
        setattr(model, f"{clean_prefix}_bu", var_u)
        setattr(model, f"{clean_prefix}_bd", var_d)

        sep_base = clean_prefix if clean_prefix.endswith("sep") else f"{clean_prefix}_sep"
        setattr(model, f"{sep_base}_sum", sep_sum)
        setattr(model, f"{sep_base}_r", sep_r)
        setattr(model, f"{sep_base}_l", sep_l)
        setattr(model, f"{sep_base}_a", sep_a)
        setattr(model, f"{sep_base}_b", sep_b)
        setattr(model, f"{sep_base}_u", sep_u)
        setattr(model, f"{sep_base}_d", sep_d)
    else:
        setattr(model, "b_r", var_r)
        setattr(model, "b_l", var_l)
        setattr(model, "b_a", var_a)
        setattr(model, "b_b", var_b)
        setattr(model, "b_u", var_u)
        setattr(model, "b_d", var_d)

        setattr(model, "sep_sum", sep_sum)
        setattr(model, "sep_r", sep_r)
        setattr(model, "sep_l", sep_l)
        setattr(model, "sep_a", sep_a)
        setattr(model, "sep_b", sep_b)
        setattr(model, "sep_u", sep_u)
        setattr(model, "sep_d", sep_d)

    for item_a, item_b in pair_list:
        min_xa, max_xa, min_ya, max_ya, min_za, max_za = get_box_a(item_a)
        min_xb, max_xb, min_yb, max_yb, min_zb, max_zb = get_box_b(item_b)

        br = var_r[item_a, item_b]
        bl = var_l[item_a, item_b]
        ba = var_a[item_a, item_b]
        bb = var_b[item_a, item_b]
        bu = var_u[item_a, item_b]
        bd = var_d[item_a, item_b]

        sep_sum.add(br + bl + ba + bb + bu + bd >= 1)
        sep_r.add(min_xa - max_xb >= 1 - 2 * m_big * (1 - br))
        sep_l.add(min_xb - max_xa >= 1 - 2 * m_big * (1 - bl))
        sep_a.add(min_ya - max_yb >= 1 - 2 * m_big * (1 - ba))
        sep_b.add(min_yb - max_ya >= 1 - 2 * m_big * (1 - bb))
        sep_u.add(min_za - max_zb >= 1 - 2 * m_big * (1 - bu))
        sep_d.add(min_zb - max_za >= 1 - 2 * m_big * (1 - bd))
