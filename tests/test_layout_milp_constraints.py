"""Unit tests for modular MILP constraint builders."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo
import pytest

from romutil.layout.milp.constraints import add_disjunctive_box_separation


class TestAddDisjunctiveBoxSeparation:
    """Tests for add_disjunctive_box_separation."""

    def test_empty_pairs_noop(self) -> None:
        """Empty sequence of pairs attaches no variables or constraints to the model."""
        model = pyo.ConcreteModel()
        add_disjunctive_box_separation(
            model=model,
            pairs=[],
            get_box_a=lambda _: (0, 0, 0, 0, 0, 0),
            get_box_b=lambda _: (0, 0, 0, 0, 0, 0),
            prefix="test",
            m_big=100,
        )
        assert not hasattr(model, "test_br")
        assert not hasattr(model, "test_bl")
        assert not hasattr(model, "test_sep_sum")
        assert not hasattr(model, "b_r")

    def test_variable_and_constraint_creation_with_prefix(self) -> None:
        """Variables and constraint lists are named cleanly with the specified prefix."""
        model = pyo.ConcreteModel()
        pairs = [("box_a", "box_b")]

        add_disjunctive_box_separation(
            model=model,
            pairs=pairs,
            get_box_a=lambda _: (0, 1, 0, 1, 0, 1),
            get_box_b=lambda _: (2, 3, 2, 3, 2, 3),
            prefix="test_box",
            m_big=50,
        )

        for suffix in ("br", "bl", "ba", "bb", "bu", "bd"):
            assert hasattr(model, f"test_box_{suffix}")
            var = getattr(model, f"test_box_{suffix}")
            assert ("box_a", "box_b") in var

        for suffix in ("sum", "r", "l", "a", "b", "u", "d"):
            assert hasattr(model, f"test_box_sep_{suffix}")
            clist = getattr(model, f"test_box_sep_{suffix}")
            assert len(clist) == 1

    def test_variable_and_constraint_creation_without_prefix(self) -> None:
        """Omitting the prefix defaults to standard variable and constraint list names."""
        model = pyo.ConcreteModel()
        pairs = [(1, 2)]

        add_disjunctive_box_separation(
            model=model,
            pairs=pairs,
            get_box_a=lambda _: (0, 0, 0, 0, 0, 0),
            get_box_b=lambda _: (1, 1, 1, 1, 1, 1),
            prefix="",
            m_big=100,
        )

        for var_name in ("b_r", "b_l", "b_a", "b_b", "b_u", "b_d"):
            assert hasattr(model, var_name)
            assert (1, 2) in getattr(model, var_name)

        for clist_name in ("sep_sum", "sep_r", "sep_l", "sep_a", "sep_b", "sep_u", "sep_d"):
            assert hasattr(model, clist_name)
            assert len(getattr(model, clist_name)) == 1

    def test_prefix_trailing_underscore_sanitization(self) -> None:
        """Prefixes containing trailing underscores do not produce double underscores."""
        model = pyo.ConcreteModel()
        pairs = [("u", "v")]

        add_disjunctive_box_separation(
            model=model,
            pairs=pairs,
            get_box_a=lambda _: (0, 0, 0, 0, 0, 0),
            get_box_b=lambda _: (1, 1, 1, 1, 1, 1),
            prefix="parent_room_",
            m_big=100,
        )

        assert hasattr(model, "parent_room_br")
        assert not hasattr(model, "parent_room__br")
        assert hasattr(model, "parent_room_sep_sum")
        assert not hasattr(model, "parent_room__sep_sum")

    def test_prefix_named_sep(self) -> None:
        """Prefix 'sep' maps constraint lists to sep_sum and sep_r cleanly."""
        model = pyo.ConcreteModel()
        pairs = [(10, 20)]

        add_disjunctive_box_separation(
            model=model,
            pairs=pairs,
            get_box_a=lambda _: (0, 0, 0, 0, 0, 0),
            get_box_b=lambda _: (1, 1, 1, 1, 1, 1),
            prefix="sep",
            m_big=100,
        )

        assert hasattr(model, "sep_br")
        assert hasattr(model, "sep_sum")
        assert hasattr(model, "sep_r")

    def test_point_separation_optimal_solve(self) -> None:
        """Solving with disjunctive separation forces distinct integer positions."""
        solver = pyo.SolverFactory("cbc")
        if not solver.available():
            pytest.skip("CBC solver not available")

        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2], domain=pyo.Integers, bounds=(0, 10))
        model.y = pyo.Var([1, 2], domain=pyo.Integers, bounds=(0, 10))
        model.z = pyo.Var([1, 2], domain=pyo.Integers, bounds=(0, 10))

        # Anchor point 1 at the origin
        model.x[1].fix(0)
        model.y[1].fix(0)
        model.z[1].fix(0)

        add_disjunctive_box_separation(
            model=model,
            pairs=[(1, 2)],
            get_box_a=lambda u: (model.x[u], model.x[u], model.y[u], model.y[u], model.z[u], model.z[u]),
            get_box_b=lambda v: (model.x[v], model.x[v], model.y[v], model.y[v], model.z[v], model.z[v]),
            prefix="pts",
            m_big=100,
        )

        model.obj = pyo.Objective(expr=model.x[2] + model.y[2] + model.z[2], sense=pyo.minimize)
        solver.solve(model)

        p1 = (pyo.value(model.x[1]), pyo.value(model.y[1]), pyo.value(model.z[1]))
        p2 = (pyo.value(model.x[2]), pyo.value(model.y[2]), pyo.value(model.z[2]))

        assert p1 == (0, 0, 0)
        # Point 2 must be separated from point 1 by at least 1 along one coordinate axis
        manhattan_dist = sum(abs(p2[i] - p1[i]) for i in range(3))
        assert manhattan_dist >= 1.0

    def test_box_to_box_interval_separation(self) -> None:
        """Separation between non-trivial bounding boxes prevents interval penetration."""
        solver = pyo.SolverFactory("cbc")
        if not solver.available():
            pytest.skip("CBC solver not available")

        model = pyo.ConcreteModel()
        model.x = pyo.Var(domain=pyo.Integers, bounds=(0, 20))
        model.y = pyo.Var(domain=pyo.Integers, bounds=(0, 20))
        model.z = pyo.Var(domain=pyo.Integers, bounds=(0, 20))

        # Box A: [0, 3] x [0, 3] x [0, 3]
        # Box B: [x, x+2] x [y, y+2] x [z, z+2]
        def box_a(_: Any) -> tuple[int, int, int, int, int, int]:
            return (0, 3, 0, 3, 0, 3)

        def box_b(_: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
            return (model.x, model.x + 2, model.y, model.y + 2, model.z, model.z + 2)

        add_disjunctive_box_separation(
            model=model,
            pairs=[("a", "b")],
            get_box_a=box_a,
            get_box_b=box_b,
            prefix="bbox",
            m_big=200,
        )

        model.obj = pyo.Objective(expr=model.x + model.y + model.z, sense=pyo.minimize)
        solver.solve(model)

        bx = pyo.value(model.x)
        by = pyo.value(model.y)
        bz = pyo.value(model.z)

        # Box B must be completely outside Box A: at least one coordinate >= 4 (since Box A max is 3)
        assert (bx >= 4) or (by >= 4) or (bz >= 4)

    def test_multiple_pairwise_separations(self) -> None:
        """Multiple pairs generate separate decision variables for every pair."""
        model = pyo.ConcreteModel()
        pairs = [(1, 2), (1, 3), (2, 3)]

        add_disjunctive_box_separation(
            model=model,
            pairs=pairs,
            get_box_a=lambda _: (0, 0, 0, 0, 0, 0),
            get_box_b=lambda _: (1, 1, 1, 1, 1, 1),
            prefix="multi",
            m_big=100,
        )

        for p in pairs:
            assert p in model.multi_br
            assert p in model.multi_bl
            assert p in model.multi_ba
            assert p in model.multi_bb
            assert p in model.multi_bu
            assert p in model.multi_bd

        assert len(model.multi_sep_sum) == 3
        assert len(model.multi_sep_r) == 3
        assert len(model.multi_sep_l) == 3
        assert len(model.multi_sep_a) == 3
        assert len(model.multi_sep_b) == 3
        assert len(model.multi_sep_u) == 3
        assert len(model.multi_sep_d) == 3
