"""Grader for `num-fix-wrong-topo-order-breaking-grads`.

Oracle: central finite differences of the plain-Python function `_f`,
which is a hand-written, ordinary-float mirror of the exact same
expression graph built with `sol.Value` in `_build_graph`. This makes the
oracle fully independent of the candidate's autograd engine — it never
calls `sol.Value` at all.
"""
from __future__ import annotations

import numpy as np

from mlsys import scorers


def _build_graph(xs):
    """Build a small diamond-shaped tape: x1 and `a` are each reused
    more than once, at different depths, which is exactly what a wrong
    (non-reverse) topological replay order corrupts."""
    a = xs[0] * xs[1]
    b = a + xs[2]
    c = a * b
    d = c + xs[3]
    e = d * xs[1]
    out = e + a
    return out


def _f(x0, x1, x2, x3):
    """Plain-float mirror of `_build_graph`, used only for the finite-
    difference oracle — completely independent of `sol.Value`."""
    a = x0 * x1
    b = a + x2
    c = a * b
    d = c + x3
    e = d * x1
    out = e + a
    return out


def _numeric_grad(xvals, h=1e-5):
    g = []
    for i in range(len(xvals)):
        xp = list(xvals)
        xp[i] += h
        xm = list(xvals)
        xm[i] -= h
        g.append((_f(*xp) - _f(*xm)) / (2.0 * h))
    return np.array(g, dtype=np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    rel_errs = []
    fwd_errs = []

    for _ in range(6):
        xvals = list(rng.uniform(0.3, 2.0, size=4) * rng.choice([-1.0, 1.0], size=4))

        try:
            xs = [sol.Value(v) for v in xvals]
            out = _build_graph(xs)
            out.backward()
            analytic = np.array([float(x.grad) for x in xs], dtype=np.float64)
            got_val = float(out.data)
        except Exception:
            return {"rel_err": float("inf"), "forward_max_abs_err": float("inf")}

        if not np.all(np.isfinite(analytic)) or not np.isfinite(got_val):
            return {"rel_err": float("inf"), "forward_max_abs_err": float("inf")}

        expected_val = _f(*xvals)
        fwd_errs.append(abs(got_val - expected_val))

        numeric = _numeric_grad(xvals)
        rel_errs.append(scorers.rel_err(numeric, analytic))

    return {
        "rel_err": float(np.mean(rel_errs)),
        "forward_max_abs_err": float(max(fwd_errs)),
    }
