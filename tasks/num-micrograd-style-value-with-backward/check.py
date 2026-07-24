"""Grader for `num-micrograd-style-value-with-backward`.

Two independent oracles, neither calling `sol.Value`:
  - forward value: the same expression evaluated with plain math.tanh/exp
    on ordinary Python floats (`_f`).
  - gradients: central finite differences of `_f` w.r.t. each of the 11
    leaf scalars.
"""
from __future__ import annotations

import math

import numpy as np

N_LEAVES = 11  # x1, x2, w1a, w1b, w2a, w2b, b1, b2, v1, v2, b3


def _f(xs):
    x1, x2, w1a, w1b, w2a, w2b, b1, b2, v1, v2, b3 = xs
    h1 = math.tanh(w1a * x1 + w1b * x2 + b1)
    h2 = math.tanh(w2a * x1 + w2b * x2 + b2)
    z = v1 * h1 + v2 * h2 + b3
    return math.exp(z)


def _numeric_grad(xvals, h=1e-5):
    g = []
    for i in range(len(xvals)):
        xp = list(xvals)
        xp[i] += h
        xm = list(xvals)
        xm[i] -= h
        g.append((_f(xp) - _f(xm)) / (2.0 * h))
    return np.array(g, dtype=np.float64)


def _build_graph(sol, xs):
    x1, x2, w1a, w1b, w2a, w2b, b1, b2, v1, v2, b3 = xs
    h1 = (w1a * x1 + w1b * x2 + b1).tanh()
    h2 = (w2a * x1 + w2b * x2 + b2).tanh()
    z = v1 * h1 + v2 * h2 + b3
    return z.exp()


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    grad_errs = []
    fwd_errs = []

    for _ in range(8):
        xvals = list(rng.uniform(-1.0, 1.0, size=N_LEAVES))

        try:
            xs = [sol.Value(v) for v in xvals]
            out = _build_graph(sol, xs)
            out.backward()
            analytic = np.array([float(x.grad) for x in xs], dtype=np.float64)
            got_val = float(out.data)
        except Exception:
            return {"max_abs_err": float("inf"), "forward_max_abs_err": float("inf")}

        if not np.all(np.isfinite(analytic)) or not np.isfinite(got_val):
            return {"max_abs_err": float("inf"), "forward_max_abs_err": float("inf")}

        expected_val = _f(xvals)
        fwd_errs.append(abs(got_val - expected_val))

        numeric = _numeric_grad(xvals)
        grad_errs.append(float(np.max(np.abs(numeric - analytic))))

    return {
        "max_abs_err": max(grad_errs),
        "forward_max_abs_err": max(fwd_errs),
    }
