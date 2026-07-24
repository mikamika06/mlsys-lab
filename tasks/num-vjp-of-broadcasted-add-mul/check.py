"""Grader for `num-vjp-of-broadcasted-add-mul`.

Oracle: central finite differences of L(a, b) = sum(grad_out * (a OP b))
w.r.t. every element of `a` and every element of `b`. By the chain rule
this is exactly the VJP: dL/da = grad_a, dL/db = grad_b for the given
`grad_out` -- so the finite-difference gradient of this scalar `L` is a
completely independent oracle for `add_vjp` / `mul_vjp`, never calling
into `sol` and never hardcoding an expected array.
"""
from __future__ import annotations

import numpy as np


def _numeric_grad(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x.copy(); xp[idx] += h
        xm = x.copy(); xm[idx] -= h
        g[idx] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def _shape_pairs():
    return [
        ((3, 1, 4), (5, 4)),
        ((1,), (3, 4)),
        ((2, 3), (3,)),
        ((4, 1), (1, 5)),
        ((2, 1, 3), (4, 3)),
    ]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0

    for a_shape, b_shape in _shape_pairs():
        out_shape = np.broadcast_shapes(a_shape, b_shape)
        a = rng.uniform(0.5, 2.0, size=a_shape)
        b = rng.uniform(0.5, 2.0, size=b_shape)
        grad_out = rng.uniform(-1.0, 1.0, size=out_shape)

        for op, vjp_name in (("add", "add_vjp"), ("mul", "mul_vjp")):
            f = (lambda x, y: x + y) if op == "add" else (lambda x, y: x * y)

            ga_num = _numeric_grad(lambda av: float(np.sum(grad_out * f(av, b))), a)
            gb_num = _numeric_grad(lambda bv: float(np.sum(grad_out * f(a, bv))), b)

            try:
                vjp = getattr(sol, vjp_name)
                ga, gb = vjp(a.copy(), b.copy(), grad_out.copy())
                ga = np.asarray(ga, dtype=np.float64)
                gb = np.asarray(gb, dtype=np.float64)
            except Exception:
                return {"max_abs_err": float("inf")}

            if ga.shape != a.shape or gb.shape != b.shape:
                return {"max_abs_err": float("inf")}
            if not (np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))):
                return {"max_abs_err": float("inf")}

            max_err = max(max_err, float(np.max(np.abs(ga - ga_num))))
            max_err = max(max_err, float(np.max(np.abs(gb - gb_num))))

    return {"max_abs_err": float(max_err)}
