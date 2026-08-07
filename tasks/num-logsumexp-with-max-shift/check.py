"""Grader for `num-logsumexp-with-max-shift`.

Oracle: `scipy.special.logsumexp` — the standard, independently-implemented
numerically stable reference — on fixtures spanning -1e4..1e4 (a range
where the naive `log(sum(exp(x)))` overflows to `inf`).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp

from mlsys import scorers


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    rel_errs = []

    for _ in range(8):
        shape = tuple(int(v) for v in rng.integers(2, 6, size=2))
        x = rng.uniform(-1e4, 1e4, size=shape)
        axis = int(rng.choice([0, 1, -1]))

        expected = _scipy_logsumexp(x, axis=axis)

        try:
            got = np.asarray(sol.logsumexp(x.tolist(), axis=axis), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != expected.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf")}

        rel_errs.append(scorers.rel_err(expected, got))

    return {"rel_err": float(np.mean(rel_errs))}
