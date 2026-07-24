"""Grader for `num-prove-kahan-sum-is-a-real-per-element-loop`.

Two independent gates:
  * line_count — `sys.settrace`-based line-event count during the call,
    proving a real per-element Python loop ran (a vectorised `np.sum`
    call executes almost no Python-level lines).
  * rel_err — accuracy against `math.fsum` (Python's real, correctly-
    rounded summation oracle) on adversarial catastrophic-cancellation
    fixtures: a huge value, many `+1.0`s, then the huge value negated.
    Plain sequential summation loses every `+1.0` (they're below the
    huge value's ULP) and returns ~0; Kahan summation recovers the
    exact answer.
"""
from __future__ import annotations

import math

import numpy as np

from mlsys import probe


def _cases():
    # (magnitude of the cancelling pair, count of +1.0 terms in between)
    return [(1e16, 2000), (5e15, 1500), (2e16, 3000)]


def grade(sol, fx) -> dict:
    max_rel_err = 0.0
    min_line_count = None

    for big, n in _cases():
        x = np.concatenate([[big], np.ones(n), [-big]]).astype(np.float64)
        expected = math.fsum(x.tolist())   # real, correctly-rounded oracle

        got_holder: list = []

        def _call(arr=x, holder=got_holder):
            holder.append(sol.kahan_sum(arr))

        try:
            lc = probe.count_line_events(_call)
            got = float(got_holder[0])
        except Exception:
            return {"line_count": 0.0, "rel_err": float("inf")}

        if not np.isfinite(got):
            return {"line_count": 0.0, "rel_err": float("inf")}

        rel = abs(got - expected) / (abs(expected) + 1e-12)
        max_rel_err = max(max_rel_err, rel)
        min_line_count = lc if min_line_count is None else min(min_line_count, lc)

    return {
        "line_count": float(min_line_count),
        "rel_err": float(max_rel_err),
    }
