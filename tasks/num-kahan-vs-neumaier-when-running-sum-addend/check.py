import math

import numpy as np

from mlsys.scorers import max_abs_err, rel_err


def _cases(rng):
    cases = []
    # worked example from task.md
    cases.append(np.array([1.0, 1e16, 1.0, 1.0, -1e16, 1.0, 1.0, 1.0], dtype=np.float64))
    # scaled variants where |running sum| << |next addend|
    for scale in (1e16, 1e18, 1e20, 1e24):
        n_small = int(rng.integers(3, 7))
        small = rng.uniform(0.5, 2.0, size=n_small)
        arr = np.concatenate([small[:1], [scale], small[1:], [-scale], small[:2]])
        cases.append(arr.astype(np.float64))
    return cases


def _oracle_kahan(arr) -> float:
    s = 0.0
    c = 0.0
    for x in arr:
        x = float(x)
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def grade(sol, fx) -> dict:
    cases = _cases(np.random.default_rng(0))

    neumaier_err = 0.0
    kahan_match = 0.0
    worst_advantage = float("inf")

    for arr in cases:
        truth = math.fsum(float(v) for v in arr)
        ref_kahan = _oracle_kahan(arr)

        try:
            sol_kahan = float(sol.kahan_sum(arr.copy()))
            sol_neumaier = float(sol.neumaier_sum(arr.copy()))
        except Exception:
            return {
                "neumaier_rel_err": float("inf"),
                "kahan_match_err": float("inf"),
                "neumaier_advantage": float("-inf"),
            }

        if not (math.isfinite(sol_kahan) and math.isfinite(sol_neumaier)):
            return {
                "neumaier_rel_err": float("inf"),
                "kahan_match_err": float("inf"),
                "neumaier_advantage": float("-inf"),
            }

        neumaier_err = max(neumaier_err, rel_err(np.array([truth]), np.array([sol_neumaier])))
        kahan_match = max(kahan_match, max_abs_err(np.array([ref_kahan]), np.array([sol_kahan])))

        advantage = abs(sol_kahan - truth) - abs(sol_neumaier - truth)
        worst_advantage = min(worst_advantage, advantage)

    return {
        "neumaier_rel_err": float(neumaier_err),
        "kahan_match_err": float(kahan_match),
        "neumaier_advantage": float(worst_advantage),
    }
