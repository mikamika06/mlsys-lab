import numpy as np

from mlsys import scorers


def _tokens_processed(request_tokens, budget) -> int:
    running = 0
    for t in request_tokens:
        if running + t <= budget:
            running += t
    return running


def _ref(request_tokens, budgets):
    curve = [_tokens_processed(request_tokens, b) for b in budgets]
    saturation = sum(request_tokens)
    return curve, saturation


def _scenarios():
    scenarios = []

    scenarios.append(([5, 8, 2, 10], [6, 9, 100]))
    scenarios.append(([1, 1, 1, 1, 1], [0, 1, 2, 3, 5, 5, 1000]))
    scenarios.append(([50, 1, 1, 1, 1, 1, 1], [5, 6, 49, 50, 55, 56]))
    scenarios.append(([10], [1, 9, 10, 11]))
    scenarios.append(([3, 3, 3, 3], [3, 6, 6, 9, 12, 11]))  # duplicate budgets

    rng = np.random.default_rng(0)
    for n in (10, 20, 15):
        request_tokens = rng.integers(1, 50, size=n).tolist()
        total = sum(request_tokens)
        budgets = rng.integers(0, total + 20, size=12).tolist()
        scenarios.append((request_tokens, budgets))

    return scenarios


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    total = 0
    correct = 0

    for request_tokens, budgets in _scenarios():
        total += 1
        curve_ref, sat_ref = _ref(request_tokens, budgets)

        try:
            curve_got, sat_got = sol.throughput_vs_budget(list(request_tokens), list(budgets))
        except Exception:
            return {"rel_err": float("inf"), "exact_match": 0.0}

        try:
            curve_got = [int(x) for x in curve_got]
            sat_got = int(sat_got)
        except Exception:
            return {"rel_err": float("inf"), "exact_match": 0.0}

        if len(curve_got) != len(curve_ref):
            return {"rel_err": float("inf"), "exact_match": 0.0}

        err = scorers.rel_err(np.array(curve_ref, dtype=np.float64), np.array(curve_got, dtype=np.float64))
        if not np.isfinite(err):
            return {"rel_err": float("inf"), "exact_match": 0.0}
        worst_rel = max(worst_rel, err)

        if sat_got == sat_ref:
            correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"rel_err": worst_rel, "exact_match": exact_match}
