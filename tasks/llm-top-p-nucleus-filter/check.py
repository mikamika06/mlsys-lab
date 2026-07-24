import numpy as np


def _ref_top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    boundary = int(np.searchsorted(cumulative, p, side="left"))
    return np.sort(order[: boundary + 1])


def grade(sol, fx) -> dict:
    cases = [
        (np.array([0.05, 0.60, 0.10, 0.25]), 0.70),
        (np.array([0.40, 0.30, 0.20, 0.10]), 0.50),
        (np.array([0.01, 0.01, 0.49, 0.49]), 0.51),
        (np.array([0.8, 0.05, 0.05, 0.05, 0.05]), 0.90),
        (np.array([0.25, 0.25, 0.25, 0.25]), 0.76),
    ]
    ok = 1.0
    for probs, p in cases:
        try:
            got = np.sort(np.asarray(sol.top_p_filter(probs.copy(), p), dtype=np.int64))
        except Exception:
            ok = 0.0
            break
        expected = _ref_top_p_filter(probs, p)
        if not np.array_equal(got, expected):
            ok = 0.0
            break
    return {"exact_match": ok}
