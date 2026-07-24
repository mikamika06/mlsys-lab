import numpy as np


def _oracle_top_p_keep(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.argmax(cumulative >= p))
    return order[: cutoff + 1].tolist()


def grade(sol, fx) -> dict:
    cases = [
        ([0.40, 0.35, 0.25, 0.10], 0.75),
        ([0.5, 0.2, 0.2, 0.1], 0.9),
        ([0.6, 0.25, 0.15], 0.85),
        ([0.51, 0.30, 0.19], 0.81),
        ([0.45, 0.35, 0.10, 0.10], 0.80),
    ]
    ok = 1.0
    for probs, p in cases:
        try:
            got = list(sol.top_p_keep(list(probs), p))
        except Exception:
            ok = 0.0
            break
        expected = _oracle_top_p_keep(probs, p)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
