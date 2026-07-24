import numpy as np


def _oracle(router_probs):
    probs = np.asarray(router_probs, dtype=np.float64)
    tokens, experts = probs.shape
    chosen = np.argmax(probs, axis=1)
    counts = np.bincount(chosen, minlength=experts)
    f = counts.astype(np.float64) / tokens
    p = np.mean(probs, axis=0)
    return float(experts * np.sum(f * p))


def grade(sol, fx) -> dict:
    cases = [
        np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
        ], dtype=np.float64),
        np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ], dtype=np.float64),
        np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.1, 0.05],
            [0.8, 0.1, 0.1],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ], dtype=np.float64),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = float(sol.load_balancing_aux_loss(x))
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(x)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)
    return {"rel_err": worst}
