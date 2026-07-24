import numpy as np


def _accepted_tokens(alpha, k):
    # sum_{i=0}^{k} alpha^i, elementwise, per-row k limit, avoids the
    # 1/(1-alpha) singularity at alpha == 1 entirely.
    max_k = int(np.max(k))
    total = np.zeros_like(alpha)
    term = np.ones_like(alpha)
    for i in range(max_k + 1):
        active = i <= k
        total = total + np.where(active, term, 0.0)
        term = term * alpha
    return total


def _oracle(configs, iters=200):
    configs = np.asarray(configs, dtype=np.float64)
    c = configs[:, 0]
    k = configs[:, 1].astype(np.int64)

    target = 1.0 + k * c
    lo = np.zeros_like(c)
    hi = np.ones_like(c)

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        val = _accepted_tokens(mid, k)
        go_up = val < target
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)

    return (lo + hi) / 2.0


def grade(sol, fx) -> dict:
    cases = np.array(
        [
            [0.0, 3],
            [0.2, 2],
            [0.5, 4],
            [0.1, 8],
            [0.9, 1],
            [0.5, 1],
            [0.05, 16],
            [0.75, 6],
        ],
        dtype=np.float64,
    )

    try:
        got = np.asarray(sol.break_even_alpha(cases), dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    ref = _oracle(cases)

    if got.shape != ref.shape:
        return {"rel_err": float("inf")}

    err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
    return {"rel_err": err}
