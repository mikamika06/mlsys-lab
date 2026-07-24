import numpy as np


def _oracle(accept_probs) -> float:
    """Independent computation via the exact stopping-position distribution
    Pr[L = l], rather than the telescoped closed form used by the reference
    solution — a genuinely different derivation path for cross-validation."""
    p = np.asarray(accept_probs, dtype=np.float64)
    K = p.shape[0]
    if K == 0:
        return 1.0
    survive = np.concatenate(([1.0], np.cumprod(p)))  # survive[l] = prod_{i=1}^{l} p_i
    total = 0.0
    for ell in range(K):
        prob_stop_here = survive[ell] * (1.0 - p[ell])
        total += (ell + 1) * prob_stop_here
    total += (K + 1) * survive[K]
    return float(total)


def _cases():
    rng = np.random.default_rng(0)
    cases = [
        np.array([]),
        np.array([0.5]),
        np.array([0.5, 0.5]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.9]),
        np.array([0.9, 0.0, 0.9]),
    ]
    for K in (1, 2, 3, 5, 8):
        cases.append(rng.uniform(0.0, 1.0, size=K))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for probs in _cases():
        ref = _oracle(probs)
        try:
            got = float(sol.expected_tokens_per_step(probs))
        except Exception:
            return {"rel_err": float("inf")}
        err = abs(got - ref) / max(abs(ref), 1e-15)
        worst = max(worst, err)
    return {"rel_err": worst}
