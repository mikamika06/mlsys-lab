import numpy as np


def _true_mask(w: np.ndarray, amount: float) -> np.ndarray:
    n = w.size
    k = int(round(amount * n))  # PyTorch's _compute_nparams_toprune: python round()
    m = np.ones(n, dtype=bool)
    if k > 0:
        idx = np.argsort(np.abs(w))  # ascending -> smallest |w| first
        m[idx[:k]] = False
    return m


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0

    for _ in range(20):
        n = int(rng.integers(15, 40))
        w = rng.normal(size=n)
        amount = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        exp_mask = _true_mask(w, amount)
        expected_verdict = True

        cand = exp_mask.copy()
        if rng.uniform() < 0.5:
            # corrupt it: flip >=1 bits, so it provably differs from the true keep-set
            n_flip = int(rng.integers(1, max(2, n // 6)))
            flip_idx = rng.choice(n, size=n_flip, replace=False)
            cand[flip_idx] = ~cand[flip_idx]
            expected_verdict = False

        try:
            got_verdict = bool(sol.is_valid_l1_mask(w.copy(), cand.copy(), amount))
        except Exception:
            ok = 0.0
            continue

        if got_verdict != bool(expected_verdict):
            ok = 0.0

    return {"exact_match": ok}
