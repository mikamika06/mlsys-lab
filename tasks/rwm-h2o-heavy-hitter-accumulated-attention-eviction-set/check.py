import numpy as np


def _oracle(S, budget, recent_window):
    n = S.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    S_masked = np.where(mask, -np.inf, S)
    S_masked = S_masked - np.max(S_masked, axis=1, keepdims=True)
    P = np.exp(S_masked)
    P = P / np.sum(P, axis=1, keepdims=True)

    h = P.sum(axis=0)

    recent = set(range(max(0, n - recent_window), n))
    n_heavy = budget - len(recent)

    candidates = [j for j in range(n) if j not in recent]
    candidates.sort(key=lambda j: (-h[j], j))
    heavy = candidates[:n_heavy]

    retained = sorted(set(heavy) | recent)
    retained_idx = np.array(retained, dtype=np.int64)
    preserved_mass = float(h[retained_idx].sum() / h.sum())
    return retained_idx, preserved_mass


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    for n, budget, recent_window in [(10, 5, 2), (16, 8, 3), (6, 6, 1), (20, 7, 4)]:
        S = rng.standard_normal((n, n)) * rng.uniform(0.5, 3.0)
        cases.append((S, budget, recent_window))

    # A case with a clear dominant "heavy hitter" column to sanity check
    # the ranking direction.
    n, budget, recent_window = 12, 5, 2
    S = rng.standard_normal((n, n)) * 0.3
    S[:, 2] += 20.0  # token 2 should dominate accumulated attention
    cases.append((S, budget, recent_window))

    return cases


def grade(sol, fx) -> dict:
    ok = 1.0
    for S, budget, recent_window in _cases():
        ref_idx, ref_mass = _oracle(S.astype(np.float64), budget, recent_window)
        try:
            got = sol.h2o_eviction_set(S.copy(), budget, recent_window)
            got_idx, got_mass = got
            got_idx = np.asarray(got_idx, dtype=np.int64)
        except Exception:
            ok = 0.0
            break

        if not np.array_equal(got_idx, ref_idx):
            ok = 0.0
            break
        if abs(float(got_mass) - ref_mass) > 1e-9:
            ok = 0.0
            break

    return {"exact_match": ok}
