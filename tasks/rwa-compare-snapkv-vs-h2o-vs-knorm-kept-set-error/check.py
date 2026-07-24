import numpy as np


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attend(q, K, V, d) -> np.ndarray:
    scores = (q @ K.T) / np.sqrt(d)
    weights = _softmax_rows(scores[None, :])[0]
    return weights @ V


def _knorm_idx(K, budget):
    norms = np.linalg.norm(K, axis=1)
    idx = np.argsort(norms, kind="stable")[:budget]
    return np.sort(idx)


def _h2o_idx(K, Q_hist, budget, recent_window, d):
    n = K.shape[0]
    attn = _softmax_rows((Q_hist @ K.T) / np.sqrt(d))
    score = attn.sum(axis=0)
    recent = np.arange(n - recent_window, n)
    k_extra = budget - recent_window
    if k_extra <= 0:
        return np.sort(recent[-budget:])
    mask = np.ones(n, dtype=bool)
    mask[recent] = False
    cand = np.nonzero(mask)[0]
    top_extra = cand[np.argsort(-score[cand], kind="stable")[:k_extra]]
    return np.sort(np.concatenate([recent, top_extra]))


def _snapkv_idx(K, Q_hist, budget, snap_window, pool_size, d):
    n = K.shape[0]
    Qw = Q_hist[-snap_window:]
    attn = _softmax_rows((Qw @ K.T) / np.sqrt(d))
    raw_score = attn.sum(axis=0)
    pad = pool_size // 2
    padded = np.pad(raw_score, (pad, pad), mode="edge")
    kernel = np.ones(pool_size) / pool_size
    pooled = np.convolve(padded, kernel, mode="valid")
    win = np.arange(n - snap_window, n)
    k_extra = budget - snap_window
    if k_extra <= 0:
        return np.sort(win[-budget:])
    mask = np.ones(n, dtype=bool)
    mask[win] = False
    cand = np.nonzero(mask)[0]
    top_extra = cand[np.argsort(-pooled[cand], kind="stable")[:k_extra]]
    return np.sort(np.concatenate([win, top_extra]))


def _oracle(K, V, Q_hist, q_new, budget, recent_window, snap_window, pool_size) -> dict:
    d = K.shape[1]
    full_out = _attend(q_new, K, V, d)

    knorm_idx = _knorm_idx(K, budget)
    h2o_idx = _h2o_idx(K, Q_hist, budget, recent_window, d)
    snap_idx = _snapkv_idx(K, Q_hist, budget, snap_window, pool_size, d)

    def err(idx):
        out = _attend(q_new, K[idx], V[idx], d)
        return float(np.max(np.abs(out - full_out)))

    ks, hs, ss = set(knorm_idx.tolist()), set(h2o_idx.tolist()), set(snap_idx.tolist())
    return {
        "knorm_error": err(knorm_idx),
        "h2o_error": err(h2o_idx),
        "snapkv_error": err(snap_idx),
        "overlap_knorm_h2o": len(ks & hs),
        "overlap_knorm_snapkv": len(ks & ss),
        "overlap_h2o_snapkv": len(hs & ss),
    }


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    overlap_correct = 0
    overlap_total = 0

    for _ in range(6):
        n = int(rng.integers(24, 60))
        d = int(rng.integers(4, 12))
        T = int(rng.integers(8, 16))
        budget = int(rng.integers(8, n // 2))
        recent_window = int(rng.integers(2, min(budget, 6) + 1))
        snap_window = int(rng.integers(2, min(budget, T, 6) + 1))
        pool_size = 3

        K = rng.standard_normal((n, d))
        V = rng.standard_normal((n, d))
        Q_hist = rng.standard_normal((T, d))
        q_new = rng.standard_normal(d)

        ref = _oracle(K, V, Q_hist, q_new, budget, recent_window, snap_window, pool_size)

        try:
            got = sol.compare_eviction_methods(
                K.copy(), V.copy(), Q_hist.copy(), q_new.copy(),
                budget, recent_window, snap_window, pool_size,
            )
            for key in ("knorm_error", "h2o_error", "snapkv_error"):
                v = float(got[key])
                max_err = max(max_err, abs(v - ref[key]))
            for key in ("overlap_knorm_h2o", "overlap_knorm_snapkv", "overlap_h2o_snapkv"):
                overlap_total += 1
                if int(got[key]) == ref[key]:
                    overlap_correct += 1
        except Exception:
            max_err = float("inf")
            overlap_total += 3

    overlap_match = (overlap_correct / overlap_total) if overlap_total else 0.0
    return {"max_abs_err": max_err, "overlap_exact_match": overlap_match}
