import numpy as np


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attend(q: np.ndarray, K: np.ndarray, V: np.ndarray, d: int) -> np.ndarray:
    """Single-query scaled dot-product attention output, shape (d,)."""
    scores = (q @ K.T) / np.sqrt(d)
    weights = _softmax_rows(scores[None, :])[0]
    return weights @ V


def _knorm_idx(K: np.ndarray, budget: int) -> np.ndarray:
    """Knorm: keep the `budget` tokens with the SMALLEST key L2 norm
    (low-norm keys receive disproportionately high attention weight)."""
    norms = np.linalg.norm(K, axis=1)
    idx = np.argsort(norms, kind="stable")[:budget]
    return np.sort(idx)


def _h2o_idx(K: np.ndarray, Q_hist: np.ndarray, budget: int, recent_window: int, d: int) -> np.ndarray:
    """H2O: always keep the most recent `recent_window` tokens, plus the
    top remaining tokens by cumulative attention mass received across every
    row of Q_hist (the 'heavy hitters')."""
    n = K.shape[0]
    attn = _softmax_rows((Q_hist @ K.T) / np.sqrt(d))  # (T, n)
    score = attn.sum(axis=0)  # (n,) cumulative mass per token

    recent = np.arange(n - recent_window, n)
    k_extra = budget - recent_window
    if k_extra <= 0:
        return np.sort(recent[-budget:])

    mask = np.ones(n, dtype=bool)
    mask[recent] = False
    cand = np.nonzero(mask)[0]
    top_extra = cand[np.argsort(-score[cand], kind="stable")[:k_extra]]
    return np.sort(np.concatenate([recent, top_extra]))


def _snapkv_idx(K: np.ndarray, Q_hist: np.ndarray, budget: int, snap_window: int,
                 pool_size: int, d: int) -> np.ndarray:
    """SnapKV: always keep the last `snap_window` tokens (the observation
    window), plus the top remaining tokens by attention mass received from
    ONLY the observation-window queries, average-pooled with `pool_size`
    (odd) to favour contiguous clusters of important tokens."""
    n = K.shape[0]
    Qw = Q_hist[-snap_window:]
    attn = _softmax_rows((Qw @ K.T) / np.sqrt(d))  # (snap_window, n)
    raw_score = attn.sum(axis=0)  # (n,)

    pad = pool_size // 2
    padded = np.pad(raw_score, (pad, pad), mode="edge")
    kernel = np.ones(pool_size) / pool_size
    pooled = np.convolve(padded, kernel, mode="valid")  # (n,)

    win = np.arange(n - snap_window, n)
    k_extra = budget - snap_window
    if k_extra <= 0:
        return np.sort(win[-budget:])

    mask = np.ones(n, dtype=bool)
    mask[win] = False
    cand = np.nonzero(mask)[0]
    top_extra = cand[np.argsort(-pooled[cand], kind="stable")[:k_extra]]
    return np.sort(np.concatenate([win, top_extra]))


def compare_eviction_methods(K: np.ndarray, V: np.ndarray, Q_hist: np.ndarray, q_new: np.ndarray,
                              budget: int, recent_window: int, snap_window: int,
                              pool_size: int) -> dict:
    """Compare three KV-cache eviction policies (Knorm, H2O, SnapKV) on the
    same context by selecting each policy's kept-token set and measuring
    the resulting single-query attention output error against full (no
    eviction) attention, plus the pairwise overlap of the kept sets.

    K, V        : (n, d) cached keys/values.
    Q_hist      : (T, d) queries already issued while this context was in
                  the KV cache (used by H2O/SnapKV to score tokens).
    q_new       : (d,) a new query to attend with, after eviction.
    budget      : number of tokens each policy is allowed to keep.
    recent_window : H2O's always-kept trailing window size (<= budget).
    snap_window : SnapKV's observation window size (<= budget, <= len(Q_hist)).
    pool_size   : odd kernel size for SnapKV's average pooling.

    Returns a dict with keys:
      "knorm_error", "h2o_error", "snapkv_error"       -- float, max abs
          error of that policy's compressed-KV attention output vs the
          full-context attention output for q_new.
      "overlap_knorm_h2o", "overlap_knorm_snapkv",
      "overlap_h2o_snapkv"                              -- int, size of the
          intersection of the two policies' kept index sets.
    """
    d = K.shape[1]
    full_out = _attend(q_new, K, V, d)

    knorm_idx = _knorm_idx(K, budget)
    h2o_idx = _h2o_idx(K, Q_hist, budget, recent_window, d)
    snap_idx = _snapkv_idx(K, Q_hist, budget, snap_window, pool_size, d)

    def err(idx: np.ndarray) -> float:
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
