"""Oracle: independent re-implementation of SnapKV's observation-window
pooled-vote KV-cache eviction. Aggregates attention over heads and the
window, average-pools the vote with a fixed kernel, and takes the
`capacity` prefill positions with the highest pooled score.
"""
import numpy as np


def _pool1d(scores, kernel_size):
    pad = kernel_size // 2
    padded = np.zeros(len(scores) + 2 * pad)
    padded[pad:pad + len(scores)] = scores
    out = np.zeros(len(scores))
    for i in range(len(scores)):
        out[i] = np.mean(padded[i:i + kernel_size])
    return out


def _oracle(attn, window_size, kernel_size, capacity):
    attn = np.asarray(attn, dtype=np.float64)
    H, W, L_prefix = attn.shape
    scores = np.zeros(L_prefix)
    for h in range(H):
        for w in range(W):
            scores += attn[h, w]
    pooled = _pool1d(scores, kernel_size)

    k = min(int(capacity), L_prefix)
    ranked = sorted(range(L_prefix), key=lambda i: (-pooled[i], i))
    selected = np.array(sorted(ranked[:k]), dtype=np.int64)
    kept_total = k + window_size
    ratio = kept_total / (L_prefix + window_size)
    return selected, kept_total, ratio


def _cases():
    return [
        dict(H=4, W=8, L=64, kernel=5, capacity=20, seed=0),
        dict(H=2, W=4, L=32, kernel=3, capacity=10, seed=1),
        dict(H=1, W=1, L=16, kernel=1, capacity=6, seed=2),   # no smoothing
        dict(H=3, W=6, L=20, kernel=7, capacity=100, seed=3),  # capacity clips to L (full retention)
        dict(H=6, W=10, L=100, kernel=9, capacity=1, seed=4),  # keep almost nothing
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for c in _cases():
        rng = np.random.default_rng(c["seed"])
        H, W, L = c["H"], c["W"], c["L"]
        raw = rng.exponential(scale=1.0, size=(H, W, L))
        attn = raw / raw.sum(axis=-1, keepdims=True)  # rows sum to 1, like real attention

        sel_ref, kept_ref, ratio_ref = _oracle(attn, W, c["kernel"], c["capacity"])

        try:
            out = sol.snapkv_select(attn.copy(), W, c["kernel"], c["capacity"])
            sel_got, kept_got, ratio_got = out
            sel_got = np.asarray(sel_got, dtype=np.int64)
        except Exception:
            return {"exact_match": 0.0}

        if sel_got.shape != sel_ref.shape:
            ok = 0.0
            break
        if not np.array_equal(sel_got, sel_ref):
            ok = 0.0
            break
        if int(kept_got) != int(kept_ref):
            ok = 0.0
            break
        if abs(float(ratio_got) - ratio_ref) > 1e-9:
            ok = 0.0
            break

    return {"exact_match": ok}
