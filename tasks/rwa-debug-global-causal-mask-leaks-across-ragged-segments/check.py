import numpy as np


def _segment_ids(n, cu_seqlens):
    seg_id = np.zeros(n, dtype=np.int64)
    for s in range(len(cu_seqlens) - 1):
        seg_id[cu_seqlens[s]:cu_seqlens[s + 1]] = s
    return seg_id


def _oracle(Q, K, V, cu_seqlens):
    n, d = Q.shape
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    seg_id = _segment_ids(n, cu_seqlens)
    same_segment = seg_id[:, None] == seg_id[None, :]
    causal = col <= row
    allowed = same_segment & causal

    scores = np.where(allowed, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V.astype(np.float64)


def _make_case(rng):
    """A packed batch: several variable-length sequences concatenated along
    the token axis, with cu_seqlens marking their boundaries."""
    num_segments = int(rng.integers(2, 5))
    lens = rng.integers(1, 6, size=num_segments)
    cu_seqlens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    n = int(cu_seqlens[-1])
    d = int(rng.integers(2, 8))
    Q = rng.standard_normal((n, d))
    K = rng.standard_normal((n, d))
    V = rng.standard_normal((n, d))
    return Q, K, V, cu_seqlens


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(8):
        Q, K, V, cu_seqlens = _make_case(rng)
        ref = _oracle(Q, K, V, cu_seqlens)
        try:
            got = np.asarray(
                sol.ragged_causal_attention(Q.copy(), K.copy(), V.copy(), cu_seqlens.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
