import numpy as np


def _oracle(Q, K, V, segment_ids):
    n, d = Q.shape
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)
    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    same_seg = segment_ids[:, None] == segment_ids[None, :]
    causal = col <= row
    allowed = same_seg & causal
    scores = np.where(allowed, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V.astype(np.float64)


def _make_case(rng):
    """Several segments (documents) packed into one training sequence, plus
    an ADVERSARIAL twist: the LAST token of every non-final segment carries
    a huge, distinctive value vector. A correct implementation must never
    let that poison leak into the next segment's outputs -- any leak makes
    the very first rows of the following segment explode in error."""
    num_segments = int(rng.integers(2, 5))
    lens = rng.integers(2, 7, size=num_segments)
    segment_ids = np.concatenate([np.full(int(l), s, dtype=np.int64) for s, l in enumerate(lens)])
    n = int(segment_ids.shape[0])
    d = int(rng.integers(3, 8))

    Q = rng.standard_normal((n, d))
    K = rng.standard_normal((n, d))
    V = rng.standard_normal((n, d))

    cum = np.cumsum(lens)
    for s in range(num_segments - 1):
        last_idx = int(cum[s]) - 1
        V[last_idx] = rng.choice([-1.0, 1.0], size=d) * 1e4

    return Q, K, V, segment_ids


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(41)
    worst = 0.0
    for _ in range(8):
        Q, K, V, segment_ids = _make_case(rng)
        ref = _oracle(Q, K, V, segment_ids)
        try:
            got = np.asarray(
                sol.packed_attention_with_reset_mask(Q.copy(), K.copy(), V.copy(), segment_ids.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
