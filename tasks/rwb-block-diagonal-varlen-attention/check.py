import numpy as np


def _lengths_from_cu(cu_seqlens):
    cu = np.asarray(cu_seqlens)
    return (cu[1:] - cu[:-1]).tolist()


def _oracle(q, k, v, cu_seqlens):
    """Per-sequence dense attention, computed by unpacking each segment
    (via cu_seqlens) and concatenating -- never references a
    block-diagonal mask, independent of the packed implementation."""
    d = q.shape[1]
    lengths = _lengths_from_cu(cu_seqlens)
    outs = []
    pos = 0
    for L in lengths:
        qs = q[pos:pos + L]
        ks = k[pos:pos + L]
        vs = v[pos:pos + L]
        scores = (qs @ ks.T) / np.sqrt(d)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        outs.append(probs @ vs)
        pos += L
    return np.concatenate(outs, axis=0)


def _synthetic_cases():
    rng = np.random.default_rng(37)
    cases = []
    for _ in range(4):
        n_seqs = int(rng.integers(1, 6))
        lengths = [int(rng.integers(1, 12)) for _ in range(n_seqs)]
        cu_seqlens = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        N = int(cu_seqlens[-1])
        d = int(rng.integers(2, 8))
        q = rng.standard_normal((N, d))
        k = rng.standard_normal((N, d))
        v = rng.standard_normal((N, d))
        cases.append((q, k, v, cu_seqlens))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["q"], fx["k"], fx["v"], fx["cu_seqlens"])] + _synthetic_cases()

    worst = 0.0
    for q, k, v, cu_seqlens in cases:
        ref = _oracle(q, k, v, cu_seqlens)
        try:
            got = np.asarray(
                sol.varlen_block_diagonal_attention(q.copy(), k.copy(), v.copy(), cu_seqlens.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
