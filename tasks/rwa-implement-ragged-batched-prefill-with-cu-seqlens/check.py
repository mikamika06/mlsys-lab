import numpy as np


def _causal_attention_segment(Qs, Ks, Vs):
    # Qs, Ks, Vs: (seg_len, n_heads, d) -> (seg_len, n_heads, d)
    seg_len, n_heads, d = Qs.shape
    Qh = Qs.transpose(1, 0, 2)  # (n_heads, seg_len, d)
    Kh = Ks.transpose(1, 0, 2)
    Vh = Vs.transpose(1, 0, 2)

    scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)  # (n_heads, seg_len, seg_len)
    disallowed = np.triu(np.ones((seg_len, seg_len), dtype=bool), k=1)
    scores = np.where(disallowed[None, :, :], -np.inf, scores)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    out = weights @ Vh  # (n_heads, seg_len, d)
    return out.transpose(1, 0, 2)


def _oracle(Q, K, V, cu_seqlens):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    cu_seqlens = np.asarray(cu_seqlens, dtype=np.int64)

    n_tok, n_heads, d = Q.shape
    out = np.zeros((n_tok, n_heads, d), dtype=np.float64)

    for s in range(len(cu_seqlens) - 1):
        lo, hi = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        if hi <= lo:
            continue
        out[lo:hi] = _causal_attention_segment(Q[lo:hi], K[lo:hi], V[lo:hi])

    return out


def _make_case(rng, seg_lens, n_heads, d):
    cu_seqlens = np.concatenate([[0], np.cumsum(seg_lens)]).astype(np.int64)
    n_tok = int(cu_seqlens[-1])
    Q = rng.standard_normal((n_tok, n_heads, d))
    K = rng.standard_normal((n_tok, n_heads, d))
    V = rng.standard_normal((n_tok, n_heads, d))
    return Q, K, V, cu_seqlens


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    case_specs = [
        ([3, 4], 2, 4),
        ([1, 1, 1], 1, 3),
        ([5, 2, 3, 4], 3, 2),
        ([2, 6], 4, 8),
        ([1, 5, 1], 2, 5),
        ([4, 4, 4, 4, 4], 1, 4),
    ]

    max_err = 0.0
    for seg_lens, n_heads, d in case_specs:
        Q, K, V, cu_seqlens = _make_case(rng, seg_lens, n_heads, d)
        ref = _oracle(Q, K, V, cu_seqlens)

        try:
            got = sol.ragged_batched_prefill_attention(
                Q.tolist(), K.tolist(), V.tolist(), cu_seqlens.tolist()
            )
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
