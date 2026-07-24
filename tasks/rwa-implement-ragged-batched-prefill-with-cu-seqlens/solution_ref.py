import numpy as np


def _causal_attention_segment(Qs, Ks, Vs):
    seg_len, n_heads, d = Qs.shape
    Qh = Qs.transpose(1, 0, 2)
    Kh = Ks.transpose(1, 0, 2)
    Vh = Vs.transpose(1, 0, 2)

    scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)
    disallowed = np.triu(np.ones((seg_len, seg_len), dtype=bool), k=1)
    scores = np.where(disallowed[None, :, :], -np.inf, scores)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    out = weights @ Vh
    return out.transpose(1, 0, 2)


def ragged_batched_prefill_attention(Q, K, V, cu_seqlens):
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
