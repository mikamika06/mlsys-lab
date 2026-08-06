import math
import numpy as np


def _causal_attention_segment(Qs, Ks, Vs):
    seg_len, n_heads, d = Qs.shape
    out_transposed = [[[0.0 for _ in range(d)] for _ in range(n_heads)] for _ in range(seg_len)]
    sqrt_d = math.sqrt(d)

    for h in range(n_heads):
        scores = [[0.0 for _ in range(seg_len)] for _ in range(seg_len)]
        for i in range(seg_len):
            for j in range(seg_len):
                if j > i:
                    scores[i][j] = float("-inf")
                else:
                    dot_sum = 0.0
                    for c in range(d):
                        dot_sum += Qs[i, h, c] * Ks[j, h, c]
                    scores[i][j] = dot_sum / sqrt_d

        for i in range(seg_len):
            max_val = scores[i][0]
            for j in range(1, seg_len):
                if scores[i][j] > max_val:
                    max_val = scores[i][j]
            for j in range(seg_len):
                scores[i][j] -= max_val

        weights = [[0.0 for _ in range(seg_len)] for _ in range(seg_len)]
        for i in range(seg_len):
            row_sum = 0.0
            for j in range(seg_len):
                val = math.exp(scores[i][j])
                weights[i][j] = val
                row_sum += val
            for j in range(seg_len):
                weights[i][j] /= row_sum

        for i in range(seg_len):
            for k in range(d):
                acc = 0.0
                for j in range(seg_len):
                    acc += weights[i][j] * Vs[j, h, k]
                out_transposed[i][h][k] = acc

    return np.array(out_transposed, dtype=np.float64)


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
