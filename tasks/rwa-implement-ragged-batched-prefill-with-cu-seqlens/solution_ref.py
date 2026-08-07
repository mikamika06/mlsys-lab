import math


def _causal_attention_segment(Qs, Ks, Vs):
    seg_len = len(Qs)
    n_heads = len(Qs[0])
    d = len(Qs[0][0])
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
                        dot_sum += Qs[i][h][c] * Ks[j][h][c]
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
                    acc += weights[i][j] * Vs[j][h][k]
                out_transposed[i][h][k] = acc

    return out_transposed


def ragged_batched_prefill_attention(
    Q: list[list[list[float]]],
    K: list[list[list[float]]],
    V: list[list[list[float]]],
    cu_seqlens: list[int],
) -> list[list[list[float]]]:
    n_tok = len(Q)
    n_heads = len(Q[0])
    d = len(Q[0][0])
    out = [[[0.0 for _ in range(d)] for _ in range(n_heads)] for _ in range(n_tok)]

    for s in range(len(cu_seqlens) - 1):
        lo, hi = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        if hi <= lo:
            continue
        seg_Q = Q[lo:hi]
        seg_K = K[lo:hi]
        seg_V = V[lo:hi]
        seg_out = _causal_attention_segment(seg_Q, seg_K, seg_V)
        for i in range(hi - lo):
            out[lo + i] = seg_out[i]

    return out
