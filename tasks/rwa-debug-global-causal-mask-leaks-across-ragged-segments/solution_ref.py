import math


def ragged_causal_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], cu_seqlens: list[int]) -> list[list[float]]:
    """Causal self-attention over a PACKED (ragged) batch.

    Q, K, V: (n, d) -- multiple variable-length sequences concatenated along
    the token axis. cu_seqlens: list of int of length (num_segments + 1)
    giving cumulative sequence boundaries, e.g. [0, 3, 7, 10] means segment 0
    is tokens[0:3], segment 1 is tokens[3:7], segment 2 is tokens[7:10].

    Row i may only attend to keys/values at position j such that:
      1. j <= i (causal), AND
      2. j is in the SAME segment as i (no cross-sequence leakage).

    Returns (n, d).
    """
    n = len(Q)
    if n == 0:
        return []
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    seg_id = [0] * n
    for s in range(len(cu_seqlens) - 1):
        for i in range(cu_seqlens[s], cu_seqlens[s + 1]):
            seg_id[i] = s

    scores = []
    for i in range(n):
        row_scores = []
        for j in range(n):
            if j <= i and seg_id[i] == seg_id[j]:
                dot = sum(Q[i][k] * K[j][k] for k in range(d))
                row_scores.append(dot * scale)
            else:
                row_scores.append(-float('inf'))
        scores.append(row_scores)

    probs = []
    for i in range(n):
        row = scores[i]
        max_val = -float('inf')
        for val in row:
            if val > max_val:
                max_val = val

        exps = []
        sum_exp = 0.0
        for val in row:
            if val == -float('inf'):
                exps.append(0.0)
            else:
                e = math.exp(val - max_val)
                exps.append(e)
                sum_exp += e

        if sum_exp > 0:
            probs.append([e / sum_exp for e in exps])
        else:
            probs.append([0.0] * n)

    out = []
    for i in range(n):
        out_row = []
        for k in range(d):
            val = sum(probs[i][j] * V[j][k] for j in range(n))
            out_row.append(val)
        out.append(out_row)

    return out
