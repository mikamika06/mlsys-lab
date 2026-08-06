import numpy as np
import math


def ragged_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, cu_seqlens: np.ndarray) -> np.ndarray:
    """Causal self-attention over a PACKED (ragged) batch.

    Q, K, V: (n, d) -- multiple variable-length sequences concatenated along
    the token axis. cu_seqlens: 1-D int array of length (num_segments + 1)
    giving cumulative sequence boundaries, e.g. [0, 3, 7, 10] means segment 0
    is tokens[0:3], segment 1 is tokens[3:7], segment 2 is tokens[7:10].

    Row i may only attend to keys/values at position j such that:
      1. j <= i (causal), AND
      2. j is in the SAME segment as i (no cross-sequence leakage).

    Returns (n, d).
    """
    n, d = Q.shape
    Q_f64 = Q.astype(np.float64)
    K_f64 = K.astype(np.float64)
    V_f64 = V.astype(np.float64)

    scores = [[0.0] * n for _ in range(n)]
    sqrt_d = math.sqrt(d)
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                dot += Q_f64[i, k] * K_f64[j, k]
            scores[i][j] = dot / sqrt_d

    seg_id = np.zeros(n, dtype=np.int64)
    for s in range(len(cu_seqlens) - 1):
        seg_id[cu_seqlens[s]:cu_seqlens[s + 1]] = s

    allowed = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if seg_id[i] == seg_id[j] and j <= i:
                allowed[i][j] = True

    max_scores = []
    for i in range(n):
        m = -float('inf')
        for j in range(n):
            if allowed[i][j]:
                if scores[i][j] > m:
                    m = scores[i][j]
        max_scores.append(m)

    probs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if allowed[i][j]:
                probs[i][j] = math.exp(scores[i][j] - max_scores[i])

    row_sums = []
    for i in range(n):
        s_val = 0.0
        for j in range(n):
            s_val += probs[i][j]
        row_sums.append(s_val)

    for i in range(n):
        s_val = row_sums[i]
        for j in range(n):
            probs[i][j] /= s_val

    out = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        for c in range(d):
            val = 0.0
            for j in range(n):
                val += probs[i][j] * V_f64[j, c]
            out[i, c] = val

    return out
