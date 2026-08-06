import math
import numpy as np


def packed_attention_with_reset_mask(Q: np.ndarray, K: np.ndarray, V: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Causal self-attention over multiple documents PACKED into one
    training sequence, with the mask RESET at every segment boundary.

    Q, K, V: (n, d). segment_ids: (n,) int array; segment_ids[i] is the
    segment/document index token i belongs to (e.g. [0,0,0,1,1,2,2,2,2] for
    three packed documents of length 3, 2, 4).

    Row i may attend to column j iff j <= i (causal) AND segment_ids[j] ==
    segment_ids[i] (same document -- the mask resets, exactly like resetting
    position ids at each packed-document boundary). Returns (n, d).
    """
    n, d = Q.shape
    out = np.zeros((n, d), dtype=np.float64)
    sqrt_d = math.sqrt(d)

    for i in range(n):
        scores_row = []
        seg_i = segment_ids[i]
        for j in range(n):
            if j <= i and segment_ids[j] == seg_i:
                dot = 0.0
                for k in range(d):
                    dot += float(Q[i, k]) * float(K[j, k])
                scores_row.append(dot / sqrt_d)
            else:
                scores_row.append(-float("inf"))

        max_score = -float("inf")
        for j in range(n):
            if scores_row[j] > max_score:
                max_score = scores_row[j]

        probs_row = []
        sum_exp = 0.0
        for j in range(n):
            if scores_row[j] == -float("inf"):
                val = 0.0
            else:
                val = math.exp(scores_row[j] - max_score)
            probs_row.append(val)
            sum_exp += val

        for j in range(n):
            probs_row[j] /= sum_exp

        for k in range(d):
            val = 0.0
            for j in range(n):
                val += probs_row[j] * float(V[j, k])
            out[i, k] = val

    return out
