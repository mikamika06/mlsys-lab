import math
import numpy as np


def gathered_attention(k_phys, v_phys, block_table, seq_len, q):
    """Gather logical KV from a block table, truncate to the true seq_len,
    and compute single-query scaled dot-product attention."""
    H = k_phys.shape[-1]
    k_logical = k_phys[block_table].reshape(-1, H).astype(np.float64)[:seq_len]
    v_logical = v_phys[block_table].reshape(-1, H).astype(np.float64)[:seq_len]
    qf = np.asarray(q, dtype=np.float64)

    scores = np.empty(seq_len, dtype=np.float64)
    inv_sqrt_H = 1.0 / math.sqrt(H)
    for i in range(seq_len):
        dot = 0.0
        for j in range(H):
            dot += k_logical[i, j] * qf[j]
        scores[i] = dot * inv_sqrt_H

    max_score = scores[0]
    for i in range(1, seq_len):
        if scores[i] > max_score:
            max_score = scores[i]

    weights = np.empty(seq_len, dtype=np.float64)
    sum_weights = 0.0
    for i in range(seq_len):
        w = math.exp(scores[i] - max_score)
        weights[i] = w
        sum_weights += w

    for i in range(seq_len):
        weights[i] /= sum_weights

    result = np.zeros(H, dtype=np.float64)
    for j in range(H):
        acc = 0.0
        for i in range(seq_len):
            acc += weights[i] * v_logical[i, j]
        result[j] = acc

    return result
