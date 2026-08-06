import math
import numpy as np


def mha_forward(
    X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray
) -> np.ndarray:
    """Full multi-head attention forward pass."""
    batch, seq_len, d_model = X.shape
    H = 4
    head_dim = d_model // H
    assert d_model % H == 0

    Q = np.zeros((batch, seq_len, d_model), dtype=X.dtype)
    K = np.zeros((batch, seq_len, d_model), dtype=X.dtype)
    V = np.zeros((batch, seq_len, d_model), dtype=X.dtype)

    for b in range(batch):
        for i in range(seq_len):
            for j in range(d_model):
                q_val = 0.0
                k_val = 0.0
                v_val = 0.0
                for k in range(d_model):
                    x = X[b, i, k]
                    q_val += x * Wq[k, j]
                    k_val += x * Wk[k, j]
                    v_val += x * Wv[k, j]
                Q[b, i, j] = q_val
                K[b, i, j] = k_val
                V[b, i, j] = v_val

    out = np.zeros((batch, seq_len, d_model), dtype=X.dtype)
    scale = math.sqrt(head_dim)

    for b in range(batch):
        for h in range(H):
            head_offset = h * head_dim
            for i in range(seq_len):
                scores = []
                max_score = None
                for j in range(seq_len):
                    score = 0.0
                    for d in range(head_dim):
                        q_elem = Q[b, i, head_offset + d]
                        k_elem = K[b, j, head_offset + d]
                        score += q_elem * k_elem
                    score /= scale
                    scores.append(score)
                    if max_score is None or score > max_score:
                        max_score = score

                exp_scores = []
                sum_exp = 0.0
                for score in scores:
                    e = math.exp(score - max_score)
                    exp_scores.append(e)
                    sum_exp += e

                attn = [e / sum_exp for e in exp_scores]

                for d in range(head_dim):
                    val = 0.0
                    for j in range(seq_len):
                        val += attn[j] * V[b, j, head_offset + d]
                    out[b, i, head_offset + d] = val

    Y = np.zeros((batch, seq_len, d_model), dtype=X.dtype)
    for b in range(batch):
        for i in range(seq_len):
            for j in range(d_model):
                y_val = 0.0
                for k in range(d_model):
                    y_val += out[b, i, k] * Wo[k, j]
                Y[b, i, j] = y_val

    return Y
