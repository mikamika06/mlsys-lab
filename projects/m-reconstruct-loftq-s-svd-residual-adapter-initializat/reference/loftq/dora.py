import numpy as np


def dora_forward(x, W_q, A, B, g, use_dora):
    combined = W_q + B @ A
    if use_dora:
        col_norms = np.linalg.norm(combined, axis=1, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-8)
        normalized_weight = combined / col_norms
        weight = g[:, None] * normalized_weight
        return x @ weight.T
    else:
        return x @ W_q.T + x @ A.T @ B.T
