"""MoE top-k router with zeroed row diagnosis and recovery."""

import numpy as np


def route_tokens(logits, top_k, mask=None):
    logits = np.array(logits, dtype=np.float64)
    num_tokens, num_experts = logits.shape
    top_k = min(top_k, num_experts)

    if mask is not None:
        mask = np.array(mask, dtype=bool)
        masked_logits = np.where(mask, logits, -1e9)
    else:
        masked_logits = logits.copy()

    probs = np.zeros_like(masked_logits)
    for i in range(num_tokens):
        row = masked_logits[i]
        max_val = np.max(row)
        if max_val <= -1e8:
            exp_row = np.zeros_like(row)
        else:
            exp_row = np.exp(row - max_val)
            if mask is not None:
                exp_row = np.where(mask[i], exp_row, 0.0)
        s = np.sum(exp_row)
        if s > 0:
            probs[i] = exp_row / s
        else:
            probs[i] = np.zeros_like(row)

    top_k_indices = np.argsort(-probs, axis=1)[:, :top_k]
    top_k_weights = np.take_along_axis(probs, top_k_indices, axis=1)

    zero_row_diagnosed = []
    row_sums = np.sum(top_k_weights, axis=1)

    for i in range(num_tokens):
        if row_sums[i] <= 1e-12:
            zero_row_diagnosed.append(i)
            top_k_weights[i] = np.full(top_k, 1.0 / top_k, dtype=np.float64)
        else:
            top_k_weights[i] = top_k_weights[i] / row_sums[i]

    return {
        "indices": top_k_indices.astype(np.int64),
        "weights": top_k_weights,
        "zero_row_diagnosed": zero_row_diagnosed,
    }
