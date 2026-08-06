import numpy as np


def derive_logit_gap_threshold(logits, weights, quantized_weights, hidden_states):
    sorted_indices = np.argsort(logits, axis=-1)
    top1_idx = sorted_indices[..., -1]
    top2_idx = sorted_indices[..., -2]
    batch_size = logits.shape[0]
    thresholds = np.zeros(batch_size)
    for i in range(batch_size):
        t1 = top1_idx[i]
        t2 = top2_idx[i]
        gap = logits[i, t1] - logits[i, t2]
        thresholds[i] = abs(gap)
    return thresholds
