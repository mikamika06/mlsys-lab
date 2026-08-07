import numpy as np


def assign_tokens(logits, temperature, threshold):
    e_x = np.exp((logits / temperature) - np.max(logits / temperature, axis=-1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    num_tokens = logits.shape[0]
    top2_idx = np.argsort(-probs, axis=-1)[:, :2]
    assignments = []
    for i in range(num_tokens):
        assignments.append((i, top2_idx[i, 0]))
        if probs[i, top2_idx[i, 1]] > threshold:
            assignments.append((i, top2_idx[i, 1]))
    return np.array(assignments)


def compute_load_imbalance(assignments, num_experts):
    if len(assignments) == 0:
        return 0.0
    counts = np.bincount(assignments[:, 1], minlength=num_experts)
    mean_load = len(assignments) / num_experts
    return float(np.max(counts) / mean_load)
