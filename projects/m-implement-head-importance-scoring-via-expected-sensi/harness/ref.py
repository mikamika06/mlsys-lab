import numpy as np

def compute_importance(activations, gradients):
    return np.mean(np.abs(activations * gradients), axis=(0, 2))

def compute_removal_order(importance_matrix):
    flat_indices = np.argsort(importance_matrix.ravel())
    rows, cols = np.unravel_index(flat_indices, importance_matrix.shape)
    return list(zip(rows.tolist(), cols.tolist()))

def measure_latency(base_latency, removed_count, total_heads):
    fraction = removed_count / total_heads
    return float(base_latency * (1.0 - 0.5 * fraction))
