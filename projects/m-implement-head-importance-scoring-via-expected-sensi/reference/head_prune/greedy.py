import numpy as np


def compute_greedy_removal_order(importance_matrix, num_heads_to_remove):
    flat = []
    num_layers, num_heads = importance_matrix.shape
    for l in range(num_layers):
        for h in range(num_heads):
            flat.append((importance_matrix[l, h], l, h))
    flat.sort(key=lambda x: x[0])
    removal = []
    for i in range(min(num_heads_to_remove, len(flat))):
        _, l, h = flat[i]
        removal.append((l, h))
    return removal
