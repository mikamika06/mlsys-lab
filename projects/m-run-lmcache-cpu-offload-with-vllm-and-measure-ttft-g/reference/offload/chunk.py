import numpy as np


def find_optimal_chunk_size(token_lengths, bandwidth_profile):
    best_chunk = 64
    min_cost = float("inf")
    candidates = [64, 128, 256, 512, 1024, 2048]
    for c in candidates:
        cost = 0.0
        for length in token_lengths:
            chunks = np.ceil(length / c)
            overhead = chunks * 0.001 + (c / bandwidth_profile)
            cost += overhead
        if cost < min_cost:
            min_cost = cost
            best_chunk = c
    return int(best_chunk)
