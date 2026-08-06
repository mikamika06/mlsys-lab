import numpy as np


def derive_load_balanced_sharding(num_layers, layer_weights, num_ranks=4):
    """Derive optimal layer assignment across num_ranks minimizing max load and comm transitions."""
    weights = np.array(layer_weights, dtype=np.float64)
    n = len(weights)

    dp = np.full((n + 1, num_ranks + 1), fill_value=np.inf)
    parent = np.zeros((n + 1, num_ranks + 1), dtype=int)
    dp[0, 0] = 0.0

    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(weights)

    for k in range(1, num_ranks + 1):
        for i in range(k, n + 1):
            for j in range(k - 1, i):
                cost = prefix[i] - prefix[j]
                max_cost = max(dp[j, k - 1], cost)
                if max_cost < dp[i, k]:
                    dp[i, k] = max_cost
                    parent[i, k] = j

    assignments = np.zeros(n, dtype=int)
    curr = n
    for k in range(num_ranks, 0, -1):
        prev = parent[curr, k]
        assignments[prev:curr] = k - 1
        curr = prev

    return assignments.tolist()
