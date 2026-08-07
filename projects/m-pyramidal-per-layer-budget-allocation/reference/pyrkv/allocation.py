def compute_pyramidal_allocation(num_layers: int, total_budget: int, min_layer_budget: int) -> list[int]:
    import numpy as np
    if num_layers <= 0:
        return []
    x = np.linspace(-1.0, 1.0, num_layers)
    weights = 1.0 + 2.0 * (x ** 2)
    raw = weights / np.sum(weights) * total_budget
    alloc = np.maximum(min_layer_budget, np.round(raw)).astype(int)
    diff = total_budget - int(np.sum(alloc))
    if diff != 0:
        idx = np.argsort(weights)[::-1]
        step = 1 if diff > 0 else -1
        rem = abs(diff)
        for i in range(rem):
            alloc[idx[i % num_layers]] += step
    return [int(v) for v in alloc]
