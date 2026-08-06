import numpy as np
import math

def fixed_budget_kv(keys: np.ndarray, values: np.ndarray, budget: int) -> dict:
    """
    Return a dictionary mapping the selected keys to their corresponding rows in `values`.
    Selection is based on the largest Euclidean norm of each row.
    If `budget >= len(keys)` all entries are returned.
    """
    T = len(keys)
    if budget >= T:
        return {int(k): v for k, v in zip(keys, values)}
    norms = []
    for i in range(T):
        row = values[i]
        s = 0.0
        for val in row:
            s += val * val
        norms.append(math.sqrt(s))
    idx_sorted = sorted(range(T), key=lambda i: norms[i], reverse=True)
    selected_idx = idx_sorted[:budget]
    return {int(keys[i]): values[i] for i in selected_idx}
