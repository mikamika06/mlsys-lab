import numpy as np


def allocate_eora_ranks(weights: list[np.ndarray], total_budget_params: int, base_bits: int) -> list[int]:
    n_layers = len(weights)
    ranks = [0] * n_layers
    current_params = 0
    for idx, w in enumerate(weights):
        rows, cols = w.shape
        params_per_rank = rows + cols
        if current_params + params_per_rank <= total_budget_params:
            ranks[idx] = 1
            current_params += params_per_rank

    for idx, w in enumerate(weights):
        rows, cols = w.shape
        params_per_rank = rows + cols
        while current_params + params_per_rank <= total_budget_params:
            ranks[idx] += 1
            current_params += params_per_rank
            max_r = min(rows, cols)
            if ranks[idx] >= max_r:
                break
    return ranks
