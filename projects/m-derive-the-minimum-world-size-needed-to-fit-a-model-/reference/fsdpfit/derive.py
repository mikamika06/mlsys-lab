import numpy as np


def derive_world_size(model_bytes, overhead_bytes, per_rank_budget):
    if model_bytes + overhead_bytes <= per_rank_budget:
        return 1
    effective_budget = per_rank_budget - overhead_bytes
    if effective_budget <= 0:
        raise ValueError("budget too small")
    ws = int(np.ceil(model_bytes / effective_budget))
    return max(1, ws)
