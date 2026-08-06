import math
from typing import List

def clip_global_norm(grads: List[List[float]], max_norm: float) -> List[List[float]]:
    total_norm_sq = 0.0
    for g in grads:
        for val in g:
            total_norm_sq += val * val

    total_norm = math.sqrt(total_norm_sq)
    coef = min(1.0, max_norm / (total_norm + 1e-6))

    clipped_grads = []
    for g in grads:
        new_g = []
        for val in g:
            new_g.append(val * coef)
        clipped_grads.append(new_g)

    return clipped_grads
