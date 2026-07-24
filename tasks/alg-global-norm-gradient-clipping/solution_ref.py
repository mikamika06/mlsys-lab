import numpy as np
from typing import List

def clip_global_norm(grads: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    total_norm = np.linalg.norm([np.linalg.norm(g) for g in grads])
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * coef for g in grads]
