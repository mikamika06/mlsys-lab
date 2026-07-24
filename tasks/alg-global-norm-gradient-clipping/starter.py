import numpy as np
from typing import List

def clip_global_norm(grads: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    """
    Clips a list of gradient tensors by their global L2 norm.
    """
    raise NotImplementedError
