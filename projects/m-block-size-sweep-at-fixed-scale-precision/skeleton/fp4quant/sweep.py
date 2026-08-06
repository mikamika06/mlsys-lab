import numpy as np
from typing import List, Tuple


def sweep_block_size(x: np.ndarray, block_sizes: List[int]) -> Tuple[int, float]:
    """Sweep candidate block sizes at fixed scale precision and return (argmin_index, min_error)."""
    raise NotImplementedError
