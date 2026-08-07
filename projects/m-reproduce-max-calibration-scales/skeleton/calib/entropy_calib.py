import numpy as np


def compute_entropy_scale(
    tensor: np.ndarray,
    num_bins: int = 2048,
    num_quant_steps: int = 128,
    max_bound: float = 127.0,
) -> float:
    raise NotImplementedError
