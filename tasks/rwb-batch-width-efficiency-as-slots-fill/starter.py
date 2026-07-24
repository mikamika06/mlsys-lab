import numpy as np


def batch_width_utilization(occupancy: np.ndarray, N: int) -> dict:
    """Per-step and mean batch-width utilization from an occupancy trace.

    Returns {"per_step": np.ndarray of shape (T,), "mean": float}.
    """
    raise NotImplementedError('your code here')
