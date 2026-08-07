import numpy as np


class MaxCalibrator:
    """Tracks absolute maximum values across data batches to compute scale factors."""

    def __init__(self, max_bound: float = 127.0):
        self.max_bound = float(max_bound)
        self.amax = 0.0

    def collect(self, tensor: np.ndarray) -> None:
        raise NotImplementedError

    def compute_scale(self) -> float:
        raise NotImplementedError


def compute_max_scale(tensor: np.ndarray, max_bound: float = 127.0) -> float:
        raise NotImplementedError
