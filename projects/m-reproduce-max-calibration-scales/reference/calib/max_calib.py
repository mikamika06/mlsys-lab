import numpy as np


class MaxCalibrator:
    """Tracks absolute maximum values across data batches to compute scale factors."""

    def __init__(self, max_bound: float = 127.0):
        self.max_bound = float(max_bound)
        self.amax = 0.0

    def collect(self, tensor: np.ndarray) -> None:
        if tensor.size > 0:
            batch_max = float(np.max(np.abs(tensor)))
            self.amax = max(self.amax, batch_max)

    def compute_scale(self) -> float:
        if self.amax == 0.0:
            return 1.0
        return self.amax / self.max_bound


def compute_max_scale(tensor: np.ndarray, max_bound: float = 127.0) -> float:
    calibrator = MaxCalibrator(max_bound=max_bound)
    calibrator.collect(tensor)
    return calibrator.compute_scale()
