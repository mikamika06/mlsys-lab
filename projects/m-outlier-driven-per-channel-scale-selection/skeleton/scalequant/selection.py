import numpy as np


def select_outlier_scales(x: np.ndarray, num_bits: int = 8, n_candidates: int = 32) -> np.ndarray:
    """Select optimal per-channel scales by minimizing MSE."""
    raise NotImplementedError


def evaluate_quantization_loss(x: np.ndarray, scales: np.ndarray, num_bits: int = 8) -> float:
    """Compute total mean squared error over all channels."""
    raise NotImplementedError
