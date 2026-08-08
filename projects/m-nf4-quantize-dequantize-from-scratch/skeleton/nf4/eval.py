import numpy as np


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error between original and reconstructed arrays."""
    raise NotImplementedError


def evaluate_format_errors(weights: dict, block_size: int = 64) -> dict:
    """Evaluate MSE error for NF4, FP4, and INT4 formats across weight distributions."""
    raise NotImplementedError
