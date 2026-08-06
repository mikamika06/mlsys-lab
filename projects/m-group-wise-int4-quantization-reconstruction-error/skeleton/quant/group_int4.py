import numpy as np


def quantize_dequantize_int4(x: np.ndarray, group_size: int):
    """Symmetrically quantize and dequantize x in contiguous groups of size group_size."""
    raise NotImplementedError


def compute_reconstruction_mse(x: np.ndarray, group_size: int):
    """Compute overall and per-group MSE between original x and INT4 reconstructed x."""
    raise NotImplementedError


def classify_elements(x: np.ndarray, group_size: int):
    """Classify elements into clamped and in_range, and compute error breakdown."""
    raise NotImplementedError
