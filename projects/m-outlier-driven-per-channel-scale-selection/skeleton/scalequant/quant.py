import numpy as np


def compute_minmax_scales(x: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """Compute per-channel min-max scales."""
    raise NotImplementedError


def quantize_dequantize_channel(x: np.ndarray, scale: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """Fake-quantize and dequantize per channel."""
    raise NotImplementedError


def channel_mse(x: np.ndarray, deq: np.ndarray) -> np.ndarray:
    """Compute mean squared error per channel."""
    raise NotImplementedError
