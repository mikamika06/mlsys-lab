"""Group size sweep and MSE calculation."""

import numpy as np


def quantize_affine(weights: np.ndarray, group_size: int, bits: int = 4):
    """Quantize weights with affine scale and bias per group."""
    raise NotImplementedError


def dequantize_affine(
    qweight: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int
) -> np.ndarray:
    """Dequantize uint8 values back to float weights using scale and bias."""
    raise NotImplementedError


def sweep_group_size_mse(weights: np.ndarray, group_sizes=(32, 64, 128), bits: int = 4) -> dict:
    """Sweep group sizes and compute dequantization MSE relative to weights."""
    raise NotImplementedError
