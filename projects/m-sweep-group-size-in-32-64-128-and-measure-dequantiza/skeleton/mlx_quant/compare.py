"""Comparison of 4-bit vs 8-bit model footprint and drift."""

import numpy as np


def compare_bit_widths(weights: np.ndarray, group_size: int = 64) -> dict:
    """Compare size bytes and dequantization MSE for 4-bit vs 8-bit quantization."""
    raise NotImplementedError
