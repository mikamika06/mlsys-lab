import numpy as np


def compute_scales(tensor: np.ndarray) -> np.ndarray:
    flat = tensor.reshape(-1, 32)
    max_vals = np.max(np.abs(flat), axis=1)
    scales = max_vals / 7.0
    return scales


def simulate_q4_0(tensor: np.ndarray) -> np.ndarray:
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, 32)
    max_vals = np.max(np.abs(flat), axis=1)
    scales = max_vals / 7.0
    scales_expanded = np.where(scales == 0, 1e-12, scales)[:, np.newaxis]
    quantized = np.clip(np.round(flat / scales_expanded), -8, 7)
    dequantized = quantized * scales_expanded
    return dequantized.reshape(orig_shape)
