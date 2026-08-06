import numpy as np


def quantize_fp8_per_head(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_fp8 = 448.0
    eps = 1e-12
    max_vals = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.maximum(max_vals / max_fp8, eps)
    q = np.clip(np.round(x / scale), -448.0, 448.0)
    return q, scale


def dequantize_fp8_per_head(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q * scale
