"""NF4 quantization error simulation module."""

from typing import Dict
import numpy as np


NF4_LEVELS = np.array([
    -1.0, -0.6961928010000001, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791859447956085,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=np.float64)


def _quantize_dequantize_nf4(x: np.ndarray) -> np.ndarray:
    abs_max = np.max(np.abs(x))
    if abs_max == 0:
        return np.zeros_like(x)
    normalized = x / abs_max
    diffs = np.abs(normalized[..., None] - NF4_LEVELS)
    indices = np.argmin(diffs, axis=-1)
    dequantized = NF4_LEVELS[indices] * abs_max
    return dequantized


def simulate_nf4_compounding_error(tensor: np.ndarray, num_cycles: int = 10) -> Dict[str, np.ndarray]:
    """Simulate compounding error across repeated NF4 quantize-dequantize cycles."""
    current = np.asarray(tensor, dtype=np.float64)
    orig = current.copy()

    mse_history = []
    max_err_history = []

    for _ in range(num_cycles):
        current = _quantize_dequantize_nf4(current)
        mse = np.mean((orig - current) ** 2)
        max_err = np.max(np.abs(orig - current))
        mse_history.append(mse)
        max_err_history.append(max_err)

    return {
        "final_tensor": current,
        "mse_history": np.array(mse_history, dtype=np.float64),
        "max_err_history": np.array(max_err_history, dtype=np.float64),
    }
