import numpy as np


def quantize_k_tensor(k_tensor: np.ndarray) -> dict:
    """Quantize a K tensor in block q4_0 format along the last dimension."""
    raise NotImplementedError


def dequantize_k_tensor(q_dict: dict) -> np.ndarray:
    """Dequantize a quantized K tensor dict back into float32 array."""
    raise NotImplementedError


def compute_k_quant_stats(k_tensor: np.ndarray, q_dict: dict) -> dict:
    """Compute error and memory compression statistics for a quantized K tensor."""
    raise NotImplementedError
