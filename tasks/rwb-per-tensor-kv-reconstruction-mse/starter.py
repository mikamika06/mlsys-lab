import numpy as np


def kv_fp8_reconstruction_mse(K: np.ndarray, V: np.ndarray) -> dict:
    """Quantize K and V to E4M3 with an independent PER-TENSOR absmax
    scale for each (scale = max(|X|) / 448), dequantize, and report each
    tensor's reconstruction MSE.

    K, V : arbitrary-shape float arrays.

    Returns {"mse_k": float, "mse_v": float}.
    """
    raise NotImplementedError('your code here')
