import numpy as np

def compute_scales_and_size(K: np.ndarray, V: np.ndarray):
    """
    Compute per‑row absolute‑max scales for K and V and the memory size ratio.
    """
    # Ensure float64 output for scales
    scales_K = np.max(np.abs(K), axis=1).astype(np.float64)
    scales_V = np.max(np.abs(V), axis=1).astype(np.float64)

    n, dK = K.shape
    _, dV = V.shape

    # Original memory in bytes (float32)
    orig_bytes = (n * dK + n * dV) * 4

    # Quantized memory: int8 per element + float32 scale per row per matrix
    quant_bytes = (n * dK + n * dV) * 1 + n * 4 * 2

    size_ratio = orig_bytes / quant_bytes
    return scales_K, scales_V, float(size_ratio)
