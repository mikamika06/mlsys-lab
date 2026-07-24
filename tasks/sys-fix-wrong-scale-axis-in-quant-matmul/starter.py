import numpy as np

def quant_matmul(x, codes, scales):
    """Dequantize int8 weight codes with per-channel scales and matmul.

    Args:
        x: float64 activations, shape (M, K).
        codes: int8 quantized weight codes, shape (K, N).
        scales: float64 per-output-channel scales, shape (N,).
    Returns:
        float64 result, shape (M, N).
    """
    W = codes.astype(np.float64) * scales[:, np.newaxis]
    return x @ W
