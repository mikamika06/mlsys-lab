import numpy as np

def compute_int8_scales(X, W):
    """Compute per-row X scales and per-column W scales for int8 quantization."""
    scale_x = np.max(np.abs(X), axis=1).astype(np.float64) / 127.0
    scale_w = np.max(np.abs(W), axis=0).astype(np.float64) / 127.0
    return scale_x, scale_w
