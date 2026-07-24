import numpy as np

def int8_linear(X, W):
    """INT8 quantized linear: per-channel symmetric W, per-token symmetric X.

    X: float64 array [B, K]
    W: float64 array [N, K]
    Returns: float64 array [B, N]
    """
    # Per-channel symmetric quantization of W
    absmax_w = np.max(np.abs(W), axis=1)
    scale_w = np.where(absmax_w > 0, absmax_w / 127.0, 1.0)
    W_q = np.clip(np.round(W / scale_w[:, np.newaxis]), -128, 127).astype(np.int8)

    # Per-token symmetric quantization of X
    absmax_x = np.max(np.abs(X), axis=1)
    scale_x = np.where(absmax_x > 0, absmax_x / 127.0, 1.0)
    X_q = np.clip(np.round(X / scale_x[:, np.newaxis]), -128, 127).astype(np.int8)

    # INT8 matmul with int32 accumulation, then dequantize
    acc = X_q.astype(np.int32) @ W_q.astype(np.int32).T
    Y = acc.astype(np.float64) * (scale_x[:, np.newaxis] * scale_w[np.newaxis, :])
    return Y
