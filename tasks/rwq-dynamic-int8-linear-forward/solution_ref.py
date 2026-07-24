import numpy as np


def int8_linear_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Dynamic int8 Linear forward: per-channel symmetric int8 weight quant
    (fixed ahead of time) + per-tensor dynamic asymmetric uint8 activation
    quant (from this call's own min/max), integer matmul with zero-point
    correction, then dequantize.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    # Per-channel symmetric int8 weight quantization.
    w_absmax = np.max(np.abs(W), axis=1)
    w_absmax = np.where(w_absmax == 0.0, 1.0, w_absmax)
    scale_w = w_absmax / 127.0
    W_q = np.clip(np.round(W / scale_w[:, None]), -127, 127).astype(np.int32)

    # Per-tensor dynamic asymmetric uint8 activation quantization.
    x_min = float(np.min(X))
    x_max = float(np.max(X))
    if x_max == x_min:
        x_max = x_min + 1e-8
    scale_x = (x_max - x_min) / 255.0
    zero_point = np.clip(np.round(-x_min / scale_x), 0, 255)
    X_q = np.clip(np.round(X / scale_x + zero_point), 0, 255).astype(np.int32)

    # Integer matmul with zero-point correction, then dequantize.
    acc = (X_q - zero_point) @ W_q.T
    Y = acc.astype(np.float64) * scale_x * scale_w[None, :]
    return Y
