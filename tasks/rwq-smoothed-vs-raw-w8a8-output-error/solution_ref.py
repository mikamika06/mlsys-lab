import numpy as np


def _quant_int8_pertensor(T):
    scale = np.max(np.abs(T)) / 127.0 + 1e-12
    codes = np.clip(np.round(T / scale), -127, 127).astype(np.int8)
    return codes.astype(np.float32) * scale


def w8a8_output_errors(X, W, s):
    """Compute W8A8 MSE for raw and SmoothQuant-smoothed quantization."""
    Y_ref = X.astype(np.float64) @ W.astype(np.float64)

    # Raw
    X_dq = _quant_int8_pertensor(X)
    W_dq = _quant_int8_pertensor(W)
    Y_raw = X_dq.astype(np.float64) @ W_dq.astype(np.float64)
    mse_raw = float(np.mean((Y_raw - Y_ref) ** 2))

    # Smoothed
    s_col = s.reshape(1, -1)
    X_hat = X / s_col
    W_hat = W * s.reshape(-1, 1)
    X_hat_dq = _quant_int8_pertensor(X_hat)
    W_hat_dq = _quant_int8_pertensor(W_hat)
    Y_smooth = X_hat_dq.astype(np.float64) @ W_hat_dq.astype(np.float64)
    mse_smooth = float(np.mean((Y_smooth - Y_ref) ** 2))

    return mse_raw, mse_smooth
