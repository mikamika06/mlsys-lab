import numpy as np


def _quant_rows_int4(V: np.ndarray) -> np.ndarray:
    absmax = np.max(np.abs(V), axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1e-9, absmax)
    delta = absmax / 7.0
    return np.clip(np.round(V / delta), -8, 7) * delta


def compare_awq_rtn_error(W: np.ndarray, X: np.ndarray):
    """
    Compare plain RTN INT4 quantization of W against AWQ-scaled INT4
    quantization, on the linear layer Y = X @ W.T.

    AWQ scale: s_j = mean_b |X[b, j]| (per-input-channel average
    activation magnitude) -- scale W up by s before quantizing (so
    salient/high-activation channels get more of the quantization grid's
    precision), quantize, then scale back down.

    Returns (err_rtn, err_awq, reduction), where err_* is the relative
    Frobenius-norm output error vs the exact float layer, and
    reduction = 1 - err_awq / err_rtn (fraction of RTN's error AWQ removes).
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y_true = X @ W.T

    W_hat_rtn = _quant_rows_int4(W)
    err_rtn = float(np.linalg.norm(X @ W_hat_rtn.T - Y_true) / np.linalg.norm(Y_true))

    s = np.mean(np.abs(X), axis=0)
    W_scaled = W * s[None, :]
    W_hat_scaled = _quant_rows_int4(W_scaled)
    W_hat_awq = W_hat_scaled / s[None, :]
    err_awq = float(np.linalg.norm(X @ W_hat_awq.T - Y_true) / np.linalg.norm(Y_true))

    reduction = 1.0 - err_awq / err_rtn
    return err_rtn, err_awq, reduction
