import math
import numpy as np


def _quant_rows_int4(V: np.ndarray) -> np.ndarray:
    rows, cols = V.shape
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        absmax = 0.0
        for j in range(cols):
            val = abs(V[i, j])
            if val > absmax:
                absmax = val
        if absmax == 0:
            absmax = 1e-9
        delta = absmax / 7.0
        for j in range(cols):
            val = V[i, j] / delta
            r = round(val)
            if r < -8:
                r = -8
            elif r > 7:
                r = 7
            out[i, j] = r * delta
    return out


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

    batch, in_dim = X.shape
    out_dim = W.shape[0]

    Y_true = np.empty((batch, out_dim), dtype=np.float64)
    for b in range(batch):
        for o in range(out_dim):
            acc = 0.0
            for i in range(in_dim):
                acc += X[b, i] * W[o, i]
            Y_true[b, o] = acc

    norm_Y_true_sq = 0.0
    for b in range(batch):
        for o in range(out_dim):
            v = Y_true[b, o]
            norm_Y_true_sq += v * v
    norm_Y_true = math.sqrt(norm_Y_true_sq)

    W_hat_rtn = _quant_rows_int4(W)

    diff_rtn_sq = 0.0
    for b in range(batch):
        for o in range(out_dim):
            acc = 0.0
            for i in range(in_dim):
                acc += X[b, i] * W_hat_rtn[o, i]
            diff = acc - Y_true[b, o]
            diff_rtn_sq += diff * diff
    err_rtn = float(math.sqrt(diff_rtn_sq) / norm_Y_true)

    s = np.empty(in_dim, dtype=np.float64)
    for j in range(in_dim):
        acc = 0.0
        for b in range(batch):
            acc += abs(X[b, j])
        s[j] = acc / batch

    W_scaled = np.empty((out_dim, in_dim), dtype=np.float64)
    for o in range(out_dim):
        for i in range(in_dim):
            W_scaled[o, i] = W[o, i] * s[i]

    W_hat_scaled = _quant_rows_int4(W_scaled)

    W_hat_awq = np.empty((out_dim, in_dim), dtype=np.float64)
    for o in range(out_dim):
        for i in range(in_dim):
            W_hat_awq[o, i] = W_hat_scaled[o, i] / s[i]

    diff_awq_sq = 0.0
    for b in range(batch):
        for o in range(out_dim):
            acc = 0.0
            for i in range(in_dim):
                acc += X[b, i] * W_hat_awq[o, i]
            diff = acc - Y_true[b, o]
            diff_awq_sq += diff * diff
    err_awq = float(math.sqrt(diff_awq_sq) / norm_Y_true)

    reduction = 1.0 - err_awq / err_rtn
    return err_rtn, err_awq, reduction
