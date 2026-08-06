import numpy as np


def quantize_tensor(W, num_bits):
    qmin = 0
    qmax = (2 ** num_bits) - 1
    w_min, w_max = np.min(W), np.max(W)
    if w_max == w_min:
        return np.copy(W)
    scale = (w_max - w_min) / qmax
    zero_point = np.round(-w_min / scale)
    q = np.clip(np.round(W / scale + zero_point), qmin, qmax)
    w_hat = (q - zero_point) * scale
    return w_hat


def prune_tensor(W, sparsity):
    if sparsity <= 0.0:
        return np.copy(W)
    if sparsity >= 1.0:
        return np.zeros_like(W)
    W_out = np.copy(W)
    out_dim, in_dim = W.shape
    k = int(np.round(in_dim * sparsity))
    if k == 0:
        return W_out
    for i in range(out_dim):
        row = W_out[i]
        idx = np.argsort(np.abs(row))[:k]
        row[idx] = 0.0
    return W_out


def compare_order_error(W, X, sparsity, num_bits):
    Y = W @ X
    
    W_ptq = prune_tensor(W, sparsity)
    W_ptq = quantize_tensor(W_ptq, num_bits)
    Y_ptq = W_ptq @ X
    ptq_mse = float(np.mean((Y - Y_ptq) ** 2))
    
    W_qtp = quantize_tensor(W, num_bits)
    W_qtp = prune_tensor(W_qtp, sparsity)
    Y_qtp = W_qtp @ X
    qtp_mse = float(np.mean((Y - Y_qtp) ** 2))
    
    return {
        "ptq_mse": ptq_mse,
        "qtp_mse": qtp_mse
    }
