import numpy as np


def prune_then_quantize(W, mask, scale, zero_point):
    w_sparse = np.where(mask == 0, 0.0, W)
    q = np.clip(np.round(w_sparse / scale) + zero_point, -8, 7)
    return (q - zero_point) * scale


def quantize_then_prune(W, mask, scale, zero_point):
    q = np.clip(np.round(W / scale) + zero_point, -8, 7)
    w_deq = (q - zero_point) * scale
    return np.where(mask == 0, 0.0, w_deq)


def evaluate_accuracy(W_orig, W_compressed):
    mse = np.mean((W_orig - W_compressed) ** 2)
    return float(1.0 / (1.0 + mse))
