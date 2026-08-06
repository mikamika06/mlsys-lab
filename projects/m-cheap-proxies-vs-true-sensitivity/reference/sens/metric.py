import numpy as np


def compute_true_sensitivity(layer):
    w = layer["weight"]
    x = layer["activation"]
    out_orig = np.matmul(x, w.T)
    quant_w = np.round(w * 4.0) / 4.0
    out_quant = np.matmul(x, quant_w.T)
    return float(np.mean((out_orig - out_quant) ** 2))
