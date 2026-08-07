import numpy as np


def compute_layer_mse(W, W_quant, X):
    out_ref = W @ X
    out_quant = W_quant @ X
    diff = out_ref - out_quant
    return float(np.mean(diff ** 2))
