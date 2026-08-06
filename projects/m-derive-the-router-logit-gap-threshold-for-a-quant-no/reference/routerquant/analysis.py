import numpy as np


def compute_quant_error(weights, quantized_weights):
    diff = weights - quantized_weights
    mse = np.mean(diff ** 2)
    max_err = np.max(np.abs(diff))
    return {"mse": float(mse), "max_error": float(max_err)}
