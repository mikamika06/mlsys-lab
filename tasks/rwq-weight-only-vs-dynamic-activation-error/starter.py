import numpy as np


def weight_only_vs_dynamic_mse(x: np.ndarray, W: np.ndarray):
    """
    Return (mse_weight_only, mse_dynamic): output MSE of a linear layer
    y = x @ W.T vs the full-precision reference, for (a) int8 weight-only
    quantization (only W quantized, per-row symmetric) and (b) int8
    dynamic quantization (W and x both quantized, x per-row symmetric,
    recomputed from the batch). See task.md for the exact formulas.
    """
    raise NotImplementedError('your code here')
