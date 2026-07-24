import numpy as np


def fused_elementwise_chain(X: np.ndarray, bias: np.ndarray, residual: np.ndarray, scale: float) -> np.ndarray:
    """
    Vertical (pointwise) kernel fusion: compute the epilogue chain

        h = X + bias                     (bias-add, broadcast over the last axis)
        h = gelu_tanh(h)                 (tanh-approximation GELU)
        h = h + residual                 (residual add)
        h = h * scale                    (output scale)

    as ONE fused per-element expression, exactly as a real fused kernel
    (e.g. a Triton/CUDA epilogue) would: every element of the output is
    produced by reading its X/bias/residual/scale inputs once and writing
    the result once, without ever writing a full-size intermediate array
    for each individual stage back out to memory.

    X        : (batch, dim)
    bias     : (dim,)       -- broadcasts over the batch axis
    residual : (batch, dim)
    scale    : python float

    Returns an array the same shape as X.
    """
    X = np.asarray(X, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)

    z = X + bias
    return (0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3))) + residual) * scale
