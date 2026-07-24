import numpy as np


def gptq_vs_rtn_error_ratio(W, X, bits):
    """
    W: (d_out, d_in) weight matrix.
    X: (n, d_in) calibration activations.
    bits: quantization bit-width b.

    Quantize W with plain round-to-nearest (per-output-row symmetric
    scales) and with full column-by-column GPTQ (Hessian H = X^T X +
    0.01 * mean(diag(H)) * I, upper-triangular Cholesky factor U of
    H^-1, scales frozen from the original W, sequential column
    compensation).

    Returns the scalar ratio (float) of layer-output Frobenius-norm
    reconstruction errors:

        ||X @ W_gptq.T - X @ W.T|| / ||X @ W_rtn.T - X @ W.T||
    """
    raise NotImplementedError('your code here')
