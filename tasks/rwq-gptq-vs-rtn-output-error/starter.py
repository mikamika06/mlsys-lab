import numpy as np


def gptq_vs_rtn_output_error(W: np.ndarray, X: np.ndarray, bits: int):
    """Quantize W (d_out, d_in) to `bits` via RTN and via GPTQ (same per-row
    grid, sequential Hessian-based error feedback using X's H = X^T X / n).
    Return (mse_rtn, mse_gptq), the mean squared output error of X @ W.T vs
    X @ Wq.T for each method."""
    raise NotImplementedError('your code here')
