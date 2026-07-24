import numpy as np


def obq_column_step(W, H_inv, col, scale, nmax):
    """One GPTQ/OBQ column-quantize + Hessian error-propagation step.

    W: (rows, n) float64. H_inv: (n, n) float64 symmetric PD inverse-Hessian.
    col: int, column to quantize this step. scale: (rows,) float64 per-row
    symmetric quant scale for this column. nmax: positive int.

    q_col = clip(round(W[:, col] / scale), -nmax, nmax) * scale
    err = (W[:, col] - q_col) / H_inv[col, col]
    W_updated = copy of W with column `col` set to q_col, and columns
      col+1..n-1 corrected: W[:, col+1:] -= outer(err, H_inv[col, col+1:])
      (columns before col are left unchanged).

    Returns (q_col, W_updated).
    """
    raise NotImplementedError('your code here')
