import numpy as np


def optimal_group_scales_under_mask(W: np.ndarray, M: np.ndarray, X: np.ndarray,
                                     group_size: int, bits: int = 4,
                                     alphas: np.ndarray = None):
    """Greedy per-row, per-group coordinate-descent scale search
    minimizing the X-weighted output MSE of a masked-then-quantized
    weight matrix.

    W: (O, I) float64 weight matrix.
    M: (O, I) mask (0/1), same shape.
    X: (n, I) float64 calibration activations.
    group_size: contiguous groups along axis 1 (I % group_size == 0).
    bits: quantizer bit width, qmax = 2**(bits-1) - 1.
    alphas: 1-D array of scale multipliers to grid-search; if None, use
        np.linspace(0.6, 1.4, 9).

    Let Wm = W * M.
    1. Initialize `what` with the naive per-group quantizer applied to
       every group of every row: scale = max(|seg|)/qmax (or 1.0 if
       all-zero), dequant = scale * clip(round(seg/scale), -qmax, qmax).
    2. For each row, one left-to-right pass over its groups: for group
       g, try every alpha, computing scale = alpha*max(|seg|)/qmax (or
       1.0 if all-zero) and that group's dequant; plug it into `what`
       for that row (every other group stays at its current value) and
       measure err = sum((X @ (Wm[o] - what[o]))**2). Commit whichever
       alpha gives the lowest err before moving to the next group.
    3. mse = mean((X @ Wm.T - X @ what.T)**2), over all n*O entries.

    Returns (group_scales, mse):
      group_scales: (O, I // group_size) float64.
      mse: float.
    """
    raise NotImplementedError('your code here')
