import numpy as np


def hqq_optimize(W, scale, zero0, qmin, qmax, lp_norm, beta0, kappa, iters):
    """
    Run `iters` HQQ half-quadratic passes to refine the zero-point z
    (scale is fixed), then quantize W one final time with the converged z.
    Return (W_q, z, W_dequant). See task.md for the exact recursion,
    including the Lp shrink operator.
    """
    raise NotImplementedError('your code here')
