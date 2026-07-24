import numpy as np


def sparsegpt_then_gptq(
    W: np.ndarray,
    X: np.ndarray,
    bits: int = 4,
    lam_prune: float = 1e-2,
    damp: float = 1e-2,
) -> np.ndarray:
    """
    Two-stage compression recipe:

    1. SparseGPT 2:4 structured pruning: within every group of 4
       consecutive columns of each row, prune the 2 lowest-saliency
       weights (saliency = w^2 / diag(H_prune^-1)) and compensate the
       2 surviving weights via the inverse Hessian.
    2. GPTQ Hessian-ordered per-column int-`bits` quantization of the
       pruned ("surviving") weights, using a per-row symmetric scale
       and propagating rounding error to not-yet-quantized columns via
       a second (independently damped) inverse Hessian.

    W: (m, n) float64 weight matrix, n divisible by 4.
    X: (n, s) float64 calibration activations (rows = input features,
       columns = samples).

    Returns W_hat: (m, n) float64 pruned + quantized reconstruction.
    """
    raise NotImplementedError('your code here')
