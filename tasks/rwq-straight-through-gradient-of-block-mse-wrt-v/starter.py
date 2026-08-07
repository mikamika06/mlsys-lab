def ste_block_mse_grad_wrt_v(X: list[list[float]], W: list[list[float]], V: list[list[float]], scale: list[float], bits: int) -> list[list[float]]:
    """Straight-through-estimator gradient of the block MSE loss wrt V.

    X: (B, I) float64 activations.
    W: (O, I) float64 original full-precision weight (the target).
    V: (O, I) float64 current learnable weight parameter.
    scale: (O,) float64, positive, one fixed scale per output row.
    bits: quantizer bit width, qmax = 2**(bits-1) - 1.

    1. r = V / scale[:, None]; mask = 1 where |r| <= qmax + 0.5 else 0.
    2. Wq = scale[:, None] * clip(round(r), -qmax, qmax).
       pred = X @ Wq.T, target = X @ W.T, diff = pred - target.
    3. Return mask * (2.0 / (B * O)) * (diff.T @ X), shape (O, I).

    (This is the *defined* straight-through gradient, not the true
    analytic gradient of the loss -- round() has zero derivative almost
    everywhere, so STE substitutes "identity, except zero where the
    clip saturates" by convention.)
    """
    raise NotImplementedError('your code here')
