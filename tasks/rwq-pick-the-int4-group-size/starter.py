import numpy as np


def pick_int4_group_size(W: np.ndarray, group_sizes=(32, 64, 128, 256),
                          bits: int = 4, lam: float = 0.02):
    """Pick the group_size minimizing reconstruction MSE + scale overhead.

    W: 1-D float64 array; len(W) divisible by every value in group_sizes.
    group_sizes: candidate group sizes, in the given order.
    bits: quantizer bit width, qmax = 2**(bits-1) - 1.
    lam: overhead weight.

    For each candidate gs: quantize W in contiguous groups of gs elements
    with a symmetric int-`bits` quantizer (per-group scale = max(|group|) /
    qmax), compute mse(gs) = mean((W_hat - W)**2), and
    cost(gs) = mse(gs) + lam * (16.0 / gs)   (16.0 = fp16 bits per stored
    scale, amortized over gs weights).

    Returns (best_group_size, best_cost, costs):
      best_group_size: int, the candidate with minimum cost (ties -> earliest).
      best_cost: float, its cost.
      costs: float64 array, one cost per candidate, in group_sizes order.
    """
    raise NotImplementedError('your code here')
