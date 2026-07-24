import numpy as np


def accumulate_grad(micro_batches: list[tuple[np.ndarray, np.ndarray]], w: np.ndarray) -> np.ndarray:
    """
    Gradient of mean squared error loss w.r.t. w, computed by accumulating
    each micro-batch's contribution and normalizing by the TOTAL example
    count -- exactly the gradient a single large batch would produce.
    """
    total = np.zeros_like(w, dtype=np.float64)
    N = 0
    for X_i, y_i in micro_batches:
        r_i = X_i @ w - y_i
        total += X_i.T @ r_i          # un-normalized per-microbatch contribution
        N += X_i.shape[0]
    return (2.0 / N) * total
