import numpy as np


def moe_dispatch_combine(X: np.ndarray, expert_idx: np.ndarray,
                          gate_weight: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Route each token to its assigned expert (expert_idx), apply that
    expert's linear transform W[e], and combine results back into the
    original token order, scaled by the per-token gate_weight.

    X: (n, d) token embeddings.
    expert_idx: (n,) int array, expert assigned to each token.
    gate_weight: (n,) float array, combine weight for each token.
    W: (E, d, d) per-expert weight matrices.
    Returns: (n, d) array, y_i = gate_weight[i] * (X[i] @ W[expert_idx[i]]).
    """
    raise NotImplementedError('your code here')
