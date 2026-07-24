import numpy as np

def apply_causal_mask(S):
    """Return a float64 copy of S with all entries above the diagonal set to -inf."""
    S_masked = np.array(S, dtype=np.float64)
    n = S_masked.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            S_masked[i, j] = -np.inf
    return S_masked
