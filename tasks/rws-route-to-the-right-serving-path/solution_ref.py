import numpy as np

def route_tensor(weights: np.ndarray) -> str:
    # Check valid 2:4 structured sparsity
    if weights.ndim == 2 and weights.shape[1] % 4 == 0:
        # Reshape to (rows, groups, 4)
        groups = weights.reshape(weights.shape[0], -1, 4)
        nonzeros = np.count_nonzero(groups, axis=2)
        if np.all(nonzeros == 2):
            return "tensor-core"

    # Sparsity ratio
    sparsity = np.mean(weights == 0)
    if sparsity >= 0.5:
        return "csr"
    return "dense"
