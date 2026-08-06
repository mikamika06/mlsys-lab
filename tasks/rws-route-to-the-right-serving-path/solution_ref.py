import numpy as np

def route_tensor(weights: np.ndarray) -> str:
    if weights.ndim == 2 and weights.shape[1] % 4 == 0:
        rows = weights.shape[0]
        cols = weights.shape[1]
        is_tensor_core = True
        for i in range(rows):
            for j in range(0, cols, 4):
                nonzero_count = 0
                for k in range(4):
                    if weights[i, j + k] != 0:
                        nonzero_count += 1
                if nonzero_count != 2:
                    is_tensor_core = False
                    break
            if not is_tensor_core:
                break
        if is_tensor_core:
            return "tensor-core"

    total_elements = 0
    zero_count = 0
    rows = weights.shape[0]
    cols = weights.shape[1]
    for i in range(rows):
        for j in range(cols):
            total_elements += 1
            if weights[i, j] == 0:
                zero_count += 1

    sparsity = zero_count / total_elements
    if sparsity >= 0.5:
        return "csr"
    return "dense"
