def read_singular_values_variance_explained(A):
    import numpy as np
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    total = np.sum(S**2)
    if total == 0:
        return np.zeros_like(S, dtype=np.float64)
    var_exp = (S**2) / total
    return np.cumsum(var_exp)
