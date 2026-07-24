import numpy as np

def condition_number_via_svd(A: np.ndarray) -> float:
    """Return the 2-norm condition number of square matrix A.

    Computes via SVD: kappa = sigma_max / sigma_min.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # s is sorted descending by np.linalg.svd convention
    sigma_max = s[0]
    sigma_min = s[-1]
    if sigma_min == 0.0:
        return float("inf")
    return float(sigma_max / sigma_min)
