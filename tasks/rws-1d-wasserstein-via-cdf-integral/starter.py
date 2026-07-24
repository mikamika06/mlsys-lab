import numpy as np


def wasserstein1_cdf_integral(u: np.ndarray, v: np.ndarray) -> float:
    """Exact 1-D Wasserstein-1 distance between two empirical samples of
    possibly unequal length, via the merged-CDF step integral
    W1(u,v) = sum_k |F_U(z_k) - F_V(z_k)| * (z_{k+1} - z_k).
    """
    raise NotImplementedError('your code here')
