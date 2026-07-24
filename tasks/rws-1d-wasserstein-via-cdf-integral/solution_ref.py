import numpy as np


def wasserstein1_cdf_integral(u: np.ndarray, v: np.ndarray) -> float:
    """Exact 1-D Wasserstein-1 distance between two empirical samples of
    possibly unequal length, via the merged-CDF step integral
    W1(u,v) = sum_k |F_U(z_k) - F_V(z_k)| * (z_{k+1} - z_k).
    """
    u = np.sort(np.asarray(u, dtype=np.float64).ravel())
    v = np.sort(np.asarray(v, dtype=np.float64).ravel())

    all_values = np.concatenate([u, v])
    all_values.sort(kind="mergesort")
    deltas = np.diff(all_values)

    u_cdf = np.searchsorted(u, all_values[:-1], side="right") / u.size
    v_cdf = np.searchsorted(v, all_values[:-1], side="right") / v.size

    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))
