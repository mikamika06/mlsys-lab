import numpy as np

def nf4_levels():
    """Return the 16 NF4 level values as a float64 numpy array."""
    from scipy.stats import norm
    q = np.array([norm.ppf((i + 0.5) / 16) for i in range(16)])
    q_norm = q / np.max(np.abs(q))
    k = np.argmin(np.abs(q_norm))
    q_norm[k] = 0.0
    return q_norm.astype(np.float64)
