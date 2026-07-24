import numpy as np

def deepnorm_scaled_residuals(residuals, alpha):
    """
    Compute DeepNorm scaled residual norms.

    Parameters
    ----------
    residuals : list[np.ndarray]
        List of 1‑D arrays representing the residual at each transformer block.
    alpha : float
        Scaling hyper‑parameter (>0). The i-th residual norm is multiplied by alpha**i.

    Returns
    -------
    np.ndarray
        1‑D array of scaled L2 norms, dtype float64.
    """
    norms = []
    for i, r in enumerate(residuals):
        norm = np.linalg.norm(r.astype(np.float64), ord=2)
        scaled = float(norm) * (float(alpha) ** i)
        norms.append(scaled)
    return np.array(norms, dtype=np.float64)
