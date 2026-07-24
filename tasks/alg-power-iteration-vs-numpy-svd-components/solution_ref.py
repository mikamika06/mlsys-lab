import numpy as np

def pca_power_iteration(X: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the first `k` principal components of a centered data matrix `X`
    using power iteration with deflation.  The algorithm is deterministic
    because it uses a fixed random seed.
    """
    rng = np.random.default_rng(0)
    d = X.shape[1]
    # Covariance matrix (not normalised by n-1, but that does not affect eigenvectors)
    C = X.T @ X

    comps: list[np.ndarray] = []

    for _ in range(k):
        # Initial random vector orthogonalised against previously found components
        v = rng.standard_normal(d)
        if comps:
            for u in comps:
                v -= np.dot(u, v) * u
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            # Rare numerical issue – start over with a fresh random vector
            v = rng.standard_normals(d)
            norm_v = np.linalg.norm(v)
        v /= norm_v

        # Power iteration, orthogonalising against previous components each step
        for _ in range(2000):
            w = C @ v
            if comps:
                for u in comps:
                    w -= np.dot(u, w) * u
            norm_w = np.linalg.norm(w)
            if norm_w == 0.0:
                break
            v_new = w / norm_w
            # Convergence check – relative change below tolerance
            if np.allclose(v, v_new, atol=1e-10):
                v = v_new
                break
            v = v_new

        comps.append(v.copy())

    return np.stack(comps)
