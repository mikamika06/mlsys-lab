import numpy as np

def l2_vs_cosine_neighbor_set_divergence(X: np.ndarray, k: int) -> np.ndarray:
    """
    For each row in X compute the top‑k nearest neighbors under L2 distance
    (excluding self) and the top‑k most similar neighbors under cosine similarity.
    Return a boolean array where True indicates the two neighbor sets differ.
    """
    n = X.shape[0]

    # L2 distances
    norms_sq = np.sum(X**2, axis=1)
    D = norms_sq[:, None] + norms_sq[None, :] - 2 * X.dot(X.T)
    np.fill_diagonal(D, np.inf)
    l2_neighbors = np.argsort(D, axis=1)[:, :k]

    # Cosine similarity
    X_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / X_norms[:, None]
    cos_sim = X_normalized @ X_normalized.T
    np.fill_diagonal(cos_sim, -np.inf)
    cos_neighbors = np.argsort(-cos_sim, axis=1)[:, :k]

    return np.array([set(l2_neighbors[i]) != set(cos_neighbors[i])
                     for i in range(n)], dtype=bool)
