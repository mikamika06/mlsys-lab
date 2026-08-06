import numpy as np
import math

def l2_vs_cosine_neighbor_set_divergence(X: np.ndarray, k: int) -> np.ndarray:
    """
    For each row in X compute the top‑k nearest neighbors under L2 distance
    (excluding self) and the top‑k most similar neighbors under cosine similarity.
    Return a boolean array where True indicates the two neighbor sets differ.
    """
    n = X.shape[0]
    d = X.shape[1]

    norms_sq = [0.0] * n
    for i in range(n):
        s = 0.0
        for c in range(d):
            val = float(X[i, c])
            s += val * val
        norms_sq[i] = s

    X_norms = [math.sqrt(norms_sq[i]) for i in range(n)]

    X_normalized = [[0.0] * d for _ in range(n)]
    for i in range(n):
        norm_val = X_norms[i]
        for c in range(d):
            X_normalized[i][c] = float(X[i, c]) / norm_val

    D = [[0.0] * n for _ in range(n)]
    cos_sim = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = float('inf')
                cos_sim[i][j] = -float('inf')
            else:
                dot = 0.0
                cos_dot = 0.0
                for c in range(d):
                    val_i = float(X[i, c])
                    val_j = float(X[j, c])
                    dot += val_i * val_j
                    cos_dot += X_normalized[i][c] * X_normalized[j][c]
                D[i][j] = norms_sq[i] + norms_sq[j] - 2.0 * dot
                cos_sim[i][j] = cos_dot

    res = [False] * n
    for i in range(n):
        l2_sorted = sorted(range(n), key=lambda j: D[i][j])[:k]
        cos_sorted = sorted(range(n), key=lambda j: -cos_sim[i][j])[:k]
        res[i] = set(l2_sorted) != set(cos_sorted)

    return np.array(res, dtype=bool)
