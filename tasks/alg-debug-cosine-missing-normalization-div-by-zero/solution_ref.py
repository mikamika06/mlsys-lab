import math
import numpy as np

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return the matrix of cosine similarities between rows of A and B.
    Handles zero‑norm vectors by returning 0.0 for any pair involving a zero vector.
    """
    n_a = A.shape[0]
    n_b = B.shape[0]
    d = A.shape[1]

    norm_a = [0.0] * n_a
    for i in range(n_a):
        s = 0.0
        for k in range(d):
            val = float(A[i, k])
            s += val * val
        norm_a[i] = math.sqrt(s)

    norm_b = [0.0] * n_b
    for j in range(n_b):
        s = 0.0
        for k in range(d):
            val = float(B[j, k])
            s += val * val
        norm_b[j] = math.sqrt(s)

    sim = np.zeros((n_a, n_b), dtype=np.float64)
    for i in range(n_a):
        for j in range(n_b):
            denom = norm_a[i] * norm_b[j]
            if denom != 0.0:
                dot = 0.0
                for k in range(d):
                    dot += float(A[i, k]) * float(B[j, k])
                sim[i, j] = dot / denom

    return sim
