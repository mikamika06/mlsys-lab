import numpy as np

def leading_eigenvector(X, num_iter=1000):
    Xc = X - np.mean(X, axis=0, keepdims=True)
    n, d = Xc.shape
    v = np.ones(d).astype(np.float64)  # deterministic initial vector
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v = Xc.T @ (Xc @ v)  # implicit C @ v
        norm = np.linalg.norm(v)
        if norm == 0:
            break
        v /= norm
    return v
