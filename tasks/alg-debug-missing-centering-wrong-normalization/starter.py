import numpy as np

def leading_eigenvector(X, num_iter=1000):
    # TODO: missing mean-centering and per-step renormalisation
    n, d = X.shape
    v = np.random.randn(d)
    for _ in range(num_iter):
        v = X.T @ (X @ v)  # uses raw X, no centering
        # no normalisation inside loop
    return v / np.linalg.norm(v)
