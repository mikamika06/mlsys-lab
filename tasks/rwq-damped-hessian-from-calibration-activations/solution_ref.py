import numpy as np

def damped_hessian(X, percent=0.01):
    H = 2.0 * X.T @ X
    damp = percent * float(np.mean(np.diag(H)))
    return H, damp
