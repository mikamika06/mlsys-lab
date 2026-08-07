import numpy as np


def compute_hessian(X):
    return 2.0 * (X.T @ X) / X.shape[0]
