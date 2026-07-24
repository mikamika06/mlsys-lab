import numpy as np


def fold_diag_scales(W, X, s):
    W_fold = W * s[np.newaxis, :]
    X_fold = X / s[:, np.newaxis]
    Y_fold = W_fold @ X_fold
    range_reduction_ratio = np.max(np.abs(X)) / (
        np.max(np.abs(X_fold)) + 1e-12
    )
    return W_fold, X_fold, Y_fold, float(range_reduction_ratio)
