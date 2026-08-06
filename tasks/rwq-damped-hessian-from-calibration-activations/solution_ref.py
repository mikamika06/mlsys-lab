import numpy as np

def damped_hessian(X, percent=0.01):
    n_rows, n_cols = X.shape
    H = np.zeros((n_cols, n_cols), dtype=X.dtype)
    for i in range(n_cols):
        for j in range(n_cols):
            acc = 0.0
            for k in range(n_rows):
                acc += X[k, i] * X[k, j]
            H[i, j] = 2.0 * acc
    
    diag_sum = 0.0
    for i in range(n_cols):
        diag_sum += H[i, i]
    mean_diag = diag_sum / float(n_cols)
    damp = percent * float(mean_diag)
    
    return H, damp
