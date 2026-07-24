import numpy as np


def rotate_and_slice(W1, b1, W2, b2, X_cal, X, k):
    H_cal = X_cal @ W1 + b1
    centered = H_cal - np.mean(H_cal, axis=0, keepdims=True)
    cov = centered.T @ centered / (H_cal.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    Q = eigenvectors[:, order]

    H = X @ W1 + b1
    W2_rot = Q.T @ W2

    return (H @ Q[:, :k]) @ W2_rot[:k, :] + b2
