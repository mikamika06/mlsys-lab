import numpy as np


def magnitude_prune(w, sparsity):
    flat = np.abs(w.flatten())
    k = int(np.round(sparsity * flat.size))
    thresh = np.sort(flat)[k]
    mask = np.abs(w) > thresh
    return w * mask, mask


def wanda_prune(w, X, sparsity):
    scores = np.abs(w) * np.sqrt(np.mean(X ** 2, axis=0, keepdims=True))
    flat = scores.flatten()
    k = int(np.round(sparsity * flat.size))
    thresh = np.sort(flat)[k]
    mask = scores > thresh
    return w * mask, mask


def sparsegpt_prune(w, X, sparsity):
    XTX = X.T @ X + 1e-4 * np.eye(X.shape[1])
    try:
        H_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        H_inv = np.eye(X.shape[1])
    scores = np.abs(w) / np.sqrt(np.diag(H_inv)).reshape(1, -1)
    flat = scores.flatten()
    k = int(np.round(sparsity * flat.size))
    thresh = np.sort(flat)[k]
    mask = scores > thresh
    return w * mask, mask


def evaluate_quality(w_orig, w_pruned, X):
    y_orig = X @ w_orig.T
    y_pruned = X @ w_pruned.T
    diff = np.linalg.norm(y_orig - y_pruned)
    norm = np.linalg.norm(y_orig)
    return float(diff / (norm + 1e-8))


def compare_methods(w, X, sparsity):
    w_m, _ = magnitude_prune(w, sparsity)
    w_w, _ = wanda_prune(w, X, sparsity)
    w_s, _ = sparsegpt_prune(w, X, sparsity)
    return {
        "magnitude": evaluate_quality(w, w_m, X),
        "wanda": evaluate_quality(w, w_w, X),
        "sparsegpt": evaluate_quality(w, w_s, X)
    }
