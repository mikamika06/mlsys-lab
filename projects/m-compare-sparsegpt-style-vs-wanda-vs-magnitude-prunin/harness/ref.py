import numpy as np


def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    in_features = 64
    out_features = 32
    w = rng.standard_normal((out_features, in_features))
    X_match = rng.standard_normal((128, in_features))
    X_mismatch = rng.standard_normal((128, in_features)) * 3.5 + 1.0
    return w, X_match, X_mismatch


def magnitude_prune(w, sparsity):
    flat = np.abs(w.flatten())
    k = int(np.round(sparsity * flat.size))
    thresh = np.sort(flat)[k]
    mask = np.abs(w) > thresh
    w_pruned = w * mask
    return w_pruned, mask


def wanda_prune(w, X, sparsity):
    scores = np.abs(w) * np.sqrt(np.mean(X ** 2, axis=0, keepdims=True))
    flat = scores.flatten()
    k = int(np.round(sparsity * flat.size))
    thresh = np.sort(flat)[k]
    mask = scores > thresh
    w_pruned = w * mask
    return w_pruned, mask


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
    w_pruned = w * mask
    return w_pruned, mask


def evaluate_quality(w_orig, w_pruned, X):
    y_orig = X @ w_orig.T
    y_pruned = X @ w_pruned.T
    diff = np.linalg.norm(y_orig - y_pruned)
    norm = np.linalg.norm(y_orig)
    return float(diff / (norm + 1e-8))


def diagnose_domain_mismatch(w, X_match, X_mismatch, sparsity):
    _, mask_match = wanda_prune(w, X_match, sparsity)
    _, mask_mismatch = wanda_prune(w, X_mismatch, sparsity)
    overlap = np.mean(mask_match == mask_mismatch)
    rel_err_match = evaluate_quality(w, w * mask_match, X_match)
    rel_err_mismatch = evaluate_quality(w, w * mask_mismatch, X_match)
    degradation = float(rel_err_mismatch - rel_err_match)
    return {
        "overlap": float(overlap),
        "degradation": degradation,
        "mismatched_higher_error": bool(rel_err_mismatch > rel_err_match)
    }
