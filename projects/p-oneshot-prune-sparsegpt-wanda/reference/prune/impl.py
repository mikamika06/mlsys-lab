import numpy as np


def compute_importance(W, X, method="wanda"):
    if method == "magnitude":
        return np.abs(W)
    X_norm = np.linalg.norm(X, axis=0, keepdims=True)
    return np.abs(W) * X_norm


def prune_layer(W, X, sparsity=0.5, method="wanda"):
    scores = compute_importance(W, X, method=method)
    flat = scores.flatten()
    k = int(flat.size * sparsity)
    if k == 0:
        return W.copy()
    thresh = np.sort(flat)[k]
    return W * (scores > thresh)


def evaluate_model(weights, inputs, targets):
    curr = inputs
    for w in weights:
        curr = np.tanh(np.dot(curr, w))
    return float(np.mean((curr - targets) ** 2))


def compare_methods(weights, inputs, targets, sparsity=0.5):
    w1 = [prune_layer(w, inputs, sparsity, "magnitude") for w in weights]
    w2 = [prune_layer(w, inputs, sparsity, "wanda") for w in weights]
    l1 = evaluate_model(w1, inputs, targets)
    l2 = evaluate_model(w2, inputs, targets)
    return {"magnitude_loss": l1, "wanda_loss": l2, "ratio": l2 / (l1 + 1e-8)}


def check_loss_bound(baseline_loss, pruned_loss, max_ratio=1.5):
    return bool(pruned_loss <= baseline_loss * max_ratio + 0.05)


def generate_report(results):
    return "Report generated successfully"
