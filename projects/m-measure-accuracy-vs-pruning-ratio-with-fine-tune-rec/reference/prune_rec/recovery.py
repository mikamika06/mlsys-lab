import numpy as np


def simulate_finetune_recovery(weights, data, ratio, steps):
    X, y = data
    w = weights.copy()
    if ratio > 0.0:
        flat = np.abs(w).flatten()
        thresh = np.percentile(flat, ratio * 100.0)
        mask = np.abs(w) >= thresh
        w = w * mask

    initial_preds = np.dot(X, w)
    initial_acc = np.mean(np.argmax(initial_preds, axis=1) == y) if initial_preds.ndim > 1 else np.mean((initial_preds > 0) == y)

    lr = 0.01
    for _ in range(steps):
        preds = np.dot(X, w)
        if preds.ndim > 1:
            exp_p = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            probs = exp_p / np.sum(exp_p, axis=1, keepdims=True)
            grad = np.dot(X.T, (probs - np.eye(probs.shape[1])[y])) / X.shape[0]
        else:
            probs = 1 / (1 + np.exp(-preds))
            grad = np.dot(X.T, (probs - y)) / X.shape[0]
        w -= lr * grad
        if ratio > 0.0:
            w = w * mask

    final_preds = np.dot(X, w)
    final_acc = np.mean(np.argmax(final_preds, axis=1) == y) if final_preds.ndim > 1 else np.mean((final_preds > 0) == y)
    return float(initial_acc), float(final_acc)
