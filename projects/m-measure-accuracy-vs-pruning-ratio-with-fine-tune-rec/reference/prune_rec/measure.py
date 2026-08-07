import numpy as np


def measure_accuracy_curve(weights, data, ratios):
    X, y = data
    results = []
    w_orig = weights.copy()
    for r in ratios:
        if r > 0.0:
            flat = np.abs(w_orig).flatten()
            thresh = np.percentile(flat, r * 100.0)
            mask = np.abs(w_orig) >= thresh
            w_pruned = w_orig * mask
        else:
            w_pruned = w_orig.copy()
        preds = np.dot(X, w_pruned)
        acc = np.mean(np.argmax(preds, axis=1) == y) if preds.ndim > 1 else np.mean((preds > 0) == y)
        results.append((r, float(acc)))
    return results
