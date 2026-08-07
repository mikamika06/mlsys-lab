import numpy as np


def evaluate_accuracy(weights, X, y):
    logits = np.dot(X, weights)
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == y))


def steps_to_90_recovery(accuracies, baseline_acc, pruned_acc):
    threshold = pruned_acc + 0.9 * (baseline_acc - pruned_acc)
    for idx, acc in enumerate(accuracies):
        if acc >= threshold:
            return int(idx)
    return int(len(accuracies))
