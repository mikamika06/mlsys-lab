import numpy as np


def compute_accuracy_delta(baseline_acc, pruned_acc, current_acc):
    total_drop = baseline_acc - pruned_acc
    if total_drop <= 0:
        return 1.0
    recovered = current_acc - pruned_acc
    return float(max(0.0, recovered / total_drop))


def steps_to_recovery(steps_history, target_ratio=0.9):
    for i, acc in enumerate(steps_history):
        if acc >= target_ratio:
            return i + 1
    return len(steps_history)
