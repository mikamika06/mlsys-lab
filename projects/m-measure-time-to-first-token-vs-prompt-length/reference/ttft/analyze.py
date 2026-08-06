import numpy as np


def compute_relative_error(actual, predicted):
    act = np.array(actual, dtype=float)
    pred = np.array(predicted, dtype=float)
    return float(np.mean(np.abs(act - pred) / np.maximum(act, 1e-8)))


def evaluate_threshold(errors, max_allowed):
    return all(e <= max_allowed for e in errors)
