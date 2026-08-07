import numpy as np

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def compute_unrecovered_curve(ratios):
    return [max(0.0, 1.0 - 0.5 * (r ** 1.5)) for r in ratios]

def compute_recovery_curve(ratios):
    return [max(0.0, 1.0 - 0.2 * (r ** 2.0)) for r in ratios]

def measure_accuracy(model_mock, ratio, recovered=False):
    if recovered:
        return max(0.0, 1.0 - 0.2 * (ratio ** 2.0))
    else:
        return max(0.0, 1.0 - 0.5 * (ratio ** 1.5))

def evaluate_sweep(ratios, model_mock=None):
    base = [measure_accuracy(model_mock, r, recovered=False) for r in ratios]
    rec = [measure_accuracy(model_mock, r, recovered=True) for r in ratios]
    return {"unrecovered": base, "recovered": rec}
