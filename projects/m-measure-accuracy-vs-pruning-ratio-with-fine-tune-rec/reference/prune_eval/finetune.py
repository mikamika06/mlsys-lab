from prune_eval.simulate import measure_accuracy

def compute_recovery_curve(ratios):
    return [max(0.0, 1.0 - 0.2 * (r ** 2.0)) for r in ratios]

def evaluate_sweep(ratios, model_mock=None):
    base = [measure_accuracy(model_mock, r, recovered=False) for r in ratios]
    rec = [measure_accuracy(model_mock, r, recovered=True) for r in ratios]
    return {"unrecovered": base, "recovered": rec}
