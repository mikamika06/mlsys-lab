import numpy as np


def compute_expected(gamma, p):
    g = int(gamma)
    prob = float(p)
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        return float(g)
    return float(np.sum([prob ** k for k in range(1, g + 1)]))


def compute_acceptance(traces):
    if not traces:
        return 0.0
    total_drafted = sum(d for d, _ in traces)
    total_accepted = sum(a for _, a in traces)
    if total_drafted == 0:
        return 0.0
    return float(total_accepted) / float(total_drafted)


def compute_optimal(max_gamma, p, cost_ratio):
    best_g = 1
    best_tp = -1.0
    for g in range(1, int(max_gamma) + 1):
        exp_acc = compute_expected(g, p)
        tp = (1.0 + exp_acc) / (1.0 + float(cost_ratio))
        if tp > best_tp:
            best_tp = tp
            best_g = g
    return best_g


TEST_CASES = [
    {"gamma": 3, "p": 0.5},
    {"gamma": 5, "p": 0.8},
    {"gamma": 4, "p": 0.2},
]
