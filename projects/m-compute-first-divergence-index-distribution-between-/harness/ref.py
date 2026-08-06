import numpy as np

def get_fixtures_m1():
    np.random.seed(42)
    set_a = [np.random.randint(0, 100, size=50).tolist() for _ in range(10)]
    set_b = [x[:] for x in set_a]
    set_b[1][10] = 999
    set_b[3][5] = 999
    set_b[5][40] = 999
    set_b[7] = set_b[7][:30]
    set_a[8] = set_a[8][:25]
    return set_a, set_b

def compute_divergences(set_a, set_b):
    divs = []
    for a, b in zip(set_a, set_b):
        diverged = False
        for i, (ta, tb) in enumerate(zip(a, b)):
            if ta != tb:
                divs.append(i)
                diverged = True
                break
        if not diverged:
            if len(a) != len(b):
                divs.append(min(len(a), len(b)))
            else:
                divs.append(-1)
    return divs

def get_fixtures_m2_gate():
    return [5, -1, 10, -1, 2, 50, -1], 10, 0.2

def check_regression_gate(divergences, k, max_fail_fraction):
    if not divergences:
        return True
    fails = sum(1 for d in divergences if 0 <= d < k)
    return (fails / len(divergences)) <= max_fail_fraction

def get_fixtures_m2_ties():
    np.random.seed(42)
    logits = np.random.rand(5, 20, 100)
    logits[1, 5, 10] = 0.99999
    logits[1, 5, 20] = 0.99998
    divs = [-1, 5, -1, 10, 15]
    return logits, divs, 1e-4

def analyze_near_ties(logits, divergences, eps=1e-5):
    if not divergences:
        return 0.0
    ties = 0
    valid = 0
    for i, div in enumerate(divergences):
        if 0 <= div < logits.shape[1]:
            valid += 1
            step_logits = logits[i, div]
            top2 = np.partition(step_logits, -2)[-2:]
            diff = abs(top2[1] - top2[0])
            if diff <= eps:
                ties += 1
    return ties / valid if valid > 0 else 0.0
