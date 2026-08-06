import numpy as np

def compute_divergences(set_a: list[list[int]], set_b: list[list[int]]) -> list[int]:
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

def check_regression_gate(divergences: list[int], k: int, max_fail_fraction: float) -> bool:
    if not divergences:
        return True
    fails = sum(1 for d in divergences if 0 <= d < k)
    return (fails / len(divergences)) <= max_fail_fraction

def analyze_near_ties(logits: np.ndarray, divergences: list[int], eps: float = 1e-5) -> float:
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
