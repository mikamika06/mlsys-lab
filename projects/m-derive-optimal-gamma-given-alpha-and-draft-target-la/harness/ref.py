import numpy as np


def expected_accepted_tokens(gamma, alpha):
    s = 0.0
    for k in range(gamma + 1):
        s += (alpha ** k)
    return s


def analytical_expected_time(gamma, alpha, c):
    num = 1.0 + c * gamma
    den = expected_accepted_tokens(gamma, alpha)
    return num / den


def compute_optimal_gamma(alpha, c, max_gamma=8):
    best_gamma = 1
    best_time = float("inf")
    for g in range(1, max_gamma + 1):
        t = analytical_expected_time(g, alpha, c)
        if t < best_time:
            best_time = t
            best_gamma = g
    return best_gamma


TEST_CASES = [
    {"alpha": 0.8, "c": 0.2, "max_gamma": 8},
    {"alpha": 0.5, "c": 0.5, "max_gamma": 8},
    {"alpha": 0.9, "c": 0.1, "max_gamma": 8},
    {"alpha": 0.3, "c": 0.8, "max_gamma": 8},
    {"alpha": 0.7, "c": 0.3, "max_gamma": 8},
]
