import numpy as np


def expected_speedup(alphas):
    current_surv = 1.0
    expected_accepted = 0.0
    for alpha in alphas:
        expected_accepted += current_surv
        current_surv *= alpha
    return expected_accepted
