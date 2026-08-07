import numpy as np

def expected_accepted_length(acceptance_probs):
    probs = np.asarray(acceptance_probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    cum_prod = np.cumprod(probs)
    return float(np.sum(cum_prod))
