import numpy as np
from speculative.metrics import compute_acceptance_rate


def select_best_draft(candidates):
    rates = [compute_acceptance_rate(c) for c in candidates]
    if not rates:
        return -1
    return int(np.argmax(rates))
