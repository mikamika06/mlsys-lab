import numpy as np


def compute_acceptance_rate(accepted_counts, total_counts):
    if total_counts == 0:
        return 0.0
    return float(accepted_counts) / float(total_counts)
