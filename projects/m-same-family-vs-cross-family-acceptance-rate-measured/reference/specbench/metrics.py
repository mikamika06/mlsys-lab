import numpy as np


def family_gap_ratio(same_family_rates, cross_family_rates):
    mean_same = float(np.mean(same_family_rates)) if same_family_rates else 0.0
    mean_cross = float(np.mean(cross_family_rates)) if cross_family_rates else 0.0
    if mean_cross == 0.0:
        return float("inf") if mean_same > 0 else 1.0
    return mean_same / mean_cross
