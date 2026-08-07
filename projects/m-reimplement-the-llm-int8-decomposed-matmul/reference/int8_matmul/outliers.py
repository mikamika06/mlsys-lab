import numpy as np


def compute_outlier_curve(tensor, thresholds):
    abs_t = np.abs(tensor)
    fractions = []
    for th in thresholds:
        mask = abs_t > th
        fraction = np.mean(mask)
        fractions.append(float(fraction))
    return fractions
