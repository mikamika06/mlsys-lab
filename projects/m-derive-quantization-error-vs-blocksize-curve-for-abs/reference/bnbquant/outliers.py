import numpy as np


def expected_outlier_count(num_rows, num_cols, threshold=6.0):
    from scipy.special import erf
    prob = 2.0 * (1.0 - 0.5 * (1.0 + erf(threshold / np.sqrt(2.0))))
    expected_per_col = num_rows * prob
    total_expected = num_cols * prob
    return float(total_expected)
