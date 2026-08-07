import numpy as np

def apply_rotation(weights, matrix):
    rotated = np.matmul(weights, matrix)
    outliers_max = float(np.max(np.abs(rotated)))
    return rotated, outliers_max
