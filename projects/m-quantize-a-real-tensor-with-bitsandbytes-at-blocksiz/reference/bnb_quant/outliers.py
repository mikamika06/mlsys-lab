import numpy as np


def identify_outliers(tensor, threshold=6.0):
    flat = tensor.astype(np.float32)
    mean = np.mean(flat)
    std = np.std(flat)
    if std == 0.0:
        return np.zeros(tensor.shape[1], dtype=bool)
    z_scores = np.abs((flat - mean) / std)
    column_max_z = np.max(z_scores, axis=0)
    outliers = column_max_z > threshold
    return outliers
