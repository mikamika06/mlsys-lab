import numpy as np


def rank_by_condition(matrices):
    conditions = [
        float(np.linalg.cond(np.asarray(matrix, dtype=np.float64)))
        for matrix in matrices
    ]
    return sorted(range(len(matrices)), key=lambda i: conditions[i])
