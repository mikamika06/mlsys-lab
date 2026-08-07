import numpy as np

def compute_removal_order(importance_matrix):
    flat_indices = np.argsort(importance_matrix.ravel())
    rows, cols = np.unravel_index(flat_indices, importance_matrix.shape)
    return list(zip(rows.tolist(), cols.tolist()))
