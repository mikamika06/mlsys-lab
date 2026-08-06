import numpy as np


def compute_break_even(dense_bytes, element_size, index_size):
    num_elements = dense_bytes // element_size
    for s in np.linspace(0.0, 1.0, 1001):
        nz = int(np.round((1.0 - s) * num_elements))
        sparse_bytes = nz * element_size + np.ceil(num_elements / 8) + nz * index_size
        if sparse_bytes < dense_bytes:
            return float(s)
    return 1.0
