import numpy as np

def hadamard_matrix(dim):
    h = np.array([[1.0]], dtype=np.float32)
    n = 1
    while n < dim:
        h = np.block([[h, h], [h, -h]])
        n *= 2
    return h / np.sqrt(dim)

def rotate_activation(x, h):
    return np.matmul(x, h)

def rotate_weight(w, h):
    return np.matmul(h.T, w)
