import numpy as np


def apply_hadamard_transform(x):
    dim = x.shape[-1]
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32) / np.sqrt(2.0)
    curr = h
    while curr.shape[0] < dim:
        curr = np.kron(curr, h)
    curr = curr[:dim, :dim]
    orig_shape = x.shape
    x_2d = x.reshape(-1, dim)
    transformed = np.matmul(x_2d, curr)
    return transformed.reshape(orig_shape)
