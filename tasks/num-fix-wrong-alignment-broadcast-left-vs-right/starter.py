import numpy as np


def broadcast_add_right(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # TODO: this incorrectly aligns dimensions from the left instead of using
    # NumPy's right-aligned broadcasting semantics.
    a = np.asarray(a)
    b = np.asarray(b)

    if b.ndim < a.ndim:
        shape = b.shape + (1,) * (a.ndim - b.ndim)
        b = b.reshape(shape)

    return a + b
