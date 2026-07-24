import numpy as np
from numpy.lib.stride_tricks import as_strided


def _out_strides(shape, strides, out_shape):
    pad = len(out_shape) - len(shape)
    src_shape = (1,) * pad + tuple(shape)
    src_strides = (0,) * pad + tuple(strides)
    out_strides = []
    for tgt, dim, st in zip(out_shape, src_shape, src_strides):
        if dim == tgt:
            # no expansion on this axis (even when dim == tgt == 1): keep the
            # original stride, matching np.broadcast_arrays' own convention
            out_strides.append(st)
        elif dim == 1:
            out_strides.append(0)   # real expansion: free axis, zero stride
        else:
            raise ValueError(f"shape mismatch: {dim} vs {tgt}")
    return tuple(out_strides)


def broadcast_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-copy broadcast of a and b to their common shape (as_strided views)."""
    a = np.asarray(a)
    b = np.asarray(b)

    n = max(a.ndim, b.ndim)
    pa = (1,) * (n - a.ndim) + a.shape
    pb = (1,) * (n - b.ndim) + b.shape

    out_shape = []
    for da, db in zip(pa, pb):
        if da == db or da == 1 or db == 1:
            out_shape.append(max(da, db))
        else:
            raise ValueError(
                f"shapes {a.shape} and {b.shape} could not be broadcast together"
            )
    out_shape = tuple(out_shape)

    va = as_strided(a, shape=out_shape,
                     strides=_out_strides(a.shape, a.strides, out_shape),
                     writeable=False)
    vb = as_strided(b, shape=out_shape,
                     strides=_out_strides(b.shape, b.strides, out_shape),
                     writeable=False)
    return va, vb
