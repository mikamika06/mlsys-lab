import numpy as np


def flat_gather(buf: np.ndarray, shape: tuple, strides: tuple, offset: int) -> np.ndarray:
    """Materialize a (shape, strides, offset) view of `buf` by hand.

    Parameters
    ----------
    buf : ndarray, shape (N,)
        Flat float64 backing buffer.
    shape : tuple[int, ...]
        Logical shape of the output.
    strides : tuple[int, ...]
        Byte strides, one per axis of `shape` (may be zero or negative).
    offset : int
        Starting element offset into `buf`.

    Returns
    -------
    ndarray, shape `shape`
        out[i] = buf.ravel()[(offset*itemsize + sum(i[d]*strides[d])) // itemsize]
    """
    buf = np.asarray(buf, dtype=np.float64).ravel()
    itemsize = buf.itemsize
    shape = tuple(int(x) for x in shape)
    strides = tuple(int(x) for x in strides)

    out = np.empty(shape, dtype=np.float64)
    base = offset * itemsize
    for idx in np.ndindex(*shape):
        byte_pos = base + sum(i * st for i, st in zip(idx, strides))
        out[idx] = buf[byte_pos // itemsize]
    return out
