import numpy as np
from mlsys import scorers


def _reference_windows(x, width):
    m = len(x) - width + 1
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(m, width),
        strides=(x.strides[0], x.strides[0]),
        writeable=False,
    )


def _in_bounds(view, source):
    if view.size == 0:
        return True
    base = source.__array_interface__["data"][0]
    item = source.dtype.itemsize
    lo = view.__array_interface__["data"][0]
    hi = lo
    if view.size:
        max_offset = 0
        for dim, stride in zip(view.shape, view.strides):
            if dim:
                max_offset += (dim - 1) * stride
        hi = lo + max_offset + item
    return lo >= base and hi <= base + source.nbytes


def grade(sol, fx) -> dict:
    cases = [
        (np.arange(12, dtype=np.int64), 4),
        (np.array([3, 8, -2, 7, 9], dtype=np.int32), 2),
        (np.arange(20, dtype=np.float64), 7),
    ]

    fraction = 1.0
    for x, width in cases:
        ref = _reference_windows(x, width)
        try:
            got = sol.fixed_windows(x, width)
        except Exception:
            fraction = 0.0
            break
        if not isinstance(got, np.ndarray):
            fraction = 0.0
            break
        if not _in_bounds(got, x):
            fraction = 0.0
            break
        if got.shape != ref.shape or got.dtype != ref.dtype:
            fraction = 0.0
            break
        fraction = min(
            fraction,
            scorers.byte_exact_fraction(got, ref),
        )
    return {"byte_exact_fraction": fraction}
