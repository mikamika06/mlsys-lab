import numpy as np

def _oracle(arr):
    """Ground truth from CPython's built-in memoryview."""
    mv = memoryview(arr)
    return {
        "shape": mv.shape,
        "strides": mv.strides,
        "itemsize": mv.itemsize,
        "ndim": mv.ndim,
        "format": mv.format,
    }

def grade(sol, fx) -> dict:
    cases = [
        # 1-D contiguous
        np.array([1, 2, 3], dtype=np.int64),
        # 2-D C-contiguous
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        # 2-D transposed (non-contiguous)
        np.array([[1, 2], [3, 4]], dtype=np.float32).T,
        # reshaped 1-D -> 2-D
        np.array([1, 2, 3, 4, 5, 6], dtype=np.float64).reshape(2, 3),
        # strided slice (step=2)
        np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)[::2],
        # Fortran-ordered 2-D
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32, order="F"),
        # 3-D C-contiguous
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64),
        # empty 1-D
        np.zeros((0,), dtype=np.float32),
        # degenerate 3-D (single element)
        np.array([[[1]]], dtype=np.uint8),
        # float16
        np.array([1.5, 2.5], dtype=np.float16),
        # non-contiguous 2-D slice
        np.arange(12, dtype=np.float64).reshape(3, 4)[1:, :2],
        # 3-D transposed
        np.arange(24, dtype=np.int32).reshape(2, 3, 4).transpose(2, 0, 1),
    ]

    ok = 1.0
    for arr in cases:
        try:
            got = sol.memoryview_info(arr)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(arr)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
