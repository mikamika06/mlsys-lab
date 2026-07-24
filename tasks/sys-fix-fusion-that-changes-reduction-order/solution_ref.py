import numpy as np


def _pairwise_sum_f32(a: np.ndarray) -> np.float32:
    """Recursively sum a float32 array by splitting it in half, so that no
    single accumulator ever has to absorb more than ~log2(N) additions of
    similarly-scaled partial sums. This bounds rounding-error growth to
    O(log N) instead of the O(N) growth of naive left-to-right accumulation.
    """
    n = a.shape[0]
    if n <= 8:
        acc = np.float32(0.0)
        for v in a:
            acc = np.float32(acc + v)
        return acc
    mid = n // 2
    left = _pairwise_sum_f32(a[:mid])
    right = _pairwise_sum_f32(a[mid:])
    return np.float32(left + right)


def fused_dot_reduce(x: np.ndarray, w: np.ndarray) -> float:
    """Emulate a fused multiply-reduce kernel: y = sum(x * w), accumulated
    at the kernel's native float32 width.

    A compiler that fuses the elementwise multiply with the reduction (to
    avoid materialising the product array) must still pick a numerically
    safe order for the reduction. This implementation combines partial
    products with a pairwise (tree) reduction rather than a single
    sequential float32 accumulator, so that a handful of large-magnitude
    terms mixed with many small-magnitude terms do not silently swallow
    the small terms' contribution.

    Parameters
    ----------
    x, w : array_like, 1-D, same length.

    Returns
    -------
    float
        sum_i x[i] * w[i], computed with float32 arithmetic throughout but
        with a rounding-error-safe reduction order.
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    p = (x * w).astype(np.float32)
    return float(_pairwise_sum_f32(p))
