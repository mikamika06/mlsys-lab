import struct


def _f32(val: float) -> float:
    return struct.unpack('f', struct.pack('f', val))[0]


def _pairwise_sum_f32(a: list[float]) -> float:
    """Recursively sum a list of float32 values by splitting it in half, so that no
    single accumulator ever has to absorb more than ~log2(N) additions of
    similarly-scaled partial sums. This bounds rounding-error growth to
    O(log N) instead of the O(N) growth of naive left-to-right accumulation.
    """
    n = len(a)
    if n <= 8:
        acc = 0.0
        for v in a:
            acc = _f32(acc + v)
        return acc
    mid = n // 2
    left = _pairwise_sum_f32(a[:mid])
    right = _pairwise_sum_f32(a[mid:])
    return _f32(left + right)


def fused_dot_reduce(x: list[float], w: list[float]) -> float:
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
    x, w : list of float, same length.

    Returns
    -------
    float
        sum_i x[i] * w[i], computed with float32 arithmetic throughout but
        with a rounding-error-safe reduction order.
    """
    p = [_f32(_f32(xi) * _f32(wi)) for xi, wi in zip(x, w)]
    return float(_pairwise_sum_f32(p))
