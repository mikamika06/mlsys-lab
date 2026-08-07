import struct

def fused_dot_reduce(x: list[float], w: list[float]) -> float:
    """Emulate a fused multiply-reduce kernel: y = sum(x * w), accumulated
    at the kernel's native float32 width.

    BUG: the fusion pass generated a single sequential float32 accumulator
    that walks the product array in memory order. This is what naive
    kernel-fusion codegen does when it avoids materialising the
    intermediate product array -- but it silently changes the reduction
    order relative to a numerically-safe (e.g. pairwise/tree) reduction,
    so terms much smaller than the current accumulator's ULP get rounded
    away entirely instead of contributing to the sum.

    Parameters
    ----------
    x, w : array_like, 1-D, same length.

    Returns
    -------
    float
        sum_i x[i] * w[i], computed with float32 arithmetic.
    """
    raise NotImplementedError('your code here')
