def kahan_sum(x: list[float]) -> float:
    """Classic Kahan compensated summation (unmodified — keeps its blind spot).

    c = (t - s) - y implicitly assumes |s| >= |x| on every step; when a much
    larger addend arrives while the running sum is still small, the
    compensation recovers nothing and low-order bits of `s` are lost for good.
    """
    raise NotImplementedError('your code here')
