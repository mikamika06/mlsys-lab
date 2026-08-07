def residual_distribution(p: list[float], q: list[float]) -> list[float]:
    """
    The distribution to resample from when a speculative-decoding draft
    token is rejected.

    BUG: this just returns the target distribution p unchanged, instead of
    the normalized residual max(p - q, 0). Fix it.
    """
    raise NotImplementedError('your code here')
