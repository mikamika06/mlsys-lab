import random

def pathological_variance_input() -> list[float]:
    """Construct and return a 1-D float array x (len >= 8, finite, honest
    non-degenerate variance) on which the naive one-pass variance formula
    E[X^2] - E[X]^2 is catastrophically wrong (rel err > 0.5) while
    Welford's algorithm stays accurate (rel err < 1e-8). See task.md.
    """
    raise NotImplementedError('your code here')
