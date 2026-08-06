import random


def pathological_variance_input() -> list[float]:
    """A counterexample where the naive one-pass variance formula fails
    catastrophically while Welford's algorithm stays accurate.

    Mean ~1e8 with unit spread: E[X^2] ~ 1e16, right at float64's ~16-digit
    precision limit, so mean(x**2) - mean(x)**2 loses essentially every
    correct digit of the O(1) true variance. Welford never forms that huge
    intermediate, so it stays accurate to machine precision regardless.
    """
    rng = random.Random(0)
    n = 100
    return [1e8 + rng.gauss(0.0, 1.0) for _ in range(n)]
