import numpy as np


def accumulate_rne(start: float, c: float, n_steps: int, q: float) -> float:
    """Repeatedly add `c` into `start`, `n_steps` times, snapping the running
    total to the nearest multiple of `q` (round-half-to-even on ties) after
    every addition. Returns the final accumulator value.
    """
    raise NotImplementedError('your code here')


def accumulate_stochastic(start: float, c: float, n_steps: int, q: float,
                           rng: np.random.Generator) -> float:
    """Repeatedly add `c` into `start`, `n_steps` times, using unbiased
    stochastic rounding to the `q`-grid after every addition: round up to
    the next grid point with probability equal to the fractional distance
    to it, using `rng` for the random draw each step. Returns the final
    accumulator value.
    """
    raise NotImplementedError('your code here')
