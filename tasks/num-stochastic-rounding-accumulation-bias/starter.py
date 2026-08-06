import math

def accumulate_rne(start: float, c: float, n_steps: int, q: float) -> float:
    """Repeatedly add `c` into `start`, `n_steps` times, snapping the running
    total to the nearest multiple of `q` (round-half-to-even on ties) after
    every addition. Returns the final accumulator value.
    """
    raise NotImplementedError('your code here')
