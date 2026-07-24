import numpy as np


def accumulate_rne(start: float, c: float, n_steps: int, q: float) -> float:
    a = float(start)
    for _ in range(n_steps):
        a = q * round((a + c) / q)
    return a


def accumulate_stochastic(start: float, c: float, n_steps: int, q: float,
                           rng: np.random.Generator) -> float:
    a = float(start)
    for _ in range(n_steps):
        v = a + c
        lo = np.floor(v / q) * q
        t = (v - lo) / q
        if rng.random() < t:
            a = lo + q
        else:
            a = lo
    return a
