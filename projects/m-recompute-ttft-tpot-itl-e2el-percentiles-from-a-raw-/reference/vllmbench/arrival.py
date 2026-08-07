import numpy as np


def generate_poisson_arrivals(rate, duration, seed=42):
    rng = np.random.default_rng(seed)
    if rate <= 0:
        return []
    intervals = rng.exponential(1.0 / rate, size=int(rate * duration * 2))
    times = np.cumsum(intervals)
    times = times[times <= duration]
    return times.tolist()
