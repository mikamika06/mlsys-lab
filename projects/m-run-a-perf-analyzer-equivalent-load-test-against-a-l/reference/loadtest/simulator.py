import numpy as np

def generate_schedule(concurrency, num_requests, seed=42):
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=1.0 / concurrency, size=num_requests)
    return np.cumsum(inter_arrivals)
