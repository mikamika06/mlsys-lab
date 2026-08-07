"""Measure tensor staging costs."""
import torch
import time

def measure_staging_cost(sizes):
    costs = {}
    for size in sizes:
        x = torch.randn(size, size)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            t0 = time.time()
            _ = x.to('mps')
            torch.mps.synchronize()
            t1 = time.time()
            costs[size] = t1 - t0
        else:
            t0 = time.time()
            _ = x.to('cpu')
            t1 = time.time()
            costs[size] = t1 - t0
    return costs
