import numpy as np

def enforce_fraction(fraction, capacity):
    limit = int(capacity * fraction)
    return limit

def check_oom(limit, allocated, requested):
    return (allocated + requested) > limit

def simulate_generation(steps, cadence, alloc_per_step):
    allocated = 0
    clears = 0
    history = []
    for i in range(steps):
        allocated += alloc_per_step
        if i > 0 and i % cadence == 0:
            allocated = max(0, allocated - alloc_per_step * cadence)
            clears += 1
        history.append(allocated)
    return history, clears

def compute_fragmentation(driver_allocs, current_allocs):
    d = np.array(driver_allocs, dtype=float)
    c = np.array(current_allocs, dtype=float)
    ratio = (d - c) / np.maximum(c, 1.0)
    return ratio.tolist()

TEST_CAPACITIES = [1024 * 1024 * 1024 * 8, 1024 * 1024 * 1024 * 16]
