import time
import numpy as np

def benchmark(fn, warmup_iters, measure_iters, percentiles):
    for _ in range(warmup_iters):
        fn()
    times = []
    for _ in range(measure_iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    if not times:
        return {p: 0.0 for p in percentiles}
    res = np.atleast_1d(np.percentile(times, percentiles))
    return {p: float(val) for p, val in zip(percentiles, res)}

def find_stable_iters(fn, target_rel_err, start_iters=10, max_iters=10000):
    iters = start_iters
    while iters < max_iters:
        a = benchmark(fn, 10, iters, [90])[90]
        b = benchmark(fn, 0, iters, [90])[90]
        m = max(a, b)
        rel_err = abs(a - b) / m if m > 0 else 0.0
        if rel_err <= target_rel_err:
            return iters
        iters *= 2
    return max_iters
