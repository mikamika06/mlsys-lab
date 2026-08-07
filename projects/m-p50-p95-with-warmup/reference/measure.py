import time
import numpy as np

def benchmark(fn, warmup, iters, reject_outliers=False, clock=time.perf_counter):
    warmup_times = []
    for _ in range(warmup):
        t0 = clock()
        fn()
        t1 = clock()
        warmup_times.append(t1 - t0)

    measure_times = []
    for _ in range(iters):
        t0 = clock()
        fn()
        t1 = clock()
        measure_times.append(t1 - t0)

    if reject_outliers and len(measure_times) > 0:
        q1 = np.percentile(measure_times, 25)
        q3 = np.percentile(measure_times, 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        measure_times = [t for t in measure_times if t <= upper]

    if not measure_times:
        return {"p50": 0.0, "p95": 0.0, "cold_start_ratio": 0.0}

    p50 = np.percentile(measure_times, 50)
    p95 = np.percentile(measure_times, 95)

    if warmup > 0 and p50 > 0:
        cold = warmup_times[0] / p50
    else:
        cold = 0.0

    return {
        "p50": float(p50),
        "p95": float(p95),
        "cold_start_ratio": float(cold)
    }
