import time
import numpy as np


def benchmark(fn, warmup=5, reps=50, sync_fn=None):
    for _ in range(warmup):
        fn()
        if sync_fn is not None:
            sync_fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        if sync_fn is not None:
            sync_fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    arr = np.array(times)
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    return {"median": median, "iqr": iqr, "raw": times}
