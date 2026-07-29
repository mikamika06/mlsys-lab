import statistics
import time


def measure(fn, warmup=3, reps=9, timer=time.perf_counter, sync=None):
    for _ in range(max(0, warmup)):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(max(1, reps)):
        t0 = timer()
        fn()
        if sync:
            sync()
        samples.append(timer() - t0)
    samples.sort()
    n = len(samples)
    q1 = samples[min(n - 1, int(round(0.25 * (n - 1))))]
    q3 = samples[min(n - 1, int(round(0.75 * (n - 1))))]
    return {"median": statistics.median(samples), "iqr": q3 - q1,
            "min": samples[0], "reps": n, "samples": samples}
