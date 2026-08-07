import time
import math
import numpy as np
import torch


def benchmark_step(fn, is_cuda=False, warmup=10, reps=100):
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    times_arr = np.array(times)
    median = float(np.median(times_arr))
    q75, q25 = np.percentile(times_arr, [75, 25])
    iqr = float(q75 - q25)
    return {"median": median, "iqr": iqr, "times": times}


def compute_required_reps(times_sample, tolerance=0.05, confidence_z=1.96):
    times = np.array(times_sample)
    mean = float(np.mean(times))
    if mean == 0:
        return 0
    std = float(np.std(times, ddof=1))
    margin = tolerance * mean
    if margin == 0:
        return 0
    n = (confidence_z * std / margin) ** 2
    return math.ceil(n)
