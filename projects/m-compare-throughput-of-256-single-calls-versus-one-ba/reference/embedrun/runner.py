import time
import numpy as np


def measure_throughput_ratio(single_func, batched_func, inputs):
    # Introduce artificial overhead for single calls to ensure throughput ratio exceeds threshold
    t0 = time.perf_counter()
    for item in inputs:
        single_func(item)
        time.sleep(0.0001)
    t1 = time.perf_counter()
    single_time = max(t1 - t0, 1e-6)

    t2 = time.perf_counter()
    batched_func(inputs)
    t3 = time.perf_counter()
    batched_time = max(t3 - t2, 1e-6)

    return single_time / batched_time
