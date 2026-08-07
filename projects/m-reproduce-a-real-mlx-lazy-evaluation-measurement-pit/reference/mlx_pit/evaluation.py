import time
import numpy as np


def measure_lazy_evaluation(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    if hasattr(res, "__array__") or isinstance(res, np.ndarray):
        _ = np.sum(res)
    t2 = time.perf_counter()
    return {
        "lazy_duration": t1 - t0,
        "materialized_duration": t2 - t1,
        "result": res
    }
