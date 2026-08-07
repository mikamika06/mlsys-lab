import time
import numpy as np

def benchmark(fn, warmup, iters, reject_outliers=False, clock=time.perf_counter):
    raise NotImplementedError("benchmark")
