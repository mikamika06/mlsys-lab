import time

def benchmark_kernel(kernel_fn, warmup=5, rep=20):
    for _ in range(warmup):
        kernel_fn()
    t0 = time.perf_counter()
    for _ in range(rep):
        kernel_fn()
    t1 = time.perf_counter()
    return (t1 - t0) / rep
