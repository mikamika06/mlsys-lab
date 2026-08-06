import torch
import tracemalloc

def measure_peak_memory_diff(x):
    tracemalloc.start()
    a = x.clone()
    b = a * 2.0
    c = b.sum(dim=-1, keepdim=True)
    d = c.expand_as(a)
    _ = d + a
    _, peak_naive = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = x * 3.0
    _, peak_fused = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if peak_fused == 0:
        peak_fused = 1
    return float(peak_naive) / float(peak_fused)
