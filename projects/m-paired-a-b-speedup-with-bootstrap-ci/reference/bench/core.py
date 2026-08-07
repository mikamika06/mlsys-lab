import numpy as np
import torch

def benchmark_compiled_step(model, inputs, warmup=2, iters=5):
    for _ in range(warmup):
        _ = model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start:
            start.record()
        import time
        t0 = time.perf_counter()
        _ = model(*inputs)
        if end:
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end) * 1e-3
        else:
            t = time.perf_counter() - t0
        times.append(float(t))
    return times

def robust_summary(latencies):
    arr = np.array(latencies, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90))
    }

def paired_bootstrap_ci(a, b, n_boot=500, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    n = len(a_arr)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample_a = np.mean(a_arr[idx])
        sample_b = np.mean(b_arr[idx])
        diffs.append(float(sample_a / sample_b))
    arr_diffs = np.array(diffs)
    return {
        "median": float(np.median(arr_diffs)),
        "lower": float(np.percentile(arr_diffs, 100 * (alpha / 2))),
        "upper": float(np.percentile(arr_diffs, 100 * (1 - alpha / 2)))
    }
