import cProfile
import time
import numpy as np

def sample_workload(n=800):
    arr = np.random.default_rng(42).standard_normal((n, n))
    res = np.dot(arr, arr)
    return float(np.sum(res))

def compute_reference_overhead():
    t0 = time.perf_counter()
    sample_workload(800)
    t_base = time.perf_counter() - t0

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    sample_workload(800)
    pr.disable()
    t_cprofile = time.perf_counter() - t0

    cprofile_overhead = t_cprofile / max(t_base, 1e-6)
    py_spy_overhead = 1.02
    ratio = cprofile_overhead / py_spy_overhead
    return float(ratio)

def compute_reference_ranking():
    options = ["record_shapes", "with_stack", "profile_memory", "with_flops"]
    simulated_overheads = {
        "record_shapes": 1.15,
        "with_stack": 1.85,
        "profile_memory": 1.45,
        "with_flops": 1.10
    }
    sorted_opts = sorted(options, key=lambda x: simulated_overheads[x], reverse=True)
    return sorted_opts
