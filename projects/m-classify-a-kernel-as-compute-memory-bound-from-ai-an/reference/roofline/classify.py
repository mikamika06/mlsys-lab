def classify_kernel(ai: float, ridge_point: float) -> str:
    if ai >= ridge_point:
        return "compute-bound"
    return "memory-bound"


def max_achievable_gflops(ai: float, peak_gflops: float, bandwidth_gbps: float) -> float:
    memory_bound_perf = ai * bandwidth_gbps
    return min(peak_gflops, memory_bound_perf)
