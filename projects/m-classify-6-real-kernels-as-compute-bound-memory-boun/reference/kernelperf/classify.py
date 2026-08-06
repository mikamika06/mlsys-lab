def classify_kernel(kernel):
    if kernel.get("duration_us", 0.0) < 2.0 and kernel.get("dram_pct", 0.0) < 10.0 and kernel.get("compute_pct", 0.0) < 10.0:
        return "latency-bound"
    if kernel.get("dram_pct", 0.0) >= kernel.get("compute_pct", 0.0):
        return "memory-bound"
    return "compute-bound"


def classify_all(kernels):
    return [classify_kernel(k) for k in kernels]
