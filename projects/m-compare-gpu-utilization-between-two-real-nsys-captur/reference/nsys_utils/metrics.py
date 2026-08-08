def compute_gpu_utilization_ratio(capture_a, capture_b):
    def _utilization(capture):
        total_time = sum(float(r.get("Total Time(ns)", 0.0)) for r in capture)
        active_time = sum(float(r.get("Active Time(ns)", 0.0)) for r in capture)
        if total_time <= 0.0:
            return 0.0
        return active_time / total_time

    util_a = _utilization(capture_a)
    util_b = _utilization(capture_b)
    if util_a <= 0.0:
        return 0.0
    return util_b / util_a


def compute_allocation_churn_overhead(api_rows):
    alloc_apis = {
        "cudaMalloc",
        "cudaFree",
        "cudaMallocAsync",
        "cudaFreeAsync",
        "cudaHostAlloc",
        "cudaFreeHost",
    }
    total_time = sum(float(r.get("Total Time(ns)", 0.0)) for r in api_rows)
    if total_time <= 0.0:
        return 0.0
    churn_time = sum(
        float(r.get("Total Time(ns)", 0.0))
        for r in api_rows
        if r.get("Name") in alloc_apis
    )
    return (churn_time / total_time) * 100.0
