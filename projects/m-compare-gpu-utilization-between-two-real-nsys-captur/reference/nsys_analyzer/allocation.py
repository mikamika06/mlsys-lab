ALLOCATION_APIS = {
    "cudaMalloc",
    "cudaFree",
    "cudaMallocAsync",
    "cudaFreeAsync",
    "cudaMallocHost",
    "cudaFreeHost",
    "cudaMallocManaged",
}


def compute_allocation_churn_overhead(cuda_api_report):
    total_time_ns = 0
    alloc_time_ns = 0

    for entry in cuda_api_report:
        duration = entry.get("total_time_ns", 0)
        total_time_ns += duration
        if entry.get("name") in ALLOCATION_APIS:
            alloc_time_ns += duration

    if total_time_ns == 0:
        return 0.0

    return (alloc_time_ns / total_time_ns) * 100.0
