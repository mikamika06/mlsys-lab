"""CUDA allocation churn overhead calculator."""

ALLOC_NAMES = {"cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync"}

def compute_allocation_churn(cuda_api_report):
    """Compute percentage of total CUDA API time spent on memory allocations."""
    total_time_ns = 0
    alloc_time_ns = 0

    records = cuda_api_report.get("records", [])
    for rec in records:
        name = rec["name"]
        time_ns = rec["total_time_ns"]
        total_time_ns += time_ns
        if name in ALLOC_NAMES:
            alloc_time_ns += time_ns

    if total_time_ns == 0:
        return 0.0

    return (float(alloc_time_ns) / float(total_time_ns)) * 100.0
