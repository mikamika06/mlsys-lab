import random

random.seed(42)

CAPTURES = [
    {
        "batch_size": 16,
        "capture_window": (0, 10000000),
        "kernel_events": [
            {"start_ns": 500000, "end_ns": 2500000, "stream": 7},
            {"start_ns": 2000000, "end_ns": 3500000, "stream": 14},
            {"start_ns": 4000000, "end_ns": 5500000, "stream": 7},
            {"start_ns": 6000000, "end_ns": 7000000, "stream": 7},
        ],
    },
    {
        "batch_size": 32,
        "capture_window": (0, 10000000),
        "kernel_events": [
            {"start_ns": 100000, "end_ns": 3000000, "stream": 7},
            {"start_ns": 2500000, "end_ns": 6000000, "stream": 14},
            {"start_ns": 6500000, "end_ns": 9500000, "stream": 7},
        ],
    },
    {
        "batch_size": 64,
        "capture_window": (0, 10000000),
        "kernel_events": [
            {"start_ns": 0, "end_ns": 4500000, "stream": 7},
            {"start_ns": 4000000, "end_ns": 8500000, "stream": 14},
            {"start_ns": 8600000, "end_ns": 9900000, "stream": 7},
        ],
    },
]

CUDA_API_REPORT = [
    {"name": "cudaLaunchKernel", "total_time_ns": 45000000, "calls": 1200},
    {"name": "cudaMalloc", "total_time_ns": 8000000, "calls": 45},
    {"name": "cudaFree", "total_time_ns": 2000000, "calls": 45},
    {"name": "cudaMemcpyAsync", "total_time_ns": 15000000, "calls": 300},
    {"name": "cudaStreamSynchronize", "total_time_ns": 30000000, "calls": 150},
]


def compute_gpu_utilization(kernel_events, capture_window):
    start_win, end_win = capture_window
    if end_win <= start_win:
        return 0.0

    intervals = []
    for k in kernel_events:
        ks = max(k["start_ns"], start_win)
        ke = min(k["end_ns"], end_win)
        if ks < ke:
            intervals.append((ks, ke))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda x: x[0])
    merged = []
    curr_start, curr_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))

    active_ns = sum(e - s for s, e in merged)
    total_ns = end_win - start_win
    return (active_ns / total_ns) * 100.0


def compare_batch_utilizations(captures):
    utils = [
        compute_gpu_utilization(cap["kernel_events"], cap["capture_window"])
        for cap in captures
    ]
    min_idx = min(range(len(utils)), key=lambda i: utils[i]) if utils else 0
    return {"utilizations": utils, "argmin_index": min_idx}


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
