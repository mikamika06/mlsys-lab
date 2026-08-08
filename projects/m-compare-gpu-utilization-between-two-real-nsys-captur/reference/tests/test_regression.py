import sys

sys.path.insert(0, ".")
from nsys_analyzer.allocation import compute_allocation_churn_overhead
from nsys_analyzer.utilization import compute_gpu_utilization


def test_utilization_merged_intervals():
    events = [
        {"start_ns": 100, "end_ns": 300},
        {"start_ns": 200, "end_ns": 400},
        {"start_ns": 500, "end_ns": 600},
    ]
    window = (100, 600)
    util = compute_gpu_utilization(events, window)
    assert abs(util - 80.0) < 1e-5


def test_allocation_churn_overhead_calculation():
    report = [
        {"name": "cudaLaunchKernel", "total_time_ns": 7000},
        {"name": "cudaMalloc", "total_time_ns": 2000},
        {"name": "cudaFree", "total_time_ns": 1000},
    ]
    churn = compute_allocation_churn_overhead(report)
    assert abs(churn - 30.0) < 1e-5
