# Investigating Latency Anomalies in Profile Analysis Tooling

## Symptom
During recent load testing of the profile aggregation pipeline, the performance analytics team noticed inconsistent summary reports when analyzing Nsight Systems export databases across different workload batch sizes. Specifically, when comparing CUDA runtime allocation behaviors and GPU timeline utilization, the automated diagnostic script generated conflicting GPU efficiency metrics. In several cases, high-throughput runs were flagged with low active duration ratios, while memory-heavy iterations failed to highlight significant API memory churn.

Upon inspecting the pipeline, we found that raw profiling exports (`nsys stats` CSV/SQLite reports) are currently parsed with crude regex passes. This fails to handle multi-stream overlap correctly and miscalculates memory overhead relative to total runtime.

## Mission
You need to overhaul the profile parsing and analytics package under `profile_analyzer/` to compute accurate, robust metrics directly from raw `cuda_api_sum` and `gpu_kern_sum` report records.

1. **GPU Utilization Comparison**: Implement `compare_gpu_utilization(report_a, report_b)` to calculate the total active GPU time (merging overlapping kernel execution intervals per device) normalized by total trace duration, returning the relative utilization and identifying the less efficient capture using the `argmin_index` criterion.
2. **Allocation Churn Analysis**: Implement `compute_allocation_churn(cuda_api_report)` to compute allocation-churn overhead percentage: the time spent in memory management calls (`cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync`) as a percentage of total CUDA API duration across all threads.
3. **Regression Safety Net**: Create a suite of unit tests in `tests/test_regression.py` that verifies these profile metrics and catches naive stream-merging bugs where overlapping GPU kernel intervals on distinct streams are incorrectly merged.
