# GPU Utilization and Allocation Overhead Analysis

We are observing performance regressions and unexpected tail latencies in our ML inference and training pipelines. While profiling our workloads using Nsight Systems (nsys), engineers noticed that changing batch sizes yields non-linear GPU compute utilization scaling, and frequent memory allocations appear to create execution stalls on the CUDA API thread.

To fix these regressions, we need an automated analysis library that parses `nsys` profile report data (specifically GPU kernel execution events and `cuda_api_sum` runtime trace summaries) to compute exact GPU compute utilization ratios and memory allocation overhead metrics.

Your task is to implement the profiling analyzer across three milestones:
1. Parse raw nsys timeline traces to compute the exact active GPU compute utilization percentage for two different batch size runs and evaluate which configuration maximizes GPU compute occupancy.
2. Calculate the allocation-churn overhead percentage from `cuda_api_sum` reports by analyzing the total time spent in memory allocation and deallocation calls (`cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync`) relative to total CUDA API runtime.
3. Write a regression suite in `tests/test_regression.py` that validates utilization and allocation overhead invariants across profile trace reports, ensuring edge cases like overlapping GPU streams or low-churn execution paths are correctly verified.
