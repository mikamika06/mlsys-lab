# Diagnostic Report: Unexpected Latency and Throughput Degradation in Microbenchmark Suite

During recent performance characterizations of key CUDA kernels across our training and inference workloads, developers noticed severe pipeline throughput drops. In several microbenchmarks, execution times are significantly higher than predicted by static instruction counts and peak hardware limits.

Hardware counters indicate that warp schedulers on the SMs are encountering high stall rates, but engineers are struggling to connect raw Nsight Compute warp state statistics to concrete algorithmic bottlenecks. Specifically:
1. Profiles from warp state statistics report varying cycles-per-instruction (CPI) penalties across stall reasons, but teams lack an automated method to isolate the top drivers of execution delay.
2. Kernel profiles across memory-heavy, sync-heavy, math-bound, and control-divergent workloads exhibit distinct stall signatures, but developers are incorrectly diagnosing memory latency as execution pipeline congestion.
3. Scheduler statistics report raw issue slot counts (issued, eligible, no eligible warp), but teams are misinterpreting issue slot utilization, blinding them to scheduler starvation vs issue pipeline structural stalls.

To fix this, you must implement a profiling analysis module `warpanalyze` that processes raw warp state and scheduler statistics tables. Your tools must automatically compute top stall reason distributions by CPI contribution, classify kernel execution bottlenecks into root-cause categories using warp state profiles, and calculate issue slot utilization metrics from scheduler counter traces.
