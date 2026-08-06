# Thread Scaling, Oversubscription & NUMA Topology in Inference Workloads

During recent scaling tests of a CPU inference pipeline across both Apple Silicon workstations and multi-socket NUMA servers, performance degradation and severe throughput collapse were observed under specific thread configurations.

On a 10-core Apple Silicon machine (8 Performance cores, 2 Efficiency cores), scaling the worker thread count produced non-monotonic latency behavior. While single-threaded and modest multi-threaded runs scaled as expected, setting the thread pool size to equal or exceed total logical cores caused latency spikes and severe throughput regression.

On a dual-socket NUMA server, running memory-bound GEMM operations across sockets without NUMA-aware allocation resulted in unpredictable performance penalties compared to node-local memory access.

Your task is to implement the thread pool sweep analyzer and NUMA affinity planner to:
1. Identify optimal thread counts and pinpoint the exact oversubscription point on hybrid CPU architectures.
2. Calculate NUMA locality access penalties and memory cost ratios from system hardware topology dumps.
3. Build regression tests to prevent invalid thread sizing and locality miscalculations in production deployment scripts.
