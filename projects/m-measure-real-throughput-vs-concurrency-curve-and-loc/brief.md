# Real Throughput vs Concurrency and Knee Detection

Our serving system experienced severe latency spikes during peak traffic hours last night. Operations reports that total request throughput flatlined while request queue depths ballooned, causing client timeouts across multiple downstream services. Preliminary inspection suggests the engine scheduler was configured with a static concurrency limit derived from synthetic single-request benchmarks rather than realistic concurrent loads.

When serving requests under continuous batching, increasing concurrency improves hardware utilization up to a point. Beyond this optimal operating point—the throughput-concurrency "knee"—adding more concurrent sequences introduces severe contention for KV cache blocks, increases scheduling overhead, and leads to preemption cascades or memory thrashing without providing additional token generation throughput.

You are tasked with building an empirical benchmarking and load analysis module to measure request-level token generation throughput across varying concurrency levels, accurately locate the optimal knee capacity point, and establish regression safety tests.

## Your Goal
1. Implement `measure_concurrency_curve` in `throughput/bench.py` to simulate request execution under variable concurrency loads and record key performance metrics (total generation throughput in tokens/sec, average request latency, and cache block utilization).
2. Implement `locate_knee` in `throughput/analyzer.py` to automatically detect the concurrency knee using the maximum perpendicular distance method relative to the linear saturation baseline.
3. Write a suite of regression tests in `tests/test_regression.py` that verifies knee detection stability and ensures throughput does not degrade under optimal concurrency bounds.
