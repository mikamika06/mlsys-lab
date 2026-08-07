Our speculative decoding production cluster relies on a 2-stage real-model cascade to reduce inference latency compared to standard single-stage autoregressive generation. However, recent latency telemetry shows significant variance across different workload profiles and draft step configurations, leading to unvalidated deployments where cascading occasionally slows down generation instead of speeding it up.

We currently lack a standardized benchmarking module to measure execution time, token acceptance rates, and effective speedup ratios across multi-stage draft cascades versus single-stage baselines. Without a deterministic profiling suite, engineers cannot accurately verify if a stage configuration yields a real latency improvement or if draft overhead dominates execution.

Your task is to build the measurement and benchmarking harness for the 2-stage real-model cascade:
1. Implement execution measurement functions in `cascade/measure.py` that quantify per-stage and overall latency for a 2-stage draft-verify cascade as well as a single-stage baseline.
2. Implement benchmark execution and ratio calculation in `cascade/benchmark.py` to evaluate latency ratios against baseline models under realistic parameter sweeps.
3. Write comprehensive regression tests in `tests/test_regression.py` that enforce latency invariants and catch structural errors like broken step-scaling or invalid speedup calculations.
