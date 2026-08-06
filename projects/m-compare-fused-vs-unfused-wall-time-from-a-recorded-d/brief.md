# Wall-time and Throughput Analysis from Triton Benchmark Logs

Our profiling pipeline collects automated `do_bench` execution logs across pointwise elementwise operations to compare non-fused operation chains against single-kernel fused Triton implementations. Recently, downstream performance tracking tools reported inconsistent latency and throughput calculations when analyzing benchmark output traces, causing automated regressions to trigger false alarms.

In this exercise, you will implement a trace parser and benchmark analysis utility that converts raw Triton `do_bench` execution records into clear wall-time, latency speedup, and memory bandwidth throughput metrics.

You need to implement the analysis pipeline in `bench_analysis/parser.py`, `bench_analysis/metrics.py`, and `bench_analysis/report.py`.

1. **Parse execution traces**: Process raw execution dictionaries containing recorded `do_bench` timing samples (in milliseconds) alongside tensor shapes and element data types to extract mean execution wall-times and standard deviations.
2. **Compute throughput and latency speedups**: Calculate effective memory bandwidth in GB/s for both fused and unfused kernel chains based on exact byte transfers, and determine the relative latency speedup ratio ($T_{unfused} / T_{fused}$).
3. **Regression test suite**: Write a test suite in `tests/test_regression.py` that verifies throughput and speedup invariants across various tensor shapes and validates that incorrect throughput logic (such as ignoring element byte sizes or improperly scaling unfused overhead) is caught immediately.
