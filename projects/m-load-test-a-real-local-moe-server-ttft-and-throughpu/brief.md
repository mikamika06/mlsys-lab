# Load-test a real local MoE server: TTFT and throughput under concurrency

Our local Mixture-of-Experts (MoE) serving stack has been experiencing erratic tail latencies and unexpected throughput drops during peak concurrent traffic. Engineers suspect that under high load, Time to First Token (TTFT) degrades dramatically due to prompt processing bottlenecks and unoptimized token generation dispatch.

Your goal is to build an in-memory load test framework and client simulator that benchmarks a local MoE server endpoint under varying concurrency levels. You will measure TTFT and total request throughput, profile latency percentiles, and construct a regression test suite that catches mock load test generators reporting invalid TTFT calculations or artificially zeroed latency metrics.

### Deliverables
- `moeload/benchmark.py`: Implement the concurrent workload driver, tracking per-request metrics (TTFT, inter-token latency, total execution time, token count).
- `moeload/metrics.py`: Calculate throughput (tokens/sec), TTFT percentiles (P50, P90, P99), and analyze latency degradation ratios across low vs. high concurrency runs.
- `moeload/server.py`: Provide a local simulated MoE server model that correctly handles queueing delay and token generation latency under concurrent worker pressure.
- `tests/test_regression.py`: Write regression tests ensuring TTFT is never lower than initial prefill time and that latency distributions properly flag concurrency degradation.
