We are observing unpredictable tail latency spikes and sub-optimal throughput when serving low-level ML inference requests across varying batch sizes on Apple Silicon Macs. Engineers have reported that under certain serving workloads, p99 latency spikes drastically beyond baseline expectations, while throughput plateaus sooner than theoretical compute limits suggest.

Additionally, our current runtime pipeline handles both fixed (static) batch shapes and variable (dynamic) input batch dimensions inconsistently. It remains unclear whether setting higher batch sizes under dynamic shape re-allocations creates overheads that negate batching throughput gains altogether, or if static allocation constraints are restricting peak device utilization.

To solve this, we need a robust measurement and optimization pipeline that computes tail latency statistics (p50, p95, and p99) across batch sizes, derives latency-optimal versus throughput-optimal batch sizes, and accurately evaluates the runtime trade-offs between static allocation and dynamic shape re-binding.

Your task is to implement the benchmark and optimization routines in `latbench/stats.py`, `latbench/tradeoff.py`, and `latbench/shapes.py`. Finally, write regression tests in `tests/test_regression.py` that guard against common mistakes such as ignoring tail percentiles when computing latency-optimal operating points.
