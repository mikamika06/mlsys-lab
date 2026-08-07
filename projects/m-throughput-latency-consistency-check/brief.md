We are benchmarking a new TensorRT model using a custom Python harness, but the numbers look deeply suspicious when compared to `trtexec` logs.

Our harness reports a blazing fast latency of 0.05ms, yet the throughput peaks at only 45 requests per second even when pushing a single stream. Meanwhile, `trtexec` running the exact same ONNX model reports 22ms latency and 44 requests per second.

The team refuses to merge the new model until we can mathematically prove our benchmarking script is emitting physically possible numbers.

We have updated the harness to dump raw execution traces instead of computing metrics itself. Each trace event now contains four timestamps: `host_start`, `host_end`, `device_start`, and `device_end`.

You need to write `perf/analyzer.py` to process these traces.

1. `compute_metrics(events)`: Calculate the throughput across the entire trace, as well as the mean latency from three perspectives: host-only, device-only, and true end-to-end.
2. `consistency_error(throughput, mean_latency, concurrency)`: Calculate how far the metrics deviate from theoretical consistency using Little's Law.
3. `validate_trace(events, concurrency, tol=0.05)`: Compute metrics from a trace and raise a `ValueError` if the end-to-end metrics violate basic queuing theory bounds by more than `tol`.

Add `tests/test_regression.py` to ensure your validation strictly accepts good traces and correctly rejects impossible metrics.
