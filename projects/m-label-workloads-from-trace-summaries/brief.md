# Launch-bound vs compute-bound diagnosis in PyTorch traces

During a production throughput review of our low-latency PyTorch inference pipeline, engineers noticed that overall GPU utilization drops significantly on certain sequence lengths, yet standard roofline estimates suggest the workload should be compute-bound. Trace analysis shows erratic inter-kernel gaps and unpredictable memory throughput degradation across different execution stages.

You need to implement a trace summary analysis toolkit in `diag/` that accurately characterizes PyTorch kernel workload distributions and root-causes execution bottlenecks from raw event logs.

Your task is to build three components:
1. **Workload Classification**: Parse raw execution event traces and categorize each registered operation into key performance regimes (`launch_bound`, `compute_bound`, `memory_bound`, or `overhead`).
2. **Op Mix Roofline Analysis**: Calculate aggregate arithmetic intensity and determine the execution bound of a composite op mix against target hardware platform specs, accounting for memory access patterns.
3. **Hidden Device Sync Detection**: Identify implicitly blocking CPU-GPU synchronization points (e.g. dynamic shape queries, explicit `.item()`/`.cpu()` calls, and non-pinned host transfers) that introduce CPU bubble stalls and disrupt launch pipelines.

Finally, write regression tests in `tests/test_regression.py` that guard against regressions in hidden sync identification and workload classification.
