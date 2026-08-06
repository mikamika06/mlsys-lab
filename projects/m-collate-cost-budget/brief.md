# Collate Cost Budgeting for DataLoader Workers

## Symptom
Our distributed training workers frequently suffer from GPU starvation during the data ingestion phase. Host-side profiler traces show that DataLoader worker processes occasionally spend excessive time inside `collate_fn`, causing inter-batch delay spikes. While increasing the number of DataLoader workers reduces the average latency per worker, high-overhead collation functions (e.g., dynamic padding, image transforms, or string tokenization) still cause non-deterministic stalls when CPU core contention occurs.

We need a systematic cost budgeting and monitoring utility to profile, evaluate, and enforce latency/throughput constraints on custom collation functions before deploying datasets to training pipelines.

## Objectives
1. Implement a collation profiler and budget checker in `collate/budget.py` that measures per-sample item collate costs, batch collation overhead, and evaluates whether a given collation function stays within an allocated time budget per batch size.
2. Implement an adaptive batching strategy in `collate/adaptive.py` that dynamically scales batch sizes or trims optional sample processing steps to strictly respect a maximum target collate budget without exceeding specified maximum latency limits.
3. Write regression tests in `tests/test_regression.py` that verify collation cost tracking and correctly fail when a budget violation or unmonitored expensive collation step is injected into the pipeline.
