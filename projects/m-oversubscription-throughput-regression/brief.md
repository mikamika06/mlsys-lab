# Oversubscription Throughput Regression in CPU Inference Streams

## Symptom

A batch inference server running on CPU exhibits a sharp throughput drop when scaling up the number of worker streams under high load. With a moderate stream count, token processing throughput scales linearly. However, as the number of parallel streams exceeds the available physical CPU core capacity, the overall system throughput regresses dramatically—dropping well below the single-stream baseline despite 100% CPU utilization.

Profiles indicate high context-switching overhead and cache invalidation when thread workers contend for CPU resources. We need an automated profiling tool and thread scheduler helper to analyze the stream-count sweep, identify the knee point where oversubscription degrades performance, and automatically clamp the stream concurrency to restore optimal execution throughput.

## Tasks

1. Implement `sweep_stream_count(bench_fn, max_streams)` in `oversub/scheduler.py` to measure system throughput across worker stream counts and detect the knee point where throughput peaks before decaying due to oversubscription.
2. Implement `OptimalStreamPool` in `oversub/scheduler.py` which dynamically bounds stream execution to the identified knee point and calculates the expected throughput ratio gain over oversubscribed execution.
3. Write a regression test in `tests/test_regression.py` that verifies the detected knee point stays strictly within physical execution limits and fails if oversubscribed stream counts are selected.
