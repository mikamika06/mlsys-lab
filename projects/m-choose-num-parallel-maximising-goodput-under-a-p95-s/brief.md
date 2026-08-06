# Incident Report: Low Throughput, Intermittent 503 Ingress Drops, and P95 Violations

## Symptom Description
Our local production inference runner cluster for the batch text-generation endpoint is failing its Service Level Objective (SLO). The target P95 request latency threshold is strictly 250 ms. However, under high traffic load, client callers report frequent HTTP 503 Service Unavailable errors and extreme tail latencies.

Curiously, metrics show that the overall GPU compute utilization hovers around 40% when 503 drops begin to occur, well below physical saturation limits. Infrastructure monitoring traces indicate that incoming requests spend excessive durations queued in front of active processing slots before execution begins. Furthermore, occasional ultra-long prefill requests appear to stall subsequent short prompt evaluations, degrading overall goodput (requests served successfully within the P95 SLO).

## Task
You need to analyze the concurrency parameters and execution traces for our endpoint service:
1. Determine the optimal `num_parallel` slot configuration that maximizes useful goodput while staying within the target P95 latency SLO.
2. Identify the root cause of the unexpected 503 status responses during partial GPU utilization and identify instances of head-of-line (HoL) blocking caused by massive prefill tasks.
3. Provide automated regression test coverage to safeguard concurrency planning and diagnostic logic against regressions.
