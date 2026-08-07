# Capacity Planning & SLO Sizing from Real Inference Logs

## Context
Production services using LLM serving engines generate rich Prometheus metrics and benchmark execution logs. However, capacity planners often confuse raw throughput with true usable capacity (goodput) that satisfies tight Service Level Objectives (SLOs).

Our product team has committed to an incoming traffic guarantee:
- Target throughput: **40 requests per second** (RPS)
- Target latency SLO: **p95 latency <= 2.5 seconds**

We have captured raw telemetry logs (`metrics.jsonl`) and load benchmark results (`bench_results.json`) from a single-replica baseline node running a 70B parameter model.

## Task
You must build an automated capacity planning tool inside `capacity/planner.py` that analyzes raw logs, derives actual goodput capacity curves, and computes required replica counts and operating costs under different configurations.

Your capacity planner must implement:
1. Parsing metrics and benchmark logs into structured time-series and query telemetry.
2. Distinguishing request-level throughput from goodput that meets latency constraints.
3. Constructing an empirical Latency-vs-Load curve to find single-replica knee capacity.
4. Calculating required cluster replica counts to sustain 40 RPS at p95 <= 2.5s with a safety headroom multiplier.
5. Computing operating cost ($ per 1M generated tokens) based on hardware instance rates.
6. Modeling the impact of Prefix Caching on system throughput and required replica sizing.

Finally, write regression tests in `tests/test_regression.py` that validate your capacity estimator against capacity edge cases and parameter violations.
