# Ticket: Production Latency Decomposition & Goodput SLO Mismatches

## Incident Summary
During recent high-concurrency LLM inference benchmarks in vLLM, our customer-facing serving dashboards reported anomalous 95th percentile (p95) end-to-end latency (E2EL) spikes. Investigations revealed significant disagreements across monitoring stacks: Prometheus metrics using linear interpolation reported a p95 E2EL of 240ms, while APM tracing tools using nearest-rank percentile estimation reported 310ms on the exact same trace dataset. Because team members could not reliably attribute latency contributions between queuing delays, prefill execution, and token decode iterations, root-cause diagnosis stalled.

Furthermore, the infrastructure auto-scaler selected Configuration #7 based on raw total token throughput. Under peak production load, over 35% of incoming inference requests violated the Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT) Service Level Objectives (SLOs), causing widespread request timeouts despite high reported throughput.

## Task Overview
Build the core benchmarking library `latmetrics` to unify latency decomposition and configuration evaluation:
1. Implement percentile computation supporting both nearest-rank and linear interpolation methods, and decompose end-to-end p95 latency into exact phase contributions (queue, prefill, decode).
2. Build an SLO-aware configuration evaluator that calculates Goodput (tokens/sec meeting TTFT and TPOT SLOs) and ranks serving configurations strictly by Goodput rather than raw throughput.
3. Include a regression test suite in `tests/test_regression.py` that verifies SLO evaluation integrity and catches regressions where configurations are misranked by total throughput.
