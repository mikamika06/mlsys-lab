# GPU and ANE Profile Parsing: Active Residency and Token Rate Correlation

## Symptom
Our telemetry pipeline ingests Apple Silicon system metrics via `powermetrics` logs to track hardware resource utilization during edge LLM inference. During a recent optimization run on an M-series system, engineers noticed severe throughput dips (measured in tokens/second) across different LLM configurations. However, our automated dashboards failed to alert on these drops because our parser currently extracts global cpu and raw power metrics without isolating **GPU active residency %** or correlating hardware activity with model scale. Furthermore, Apple Neural Engine (ANE) power readings were completely ignored, making it impossible to determine whether memory-bandwidth bottlenecks or execution unit stalls were causing the performance degradation.

We need a dedicated profiling library that can parse raw multi-sample `powermetrics` outputs, compute accurate active residency metrics for both the GPU and ANE, and correlate GPU active residency against actual inference performance (tokens/sec) across varied model scales.

## What You Need To Do
1. Build a robust `powermetrics` parser that reads raw textual output (specifically extracting GPU idle/active state residency percentages and ANE power usage across recorded interval samples).
2. Implement a telemetry analyzer that computes average GPU active residency per run and correlates it against measured generation throughput (`tokens_sec`) across multiple model sizes, computing the correlation ratio and active-residency-per-token efficiency metric.
3. Extract ANE power samples, calculate average ANE wattage, and estimate ANE utilization against peak platform capacity.
4. Write a regression test suite in `tests/test_regression.py` that validates active residency extraction invariants and catches invalid or overly permissive parsing logic.
