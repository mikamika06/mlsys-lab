# Incident Alerting and PromQL Diagnostics for vLLM Serving

Our production vLLM serving cluster is experiencing intermittent latency spikes and unhandled request drops under heavy load. On-call engineers are currently receiving unstructured log alerts that trigger too late, missing critical preemption cascades and cache thrashing events until client requests time out.

You are tasked with building an observability metric engine and alert verification pipeline for vLLM clusters.

Your tasks:
1. Map 10 distinct vLLM incident scenarios (covering queue growth, cache thrashing, preemption spikes, chunked prefill bottlenecks, and KV allocation failures) to their exact metric symptoms, and parse raw Prometheus metric samples to compute health state indicators.
2. Implement PromQL query generators to derive key serving indicators: p95 Time-To-First-Token (TTFT), KV cache memory utilization, waiting queue saturation ratios, and per-minute request preemption rates.
3. Build an alert rule evaluation engine with justified threshold rules that processes frozen cluster telemetry metrics, firing appropriate alerts during true failure modes without producing false positives on normal traffic spikes.
4. Write regression tests in `tests/test_regression.py` that validate alert threshold invariants and catch unsafe alerting configurations (such as suppressed preemption alerts during queue saturation).

All python sources are located under `vllm_obs/`.
