# Incident Report: Production TTFT Spikes and Alert Storms in Cluster kv-prod-04

**Severity:** P1
**Component:** KV Cache Observability & Alerting
**Reported By:** On-Call SRE

## Symptom Description
During high-traffic periods, our latency monitoring system generates hundreds of rapidly toggling firing and resolution alerts within a 15-minute window. Concurrently, downstream services report severe time-to-first-token (TTFT) latency degradation on model serving endpoints.

Our current monitoring stack cannot differentiate whether TTFT spikes stem from compute bottlenecks, prefix cache misses, or total KV cache exhaustion. In addition, raw Prometheus histogram metrics produce distorted latency percentiles because quantile aggregation lacks linear bucket interpolation in our telemetry aggregation pipeline.

## Deliverables
1. Implement Prometheus histogram quantile calculation using linear bucket interpolation in `kvobs/histogram.py`.
2. Build an incident triage classifier in `kvobs/triage.py` that identifies KV cache saturation (100% cache usage coupled with a 5x TTFT spike over baseline) versus compute or prefix miss issues.
3. Implement a stateful hysteresis alert filter in `kvobs/alerting.py` to eliminate alert flapping across noisy metric boundaries.
4. Provide regression tests in `tests/test_regression.py` that verify bucket interpolation accuracy and alerting stability.
