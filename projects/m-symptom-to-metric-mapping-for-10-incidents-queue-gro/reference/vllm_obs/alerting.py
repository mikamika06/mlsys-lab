"""Alerting rules and evaluation engine for vLLM telemetry."""


def evaluate_alerts(metrics_snapshot: dict, thresholds: dict) -> list:
    alerts = []

    ttft = metrics_snapshot.get("p95_ttft_seconds", 0.0)
    ttft_thresh = thresholds.get("p95_ttft_max", 2.0)
    if ttft > ttft_thresh:
        alerts.append({
            "name": "HighTTFT",
            "severity": "warning",
            "value": ttft,
            "threshold": ttft_thresh,
        })

    kv_usage = metrics_snapshot.get("kv_cache_utilization", 0.0)
    kv_thresh = thresholds.get("kv_cache_utilization_max", 0.90)
    if kv_usage > kv_thresh:
        alerts.append({
            "name": "KVCacheExhaustion",
            "severity": "critical",
            "value": kv_usage,
            "threshold": kv_thresh,
        })

    preempt_rate = metrics_snapshot.get("preemptions_per_min", 0.0)
    preempt_thresh = thresholds.get("preemptions_per_min_max", 10.0)
    if preempt_rate > preempt_thresh:
        alerts.append({
            "name": "HighPreemptionRate",
            "severity": "critical",
            "value": preempt_rate,
            "threshold": preempt_thresh,
        })

    queue_sat = metrics_snapshot.get("waiting_queue_saturation", 0.0)
    queue_thresh = thresholds.get("queue_saturation_max", 0.60)
    if queue_sat > queue_thresh:
        alerts.append({
            "name": "QueueSaturationHigh",
            "severity": "warning",
            "value": queue_sat,
            "threshold": queue_thresh,
        })

    return alerts
