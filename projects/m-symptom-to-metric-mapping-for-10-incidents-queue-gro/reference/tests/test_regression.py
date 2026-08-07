"""Regression tests for vLLM alerting rules."""

import sys
sys.path.insert(0, ".")

from vllm_obs.alerting import evaluate_alerts


def test_alert_thresholds():
    snapshot = {
        "p95_ttft_seconds": 3.5,
        "kv_cache_utilization": 0.95,
        "preemptions_per_min": 15.0,
        "waiting_queue_saturation": 0.75,
    }
    thresholds = {
        "p95_ttft_max": 2.0,
        "kv_cache_utilization_max": 0.90,
        "preemptions_per_min_max": 10.0,
        "queue_saturation_max": 0.60,
    }

    alerts = evaluate_alerts(snapshot, thresholds)
    alert_names = [a["name"] for a in alerts]

    assert "HighTTFT" in alert_names, "Missing HighTTFT alert"
    assert "KVCacheExhaustion" in alert_names, "Missing KVCacheExhaustion alert"
    assert "HighPreemptionRate" in alert_names, "Missing HighPreemptionRate alert"
    assert "QueueSaturationHigh" in alert_names, "Missing QueueSaturationHigh alert"


def test_no_false_positive_on_normal_traffic():
    snapshot = {
        "p95_ttft_seconds": 0.8,
        "kv_cache_utilization": 0.50,
        "preemptions_per_min": 0.0,
        "waiting_queue_saturation": 0.10,
    }
    thresholds = {
        "p95_ttft_max": 2.0,
        "kv_cache_utilization_max": 0.90,
        "preemptions_per_min_max": 10.0,
        "queue_saturation_max": 0.60,
    }

    alerts = evaluate_alerts(snapshot, thresholds)
    assert len(alerts) == 0, f"Expected 0 alerts on normal traffic, got {len(alerts)}"
