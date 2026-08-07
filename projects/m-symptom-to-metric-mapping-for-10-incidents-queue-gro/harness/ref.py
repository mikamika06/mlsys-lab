"""Reference data generation and test fixtures for harness checkers."""

TEST_SNAPSHOTS = [
    {
        "input": {
            "p95_ttft_seconds": 1.2,
            "kv_cache_utilization": 0.70,
            "preemptions_per_min": 2.0,
            "waiting_queue_saturation": 0.25,
        },
        "thresholds": {
            "p95_ttft_max": 2.0,
            "kv_cache_utilization_max": 0.90,
            "preemptions_per_min_max": 10.0,
            "queue_saturation_max": 0.60,
        },
        "expected_alert_count": 0,
    },
    {
        "input": {
            "p95_ttft_seconds": 2.8,
            "kv_cache_utilization": 0.96,
            "preemptions_per_min": 18.0,
            "waiting_queue_saturation": 0.80,
        },
        "thresholds": {
            "p95_ttft_max": 2.0,
            "kv_cache_utilization_max": 0.90,
            "preemptions_per_min_max": 10.0,
            "queue_saturation_max": 0.60,
        },
        "expected_alert_count": 4,
    },
]

METRIC_SAMPLES = [
    {
        "raw": {
            "vllm:num_requests_waiting": 40,
            "vllm:num_requests_running": 10,
            "vllm:gpu_cache_usage_perc": 0.95,
            "vllm:num_preemptions_total": 12,
        },
        "expected": {
            "queue_saturation": 0.8,
            "gpu_cache_usage": 0.95,
            "is_thrashing": True,
            "is_queue_saturated": True,
        },
    },
    {
        "raw": {
            "vllm:num_requests_waiting": 2,
            "vllm:num_requests_running": 18,
            "vllm:gpu_cache_usage_perc": 0.40,
            "vllm:num_preemptions_total": 0,
        },
        "expected": {
            "queue_saturation": 0.1,
            "gpu_cache_usage": 0.40,
            "is_thrashing": False,
            "is_queue_saturated": False,
        },
    },
]
