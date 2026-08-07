def get_default_config():
    return {
        "max_batch_size": 32,
        "timeout_ms": 10.0,
        "slo_p99_ms": 50.0,
        "split_queues": True,
        "burst_multiplier": 2.0
    }
