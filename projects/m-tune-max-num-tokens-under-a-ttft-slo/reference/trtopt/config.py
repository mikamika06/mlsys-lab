def load_default_config():
    return {
        "candidate_tokens": [256, 512, 1024, 2048, 4096],
        "slo_ttft": 500.0,
        "arrival_rate": 10.0,
        "prefill_lengths": [128, 256, 512, 1024]
    }
