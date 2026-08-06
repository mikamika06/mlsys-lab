def compute_latency_ratio(ttft_enabled, ttft_disabled):
    if ttft_disabled <= 0:
        return 0.0
    return float(ttft_enabled) / float(ttft_disabled)
