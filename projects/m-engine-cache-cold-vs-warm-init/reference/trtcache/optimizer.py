def evaluate_latency_ratio(cold_latency, warm_latency):
    if cold_latency <= 0:
        return 0.0
    return float(warm_latency) / float(cold_latency)
