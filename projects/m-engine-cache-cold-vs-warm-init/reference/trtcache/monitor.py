def evaluate_latency_ratio(cold_latencies, warm_latencies):
    if not cold_latencies or not warm_latencies:
        return 999.0
    avg_cold = sum(cold_latencies) / len(cold_latencies)
    avg_warm = sum(warm_latencies) / len(warm_latencies)
    if avg_cold == 0:
        return 0.0
    return avg_warm / avg_cold
