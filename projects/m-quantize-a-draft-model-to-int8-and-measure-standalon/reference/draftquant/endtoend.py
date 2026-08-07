def is_net_win(latency_ratio, acceptance_ratio, verification_overhead_ratio=0.3):
    effective_speedup = latency_ratio * (acceptance_ratio + verification_overhead_ratio) / (1.0 + verification_overhead_ratio)
    return effective_speedup > 1.0, float(effective_speedup)
