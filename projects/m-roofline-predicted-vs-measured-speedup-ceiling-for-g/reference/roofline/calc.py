def compute_intensity(flops_per_token, bytes_per_token):
    if bytes_per_token <= 0:
        return float("inf")
    return float(flops_per_token) / float(bytes_per_token)


def roofline_speedup_ceiling(arithmetic_intensity, peak_flops, peak_bandwidth, baseline_time, token_flops, gamma=4):
    hw_intensity = float(peak_flops) / float(peak_bandwidth)
    perf = float(peak_flops) if arithmetic_intensity >= hw_intensity else float(arithmetic_intensity) * float(peak_bandwidth)
    time_per_token = float(token_flops) / perf if perf > 0 else 0.0
    spec_total = time_per_token * (1.0 + float(gamma))
    effective_time_per_token = spec_total / (1.0 + float(gamma))
    if effective_time_per_token <= 0:
        return 1.0
    ceiling = float(baseline_time) / effective_time_per_token
    return float(ceiling)


def measure_speedup_error(predicted_ceiling, measured_speedup):
    if predicted_ceiling == 0.0:
        return 0.0
    return abs(float(predicted_ceiling) - float(measured_speedup)) / abs(float(predicted_ceiling))
