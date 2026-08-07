def compute_intensity(flops_per_token, bytes_per_token):
    raise NotImplementedError


def roofline_speedup_ceiling(arithmetic_intensity, peak_flops, peak_bandwidth, baseline_time, token_flops, gamma=4):
    raise NotImplementedError


def measure_speedup_error(predicted_ceiling, measured_speedup):
    raise NotImplementedError
