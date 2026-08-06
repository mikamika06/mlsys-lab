def classify_roofline(arithmetic_intensity, peak_flops, peak_bandwidth):
    ridge_point = peak_flops / peak_bandwidth
    if arithmetic_intensity < ridge_point:
        return "memory_bound"
    return "compute_bound"
