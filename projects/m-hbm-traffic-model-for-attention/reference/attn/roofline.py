def classify_roofline_bound(intensity, peak_flops_per_sec, peak_bw_bytes_per_sec):
    ridge_intensity = peak_flops_per_sec / peak_bw_bytes_per_sec
    if intensity < ridge_intensity:
        return "memory"
    return "compute"
