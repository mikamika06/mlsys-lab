def roofline_ceiling(intensity: float, hw_spec: dict) -> float:
    peak_flops = float(hw_spec["peak_flops_per_sec"])
    peak_bw = float(hw_spec["peak_bandwidth_bytes_sec"])
    mem_bound_ceiling = peak_bw * float(intensity)
    return min(peak_flops, mem_bound_ceiling)


def classify_kernel(intensity: float, hw_spec: dict) -> str:
    ridge_point = float(hw_spec["peak_flops_per_sec"]) / float(hw_spec["peak_bandwidth_bytes_sec"])
    if float(intensity) < ridge_point:
        return "memory_bound"
    return "compute_bound"


def kernel_performance_bound(kernel_stats: dict, hw_spec: dict) -> dict:
    intensity = float(kernel_stats["intensity"])
    ceiling_flops_sec = roofline_ceiling(intensity, hw_spec)
    bound_type = classify_kernel(intensity, hw_spec)
    total_flops = float(kernel_stats["total_flops"])
    total_time_us = float(kernel_stats["total_time_us"])
    min_time_us = (total_flops / ceiling_flops_sec * 1e6) if ceiling_flops_sec > 0 else 0.0
    achieved_flops_sec = (total_flops / (total_time_us * 1e-6)) if total_time_us > 0 else 0.0
    efficiency = achieved_flops_sec / ceiling_flops_sec if ceiling_flops_sec > 0 else 0.0
    headroom_speedup = total_time_us / min_time_us if min_time_us > 0 else 1.0
    return {
        "ceiling_flops_sec": ceiling_flops_sec,
        "bound_type": bound_type,
        "min_time_us": min_time_us,
        "achieved_flops_sec": achieved_flops_sec,
        "efficiency": efficiency,
        "headroom_speedup": headroom_speedup
    }
