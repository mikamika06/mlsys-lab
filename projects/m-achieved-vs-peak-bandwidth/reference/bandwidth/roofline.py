def compute_arithmetic_intensity(flops: float, bytes_transferred: float) -> float:
    """Compute arithmetic intensity in FLOPs per byte."""
    return float(flops / bytes_transferred)


def compute_roofline_bound(intensity: float, peak_tflops: float, peak_gbps: float) -> dict:
    """Determine roofline bound, knee intensity, and bottleneck state."""
    mem_bound_tflops = (intensity * peak_gbps) / 1000.0
    is_mem_bound = mem_bound_tflops < peak_tflops
    attainable_tflops = min(float(peak_tflops), float(mem_bound_tflops))
    knee_intensity = (peak_tflops * 1000.0) / peak_gbps

    return {
        "attainable_tflops": float(attainable_tflops),
        "is_memory_bound": bool(is_mem_bound),
        "knee_intensity": float(knee_intensity),
    }


def analyze_kernel_execution(
    flops: float,
    bytes_transferred: float,
    execution_time_sec: float,
    peak_tflops: float,
    peak_gbps: float,
) -> dict:
    """Perform comprehensive kernel roofline and bandwidth analysis."""
    achieved_tflops = (flops / 1e12) / execution_time_sec
    achieved_gbps = (bytes_transferred / 1e9) / execution_time_sec
    intensity = compute_arithmetic_intensity(flops, bytes_transferred)
    bound = compute_roofline_bound(intensity, peak_tflops, peak_gbps)

    pct_peak_bw = achieved_gbps / peak_gbps
    pct_attainable = achieved_tflops / bound["attainable_tflops"]

    return {
        "achieved_tflops": float(achieved_tflops),
        "achieved_gbps": float(achieved_gbps),
        "arithmetic_intensity": float(intensity),
        "is_memory_bound": bool(bound["is_memory_bound"]),
        "knee_intensity": float(bound["knee_intensity"]),
        "attainable_tflops": float(bound["attainable_tflops"]),
        "pct_peak_bandwidth": float(pct_peak_bw),
        "pct_attainable_performance": float(pct_attainable),
    }
