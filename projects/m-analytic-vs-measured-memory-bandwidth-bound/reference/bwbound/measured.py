from bwbound.analytic import compute_analytic_bound


def analyze_measured_performance(measured_time_sec, total_bytes, flops, peak_bandwidth_gbps, peak_tflops):
    """Analyze empirical measurement against theoretical roofline bounds."""
    time_mem = total_bytes / (peak_bandwidth_gbps * 1e9)
    time_compute = flops / (peak_tflops * 1e12)
    analytic_time = max(time_mem, time_compute)

    achieved_gbps = (total_bytes / measured_time_sec) / 1e9
    achieved_tflops = (flops / measured_time_sec) / 1e12
    bandwidth_utilization = achieved_gbps / peak_bandwidth_gbps
    compute_utilization = achieved_tflops / peak_tflops
    rel_err = abs(measured_time_sec - analytic_time) / analytic_time

    return {
        "achieved_gbps": achieved_gbps,
        "achieved_tflops": achieved_tflops,
        "bandwidth_utilization": bandwidth_utilization,
        "compute_utilization": compute_utilization,
        "rel_err": rel_err,
        "efficiency_ratio": analytic_time / measured_time_sec,
    }
