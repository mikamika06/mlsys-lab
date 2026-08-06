import numpy as np


def compute_roofline_points(kernels, peak_flops, peak_bw):
    results = []
    for k in kernels:
        flops = k["flops"]
        bytes_trans = k["bytes"]
        time_s = k["time_s"]
        ai = flops / bytes_trans if bytes_trans > 0 else 0.0
        gflops = (flops / 1e9) / time_s if time_s > 0 else 0.0
        ridge_point = peak_flops / peak_bw
        bound = "compute" if ai >= ridge_point else "memory"
        attainable = min(peak_flops, ai * peak_bw)
        efficiency = gflops / attainable if attainable > 0 else 0.0
        results.append({
            "name": k["name"],
            "ai": float(ai),
            "gflops": float(gflops),
            "bound": bound,
            "efficiency": float(efficiency)
        })
    return results


def classify_kernels(metrics):
    return [m["bound"] for m in metrics]


def compare_attention(std_metrics, flash_metrics):
    std_ai = std_metrics["ai"]
    flash_ai = flash_metrics["ai"]
    if flash_ai > std_ai:
        return "flash"
    elif std_ai > flash_ai:
        return "standard"
    return "equal"


def find_limiter(kernel_metric, peak_flops, peak_bw):
    ai = kernel_metric["ai"]
    gflops = kernel_metric["gflops"]
    ridge_point = peak_flops / peak_bw
    attainable = min(peak_flops, ai * peak_bw)
    ratio = gflops / attainable if attainable > 0 else 0.0
    if ratio < 0.5:
        return "latency_or_overhead_bound"
    elif ai < ridge_point:
        return "memory_bandwidth_bound"
    else:
        return "compute_bound"
