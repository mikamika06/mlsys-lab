"""Draft model step latency and acceptance rate modeling."""

import math


def compute_draft_latency(params_m, precision, system_config):
    """Computes single-step latency for a draft model in milliseconds."""
    bw = system_config["memory_bandwidth_gbps"]
    launch_overhead = system_config["kernel_launch_ms"]
    bytes_per_param = 2.0 if precision == "fp16" else 1.0
    weight_bytes = params_m * 1e6 * bytes_per_param
    transfer_ms = (weight_bytes / (bw * 1e9)) * 1000.0
    dequant_ms = system_config.get("dequant_overhead_ms", 0.05) if precision == "int8" else 0.0
    return launch_overhead + transfer_ms + dequant_ms


def compute_acceptance_rate(params_m, precision, base_alpha_max, alpha_scale_m):
    """Computes expected acceptance rate alpha for a given draft model size."""
    alpha = base_alpha_max * (1.0 - math.exp(-params_m / alpha_scale_m))
    if precision == "int8":
        alpha *= 0.95
    return max(0.0, min(1.0, alpha))
