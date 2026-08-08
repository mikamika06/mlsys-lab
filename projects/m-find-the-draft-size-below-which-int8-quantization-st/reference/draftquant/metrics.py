def compute_draft_latency(params_count, precision, bandwidth_gbps, launch_overhead_us):
    """Calculate single-step draft model latency in microseconds."""
    bytes_per_param = 2 if precision == "fp16" else 1
    total_bytes = params_count * bytes_per_param
    transfer_us = (total_bytes / (bandwidth_gbps * 1e9)) * 1e6
    return transfer_us + launch_overhead_us


def compute_speculative_throughput(target_latency, draft_latency, gamma, acceptance_rate):
    """Compute tokens per second achieved by speculative decoding."""
    expected_accepted = (1.0 - (acceptance_rate ** (gamma + 1))) / (1.0 - acceptance_rate) if acceptance_rate < 1.0 else (gamma + 1.0)
    cycle_time = target_latency + (gamma * draft_latency)
    return (expected_accepted / cycle_time) * 1e6
