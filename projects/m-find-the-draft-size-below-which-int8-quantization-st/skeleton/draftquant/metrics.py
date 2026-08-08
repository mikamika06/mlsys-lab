def compute_draft_latency(params_count, precision, bandwidth_gbps, launch_overhead_us):
    raise NotImplementedError


def compute_speculative_throughput(target_latency, draft_latency, gamma, acceptance_rate):
    raise NotImplementedError
