from preempt.swap import compute_swap_cost


def choose_preemption_mode(workload_profile):
    """Select the cheaper preemption mode for a given workload profile."""
    recompute_tokens = workload_profile["recompute_tokens"]
    tps = workload_profile["token_processing_rate_tps"]
    recompute_time = recompute_tokens / tps

    num_blocks = workload_profile["num_blocks"]
    block_bytes = workload_profile["block_bytes"]
    pcie_bw = workload_profile["pcie_bandwidth_gbps"]
    roundtrip = workload_profile.get("roundtrip", True)

    swap_res = compute_swap_cost(num_blocks, block_bytes, pcie_bw, roundtrip=roundtrip)
    swap_time = swap_res["time_seconds"]

    if swap_time < recompute_time:
        return "swap"
    return "recompute"
