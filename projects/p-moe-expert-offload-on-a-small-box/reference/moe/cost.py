def estimate_transfer_cost(expert_size_bytes, bandwidth_bps):
    if bandwidth_bps <= 0:
        return float("inf")
    return expert_size_bytes / bandwidth_bps
