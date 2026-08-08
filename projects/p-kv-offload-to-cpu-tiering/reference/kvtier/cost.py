def estimate_transfer_cost(num_tokens, bytes_per_token, bandwidth_gbps):
    if bandwidth_gbps <= 0:
        return float("inf")
    total_bytes = num_tokens * bytes_per_token
    cost_seconds = total_bytes / (bandwidth_gbps * 1024.0 * 1024.0 * 1024.0)
    return cost_seconds
