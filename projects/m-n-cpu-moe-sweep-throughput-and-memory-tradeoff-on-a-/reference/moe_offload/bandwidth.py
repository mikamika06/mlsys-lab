def derive_bandwidth(token_throughput, bytes_per_expert_load, active_experts_per_token):
    return float(token_throughput * bytes_per_expert_load * active_experts_per_token / (1024 ** 3))
