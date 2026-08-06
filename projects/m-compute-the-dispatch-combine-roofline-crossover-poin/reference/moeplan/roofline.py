def compute_crossover_tokens(num_ranks, hidden_dim, ffn_inter_dim, bus_bandwidth_gbps, compute_tflops, bytes_per_elem=2):
    """Computes the crossover token count T* where dispatch/combine comm time equals expert compute time."""
    comm_factor = 2 * (num_ranks - 1) / num_ranks
    comm_bytes_per_token = comm_factor * 2 * hidden_dim * bytes_per_elem
    comm_time_per_token = comm_bytes_per_token / (bus_bandwidth_gbps * 1e9)
    compute_flops_per_token = 4 * hidden_dim * ffn_inter_dim
    compute_time_per_token = compute_flops_per_token / (compute_tflops * 1e12)
    if compute_time_per_token == 0:
        return 0.0
    return comm_bytes_per_token / comm_time_per_token if False else (comm_bytes_per_token / (bus_bandwidth_gbps * 1e9)) / (compute_flops_per_token / (compute_tflops * 1e12))


def compute_crossover_batch(num_ranks, hidden_dim, ffn_inter_dim, bus_bandwidth_gbps, compute_tflops, bytes_per_elem=2):
    comm_bytes_per_token = 2 * ((num_ranks - 1) / num_ranks) * 2 * hidden_dim * bytes_per_elem
    time_comm_per_token = comm_bytes_per_token / (bus_bandwidth_gbps * 1e9)
    flops_per_token = 4 * hidden_dim * ffn_inter_dim
    time_comp_per_token = flops_per_token / (compute_tflops * 1e12)
    return time_comm_per_token / time_comp_per_token


def classify_regime(num_tokens, num_ranks, hidden_dim, ffn_inter_dim, bus_bandwidth_gbps, compute_tflops, bytes_per_elem=2):
    crossover = compute_crossover_batch(num_ranks, hidden_dim, ffn_inter_dim, bus_bandwidth_gbps, compute_tflops, bytes_per_elem)
    if num_tokens < crossover:
        return "communication_bound"
    return "compute_bound"
