def compute_crossover(hidden_dim, num_experts, comm_bw, tflops):
    bytes_per_token = hidden_dim * 2
    comm_cost_per_token = bytes_per_token / (comm_bw * 1e9)
    compute_ops_per_token = 2.0 * hidden_dim * 2.0
    compute_cost_per_token = compute_ops_per_token / (tflops * 1e12)
    crossover_tokens = int(compute_cost_per_token / comm_cost_per_token * num_experts)
    return max(1, crossover_tokens)
