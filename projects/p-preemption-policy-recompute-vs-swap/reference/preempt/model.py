def recompute_cost(context_len, hidden_size, num_layers, tflops):
    ops_per_token = 2 * hidden_size * hidden_size * 4 * num_layers
    total_ops = context_len * ops_per_token
    return total_ops / (tflops * 1e12)

def swap_cost(num_tokens, bytes_per_token, pcie_bandwidth_gbps):
    total_bytes = num_tokens * bytes_per_token
    return total_bytes / (pcie_bandwidth_gbps * 1e9)

def breakeven_length(hidden_size, num_layers, tflops, bytes_per_token, pcie_bandwidth_gbps):
    ops_per_token = 2 * hidden_size * hidden_size * 4 * num_layers
    op_cost_per_token = ops_per_token / (tflops * 1e12)
    swap_cost_per_token = bytes_per_token / (pcie_bandwidth_gbps * 1e9)
    if op_cost_per_token <= 0:
        return float('inf')
    return swap_cost_per_token / op_cost_per_token
