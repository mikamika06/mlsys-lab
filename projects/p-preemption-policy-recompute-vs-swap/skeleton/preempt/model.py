def recompute_cost(context_len, hidden_size, num_layers, tflops):
    raise NotImplementedError

def swap_cost(num_tokens, bytes_per_token, pcie_bandwidth_gbps):
    raise NotImplementedError

def breakeven_length(hidden_size, num_layers, tflops, bytes_per_token, pcie_bandwidth_gbps):
    raise NotImplementedError
