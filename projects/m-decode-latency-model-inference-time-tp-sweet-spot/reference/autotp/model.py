def estimate_decode_latency(config, hw, tp_degree, batch_size):
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    weights_bytes = hidden * hidden * 4 * layers * 2 / tp_degree
    comm_bytes = hidden * 4 * 2 * (tp_degree - 1) / tp_degree * batch_size
    mem_time = weights_bytes / (hw["memory_bw_gbps"] * 1e9)
    comm_time = comm_bytes / (hw["comm_bw_gbps"] * 1e9)
    return float((mem_time + comm_time) * layers * batch_size)
