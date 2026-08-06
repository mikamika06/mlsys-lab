def decode_latency(config, tp_degree):
    hs = config["hidden_size"]
    layers = config["num_layers"]
    mem_bw = config["mem_bw"]
    comm_bw = config["comm_bw"]
    model_bytes = hs * hs * 4 * layers / tp_degree
    comm_cost = (hs * 4 * layers) / (comm_bw * max(1, tp_degree - 1)) if tp_degree > 1 else 0.0
    mem_cost = model_bytes / mem_bw
    return float(mem_cost + comm_cost)
