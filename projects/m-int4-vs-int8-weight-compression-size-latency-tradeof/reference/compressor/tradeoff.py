def compute_compressed_size(in_features, out_features, bits):
    total_params = in_features * out_features
    if bits == 4:
        bytes_weight = (total_params + 1) // 2
        scales = (total_params // 128) * 4
        return bytes_weight + scales
    elif bits == 8:
        bytes_weight = total_params
        scales = (total_params // 128) * 4
        return bytes_weight + scales
    else:
        raise ValueError("Unsupported bitwidth")

def simulate_latency(in_features, out_features, bits, throughput_factor=1.0):
    size = compute_compressed_size(in_features, out_features, bits)
    base_latency = size / (1024.0 * 1024.0)
    if bits == 4:
        return base_latency * 1.2 * throughput_factor
    else:
        return base_latency * 0.9 * throughput_factor

def evaluate_tradeoff(config):
    s4 = compute_compressed_size(config["in_features"], config["out_features"], 4)
    s8 = compute_compressed_size(config["in_features"], config["out_features"], 8)
    l4 = simulate_latency(config["in_features"], config["out_features"], 4)
    l8 = simulate_latency(config["in_features"], config["out_features"], 8)
    return {"size_int4": s4, "size_int8": s8, "latency_int4": l4, "latency_int8": l8}

def compute_throughput_ratio(config):
    t = evaluate_tradeoff(config)
    return float(t["latency_int8"]) / float(t["latency_int4"])
