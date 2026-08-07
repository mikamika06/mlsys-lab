def compute_scaling_table(config, tp_degrees):
    out = []
    hs = config["hidden_size"]
    ffn = config["intermediate_size"]
    layers = config["num_layers"]
    for tp in tp_degrees:
        weight_mem = float(layers * (hs * ffn * 4 * 3) / (tp * 1024**3))
        comm_bytes = float(layers * 2 * hs * 4 * (tp - 1) / tp)
        out.append({"tp": int(tp), "memory_gb": weight_mem, "comm_bytes": comm_bytes})
    return out
