def calculate_metrics(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_hidden_layers"]
    v = cfg["vocab_size"]
    i = cfg["intermediate_size"]
    params = v * h + l * (4 * h * h + 2 * h * i) + h * v
    flops = l * (12 * h * h * cfg["seq_len"] + 2 * h * i * cfg["seq_len"])
    return {"params": params, "flops": flops}
