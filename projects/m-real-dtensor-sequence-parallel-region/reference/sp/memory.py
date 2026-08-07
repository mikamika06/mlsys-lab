def measure_memory(cfg, mode):
    s = cfg["seq_len"]
    h = cfg["hidden_size"]
    b = cfg["batch_size"]
    tp = cfg["tp_size"]
    base = s * h * b * 4
    if mode == "tp_only":
        return base
    elif mode == "tp_sp":
        return base // tp
    return base
