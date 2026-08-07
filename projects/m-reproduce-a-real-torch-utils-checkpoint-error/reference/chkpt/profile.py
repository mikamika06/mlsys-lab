def measure(n_layers, checkpoint):
    total_mem = float(n_layers * 100.0)
    if checkpoint:
        interval = max(1, n_layers // 2)
        mem = total_mem * (1.0 / interval) + (n_layers * 5.0)
        time_val = float(n_layers * 1.0 + (n_layers / interval) * 0.2)
    else:
        mem = total_mem
        time_val = float(n_layers * 1.0)
    return float(mem), float(time_val)
