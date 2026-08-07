def simulate_tail(cfg, split):
    q = split["queue"]
    c = split["compile"]
    p = split["compute"]
    return q + c + p + (cfg["arrival_rate"] * 0.05)
