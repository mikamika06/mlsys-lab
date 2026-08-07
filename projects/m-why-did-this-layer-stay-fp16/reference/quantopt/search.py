def find_budget_config(layers, max_bytes):
    best_config = {}
    for l in layers:
        if 8 in l.get("supported_bits", []) and l.get("params", 0) <= max_bytes:
            best_config[l["name"]] = 8
        else:
            best_config[l["name"]] = 16
    return best_config
