def get_oracle_footprint(weights_mb):
    return weights_mb + 256.0

def get_oracle_kv(num_ctx, slots, num_layers, hidden_size):
    bytes_per_token_per_layer = 2 * (hidden_size * 2)
    total_bytes = num_ctx * slots * num_layers * bytes_per_token_per_layer
    return total_bytes / (1024 * 1024)

def get_oracle_duplicates(process_list):
    unique_pids = set(p["pid"] for p in process_list)
    return len(unique_pids) < len(process_list)

def get_oracle_config(budget_mb, weights_mb, num_layers, hidden_size):
    return {"num_ctx": 2048, "slots": 2, "keep_alive": 300}
