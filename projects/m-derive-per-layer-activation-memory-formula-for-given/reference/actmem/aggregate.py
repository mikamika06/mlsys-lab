from actmem.formula import compute_layer_activation_bytes

def total_uncheckpointed_memory(config, batch_size, seq_len, dtype_bytes):
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    num_heads = config["num_heads"]
    total = 0.0
    for _ in range(num_layers):
        total += compute_layer_activation_bytes(batch_size, seq_len, hidden_dim, num_heads, dtype_bytes)
    return total
