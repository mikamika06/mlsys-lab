CONFIGS = [
    {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "weight_bytes": 30 * 1024 * 1024 * 1024
    },
    {
        "num_hidden_layers": 40,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "weight_bytes": 15 * 1024 * 1024 * 1024
    },
    {
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "weight_bytes": 70 * 1024 * 1024 * 1024
    }
]

def compute_kv_bytes(config, seq_len, batch_size, dtype_bytes):
    num_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    total_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    return batch_size * seq_len * total_bytes_per_token

def dtype_comparison_table(config, seq_len, batch_size):
    dtypes = {"fp16": 2, "fp8": 1, "int4": 0.5}
    table = {}
    for name, b in dtypes.items():
        table[name] = compute_kv_bytes(config, seq_len, batch_size, b)
    return table
