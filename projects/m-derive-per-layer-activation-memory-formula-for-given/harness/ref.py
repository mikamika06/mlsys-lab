CONFIGS = [
    {"layers": [{"hidden_dim": 2048, "num_heads": 16}, {"hidden_dim": 2048, "num_heads": 16}]},
    {"layers": [{"hidden_dim": 4096, "num_heads": 32}]},
    {"layers": [{"hidden_dim": 1024, "num_heads": 8}, {"hidden_dim": 1024, "num_heads": 8}]}
]

def layer_activation_memory(b, s, h, heads, dtype_bytes):
    attn_mem = b * s * h * 2 * dtype_bytes + b * heads * s * s * dtype_bytes
    mlp_mem = b * s * (4 * h) * dtype_bytes
    return attn_mem + mlp_mem

def find_attention_mlp_crossover(b, h, heads, dtype_bytes):
    for s in range(1, 100000):
        attn = b * s * h * 2 * dtype_bytes + b * heads * s * s * dtype_bytes
        mlp = b * s * (4 * h) * dtype_bytes
        if attn > mlp:
            return s
    return 100000

def total_activation_memory(config, b, s, dtype_bytes):
    total = 0
    for layer in config["layers"]:
        h = layer["hidden_dim"]
        heads = layer["num_heads"]
        total += b * s * h * 2 * dtype_bytes + b * heads * s * s * dtype_bytes + b * s * (4 * h) * dtype_bytes
    return total
