CONFIGS = [
    {
        "layers": [
            {"index": 0, "kind": "full", "kv_heads": 8, "head_dim": 128},
            {"index": 1, "kind": "sliding", "window": 256, "kv_heads": 8, "head_dim": 128},
            {"index": 2, "kind": "sliding", "window": 256, "kv_heads": 8, "head_dim": 128},
            {"index": 3, "kind": "full", "kv_heads": 8, "head_dim": 128}
        ]
    },
    {
        "layers": [
            {"index": 0, "kind": "full", "kv_heads": 4, "head_dim": 64},
            {"index": 1, "kind": "full", "kv_heads": 4, "head_dim": 64}
        ]
    },
    {
        "layers": [
            {"index": 0, "kind": "sliding", "window": 512, "kv_heads": 2, "head_dim": 128},
            {"index": 1, "kind": "sliding", "window": 512, "kv_heads": 2, "head_dim": 128}
        ]
    }
]

def classify_attention(layer_cfg):
    if layer_cfg.get("window") is not None and layer_cfg.get("window") > 0:
        return "sliding"
    return "full"

def effective_bytes_per_token(config, seq_len, dtype_size=2):
    total_bytes = 0
    for layer in config.get("layers", []):
        kv_heads = layer.get("kv_heads", 1)
        head_dim = layer.get("head_dim", 128)
        variant = classify_attention(layer)
        if variant == "sliding":
            window = layer.get("window", seq_len)
            active_len = min(seq_len, window)
        else:
            active_len = seq_len
        layer_bytes = 2 * active_len * kv_heads * head_dim * dtype_size
        total_bytes += layer_bytes
    return total_bytes

def predict_startup_kv_size(config, max_seq_len, dtype_size=2):
    return effective_bytes_per_token(config, max_seq_len, dtype_size)
