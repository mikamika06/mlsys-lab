CONFIGS = [
    {
        "layers": [
            {"index": 0, "type": "embd", "hidden_dim": 4096, "ffn_dim": 11008},
            {"index": 1, "type": "attn", "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128},
            {"index": 2, "type": "attn", "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128},
            {"index": 3, "type": "output", "hidden_dim": 4096, "vocab_size": 32000}
        ],
        "bytes_per_param": 2
    },
    {
        "layers": [
            {"index": 0, "type": "embd", "hidden_dim": 5120, "ffn_dim": 13824},
            {"index": 1, "type": "attn", "hidden_dim": 5120, "kv_heads": 8, "head_dim": 128},
            {"index": 2, "type": "attn", "hidden_dim": 5120, "kv_heads": 8, "head_dim": 128},
            {"index": 3, "type": "attn", "hidden_dim": 5120, "kv_heads": 8, "head_dim": 128},
            {"index": 4, "type": "output", "hidden_dim": 5120, "vocab_size": 32000}
        ],
        "bytes_per_param": 2
    },
    {
        "layers": [
            {"index": 0, "type": "embd", "hidden_dim": 2048, "ffn_dim": 5632},
            {"index": 1, "type": "attn", "hidden_dim": 2048, "kv_heads": 4, "head_dim": 128},
            {"index": 2, "type": "output", "hidden_dim": 2048, "vocab_size": 32000}
        ],
        "bytes_per_param": 1
    }
]

def layer_bytes(layer, bpp):
    t = layer["type"]
    if t == "embd":
        return layer["hidden_dim"] * layer["ffn_dim"] * bpp
    elif t == "attn":
        h = layer["hidden_dim"]
        kv = layer.get("kv_heads", 1)
        hd = layer.get("head_dim", 128)
        return (h * h + h * kv * hd * 2) * bpp
    elif t == "output":
        return layer["hidden_dim"] * layer["vocab_size"] * bpp
    return 1024 * bpp

def compute_layer_sizes(config):
    bpp = config["bytes_per_param"]
    return [layer_bytes(l, bpp) for l in config["layers"]]

def compute_tensor_split(config):
    sizes = compute_layer_sizes(config)
    total = sum(sizes)
    target = total / 2.0
    current = 0.0
    split_idx = len(sizes)
    for i, s in enumerate(sizes):
        if current + s > target and abs((current + s) - target) >= abs(current - target):
            split_idx = i
            break
        current += s
    sum0 = sum(sizes[:split_idx])
    sum1 = sum(sizes[split_idx:])
    if sum0 == 0 or sum1 == 0:
        split_idx = max(1, len(sizes) // 2)
        sum0 = sum(sizes[:split_idx])
        sum1 = sum(sizes[split_idx:])
    return [sum0 / total, sum1 / total]
