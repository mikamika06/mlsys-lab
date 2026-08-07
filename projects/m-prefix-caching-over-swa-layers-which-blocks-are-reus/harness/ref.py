CONFIGS = [
    {
        "total_tokens": 16384,
        "block_size": 16,
        "layers": [
            {"index": 0, "kind": "full", "kv_heads": 8, "head_dim": 128},
            {"index": 1, "kind": "sliding", "window": 1024, "kv_heads": 8, "head_dim": 128},
            {"index": 2, "kind": "mamba", "state_dim": 64, "d_inner": 2048},
            {"index": 3, "kind": "sliding", "window": 1024, "kv_heads": 8, "head_dim": 128},
        ]
    },
    {
        "total_tokens": 32768,
        "block_size": 16,
        "layers": [
            {"index": 0, "kind": "full", "kv_heads": 4, "head_dim": 128},
            {"index": 1, "kind": "sliding", "window": 2048, "kv_heads": 4, "head_dim": 128},
            {"index": 2, "kind": "mamba", "state_dim": 128, "d_inner": 4096},
        ]
    },
    {
        "total_tokens": 65536,
        "block_size": 32,
        "layers": [
            {"index": 0, "kind": "full", "kv_heads": 8, "head_dim": 64},
            {"index": 1, "kind": "sliding", "window": 4096, "kv_heads": 8, "head_dim": 64},
            {"index": 2, "kind": "sliding", "window": 4096, "kv_heads": 8, "head_dim": 64},
            {"index": 3, "kind": "mamba", "state_dim": 64, "d_inner": 1024},
        ]
    }
]


def find_reusable_blocks(config):
    total = config["total_tokens"]
    bs = config["block_size"]
    num_blocks = total // bs
    reusable = []
    for b_idx in range(num_blocks):
        block_start = b_idx * bs
        block_end = (b_idx + 1) * bs
        is_reusable = True
        for layer in config["layers"]:
            if layer["kind"] == "sliding":
                window = layer["window"]
                if block_end <= (total - window):
                    is_reusable = False
                    break
        if is_reusable:
            reusable.append(b_idx)
    return reusable


def compute_mamba_state_size(layer, dtype_bytes=2):
    if layer["kind"] != "mamba":
        return 0
    state_dim = layer["state_dim"]
    d_inner = layer["d_inner"]
    return 2 * d_inner * state_dim * dtype_bytes


def compute_dense_vs_hybrid_cost(config, context_length, dtype_bytes=2):
    dense_bytes = 0
    hybrid_bytes = 0
    num_full_layers = sum(1 for l in config["layers"] if l["kind"] in ("full", "sliding"))
    total_layers = len(config["layers"])
    for layer in config["layers"]:
        if layer["kind"] in ("full", "sliding"):
            kv_heads = layer["kv_heads"]
            head_dim = layer["head_dim"]
            layer_bytes = 2 * context_length * kv_heads * head_dim * dtype_bytes
            dense_bytes += layer_bytes
            hybrid_bytes += layer_bytes
        elif layer["kind"] == "mamba":
            s_bytes = compute_mamba_state_size(layer, dtype_bytes)
            hybrid_bytes += s_bytes
            dense_bytes += 2 * context_length * 8 * 128 * dtype_bytes
    return {"dense_bytes": dense_bytes, "hybrid_bytes": hybrid_bytes}
