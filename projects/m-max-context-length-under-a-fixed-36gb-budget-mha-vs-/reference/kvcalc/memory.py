def compute_kv_bytes(config, context_length, batch_size=1):
    t = config["type"]
    L = config["num_layers"]
    b = batch_size
    s = context_length
    bytes_e = config["bytes_per_elem"]
    if t in ("mha", "gqa"):
        kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        return 2 * L * b * s * kv_heads * head_dim * bytes_e
    elif t == "mla":
        latent_dim = config["kv_lora_rank"]
        rope_dim = config["qk_rope_head_dim"]
        return L * b * s * (latent_dim + rope_dim) * bytes_e
    else:
        raise ValueError(f"Unknown type {t}")
