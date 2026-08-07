def get_layer_shapes(config: dict) -> dict[str, tuple[int, int]]:
    h = config["hidden_size"]
    inter = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = h // n_heads
    kv_dim = n_kv_heads * head_dim
    return {
        "q_proj": (h, h),
        "k_proj": (h, kv_dim),
        "v_proj": (h, kv_dim),
        "o_proj": (h, h),
        "gate_proj": (h, inter),
        "up_proj": (h, inter),
        "down_proj": (inter, h),
    }


def count_base_params(config: dict) -> int:
    h = config["hidden_size"]
    v = config["vocab_size"]
    layers = config["num_hidden_layers"]
    shapes = get_layer_shapes(config)
    attn = sum(
        din * dout
        for name, (din, dout) in shapes.items()
        if name in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    mlp = sum(
        din * dout
        for name, (din, dout) in shapes.items()
        if name in ("gate_proj", "up_proj", "down_proj")
    )
    layer_total = attn + mlp + 2 * h
    embeddings = v * h
    final_norm = h
    lm_head = v * h
    return embeddings + layers * layer_total + final_norm + lm_head
