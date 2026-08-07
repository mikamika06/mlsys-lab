def verify_partition_shapes(config: dict, tp_degree: int, log_entry: dict) -> bool:
    if tp_degree <= 0 or (tp_degree & (tp_degree - 1)) != 0:
        return False

    attn_heads = config["num_attention_heads"]
    kv_heads = config["num_kv_heads"]
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    head_dim = config.get("head_dim", hidden // attn_heads)

    if (
        attn_heads % tp_degree != 0
        or kv_heads % tp_degree != 0
        or hidden % tp_degree != 0
        or inter % tp_degree != 0
    ):
        return False

    shards = log_entry.get("shards")
    if not isinstance(shards, dict):
        return False

    expected = {
        "q_proj": (hidden, (attn_heads // tp_degree) * head_dim),
        "k_proj": (hidden, (kv_heads // tp_degree) * head_dim),
        "v_proj": (hidden, (kv_heads // tp_degree) * head_dim),
        "o_proj": ((attn_heads // tp_degree) * head_dim, hidden),
        "gate_up_proj": (hidden, (2 * inter) // tp_degree),
        "down_proj": (inter // tp_degree, hidden),
    }

    for name, expected_shape in expected.items():
        if name not in shards:
            return False
        got_shape = tuple(shards[name])
        if got_shape != expected_shape:
            return False

    return True
