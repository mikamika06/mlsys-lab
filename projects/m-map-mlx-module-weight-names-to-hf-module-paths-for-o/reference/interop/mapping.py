def map_mlx_block_weights(mlx_weights: dict, block_idx: int) -> dict:
    mapped = {}
    mapping_rules = {
        "attention.wq": f"model.layers.{block_idx}.self_attn.q_proj",
        "attention.wk": f"model.layers.{block_idx}.self_attn.k_proj",
        "attention.wv": f"model.layers.{block_idx}.self_attn.v_proj",
        "attention.wo": f"model.layers.{block_idx}.self_attn.o_proj",
        "feed_forward.w1": f"model.layers.{block_idx}.mlp.gate_proj",
        "feed_forward.w2": f"model.layers.{block_idx}.mlp.down_proj",
        "feed_forward.w3": f"model.layers.{block_idx}.mlp.up_proj",
        "attention_norm": f"model.layers.{block_idx}.input_layernorm",
        "ffn_norm": f"model.layers.{block_idx}.post_attention_layernorm",
    }
    for key, val in mlx_weights.items():
        clean_key = key
        for prefix in (f"model.layers.{block_idx}.", f"layers.{block_idx}.", f"{block_idx}."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        matched = False
        for prefix, hf_prefix in mapping_rules.items():
            if clean_key == prefix:
                mapped[hf_prefix] = val
                matched = True
                break
            elif clean_key.startswith(prefix + "."):
                suffix = clean_key[len(prefix):]
                mapped[hf_prefix + suffix] = val
                matched = True
                break
        if not matched:
            mapped[key] = val
    return mapped
