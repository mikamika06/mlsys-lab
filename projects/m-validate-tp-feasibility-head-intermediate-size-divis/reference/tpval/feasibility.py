def validate_tp_feasibility(config: dict, tp_size: int) -> dict:
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]

    heads_ok = (num_heads % tp_size == 0)
    kv_heads_ok = (num_kv_heads % tp_size == 0)
    intermediate_ok = (intermediate_size % tp_size == 0)

    is_feasible = heads_ok and kv_heads_ok and intermediate_ok
    reasons = []
    if not heads_ok:
        reasons.append(f"num_attention_heads ({num_heads}) not divisible by tp_size ({tp_size})")
    if not kv_heads_ok:
        reasons.append(f"num_key_value_heads ({num_kv_heads}) not divisible by tp_size ({tp_size})")
    if not intermediate_ok:
        reasons.append(f"intermediate_size ({intermediate_size}) not divisible by tp_size ({tp_size})")

    return {
        "is_feasible": is_feasible,
        "reasons": reasons
    }
