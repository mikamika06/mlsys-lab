def check_tp_sharding(model_config, tp_size):
    """Check if model_config can be legally sharded across tp_size GPUs."""
    if tp_size <= 0:
        return {"valid": False, "reason": "tp_size must be positive"}

    if tp_size == 1:
        return {"valid": True, "reason": "Single GPU execution is always valid"}

    num_heads = model_config.get("num_attention_heads", 0)
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    hidden_size = model_config.get("hidden_size", 0)
    intermediate_size = model_config.get("intermediate_size", 0)

    if num_heads % tp_size != 0:
        return {
            "valid": False,
            "reason": f"num_attention_heads ({num_heads}) not divisible by tp_size ({tp_size})"
        }

    if num_kv_heads < tp_size and tp_size % num_kv_heads != 0:
        return {
            "valid": False,
            "reason": f"tp_size ({tp_size}) must be divisible by num_key_value_heads ({num_kv_heads}) when tp_size > num_kv_heads"
        }

    if (hidden_size // num_heads) * num_heads != hidden_size:
        return {
            "valid": False,
            "reason": "hidden_size not divisible by num_attention_heads"
        }

    if intermediate_size % tp_size != 0:
        return {
            "valid": False,
            "reason": f"intermediate_size ({intermediate_size}) not divisible by tp_size ({tp_size})"
        }

    return {"valid": True, "reason": "TP configuration is valid"}
