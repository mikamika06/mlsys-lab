def compute_effective_context(config: dict) -> int:
    base_len = int(config.get("max_position_embeddings", 2048))
    rope = config.get("rope_scaling")
    effective = base_len
    if isinstance(rope, dict):
        if "factor" in rope and rope["factor"] is not None:
            effective = int(base_len * float(rope["factor"]))
        elif "original_max_position_embeddings" in rope and rope["original_max_position_embeddings"] is not None:
            orig = int(rope["original_max_position_embeddings"])
            if orig > 0 and base_len <= orig:
                effective = orig
            else:
                effective = base_len
    if "override_max_model_len" in config and config["override_max_model_len"] is not None:
        effective = min(effective, int(config["override_max_model_len"]))
    elif "max_model_len" in config and config["max_model_len"] is not None:
        effective = min(effective, int(config["max_model_len"]))
    return max(1, effective)
