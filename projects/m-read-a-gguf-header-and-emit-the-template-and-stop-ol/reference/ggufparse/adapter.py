def validate_adapter(base_cfg, adapter_cfg):
    """Validates adapter compatibility with base config."""
    if base_cfg.get("hidden_size") != adapter_cfg.get("hidden_size"):
        return False
    if base_cfg.get("num_attention_heads") != adapter_cfg.get("num_attention_heads"):
        return False
    return True
