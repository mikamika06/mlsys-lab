def predict_backend(config):
    """Predict PyTorch SDPA backend dispatch choice."""
    dtype = config.get("dtype", "float32")
    head_dim = config.get("head_dim")
    has_mask = config.get("has_mask", False)
    mask_type = config.get("mask_type", None)
    is_causal = config.get("is_causal", False)
    dropout_p = config.get("dropout_p", 0.0)
    q_len = config.get("q_len")
    kv_len = config.get("kv_len")

    flash_ok = True
    if dtype not in ("float16", "bfloat16"):
        flash_ok = False
    if head_dim is None or head_dim > 256 or head_dim % 8 != 0:
        flash_ok = False
    if has_mask:
        if not is_causal or mask_type is not None:
            flash_ok = False

    if flash_ok:
        return "flash"

    mem_eff_ok = True
    if head_dim is None or head_dim > 1024 or head_dim % 4 != 0:
        mem_eff_ok = False

    if mem_eff_ok:
        return "mem_efficient"

    return "math"
