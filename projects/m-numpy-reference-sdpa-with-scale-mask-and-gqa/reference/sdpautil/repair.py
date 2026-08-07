def repair_flash_config(config):
    """Repair an incompatible SDPA config to enable FlashAttention."""
    repaired = dict(config)

    dtype = repaired.get("dtype", "float32")
    if dtype not in ("float16", "bfloat16"):
        repaired["dtype"] = "float16"

    head_dim = repaired.get("head_dim", 64)
    if head_dim is None or head_dim > 256 or head_dim % 8 != 0:
        rem = head_dim % 8 if head_dim else 0
        if rem == 0 and head_dim > 256:
            repaired["head_dim"] = 256
        elif rem != 0:
            repaired["head_dim"] = min(256, max(8, ((head_dim + 7) // 8) * 8))
        else:
            repaired["head_dim"] = 64

    if repaired.get("has_mask", False):
        if repaired.get("mask_type") is not None:
            repaired["mask_type"] = None
        repaired["is_causal"] = True

    return repaired
