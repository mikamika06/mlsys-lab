def predict_backend(head_dim, dtype, has_custom_mask):
    if dtype in ("float16", "bfloat16") and head_dim in (16, 32, 64, 128, 256) and not has_custom_mask:
        return "flash"
    if dtype in ("float16", "bfloat16", "float32") and head_dim > 0 and head_dim % 8 == 0 and head_dim <= 128 and not has_custom_mask:
        return "mem_efficient"
    return "math"
