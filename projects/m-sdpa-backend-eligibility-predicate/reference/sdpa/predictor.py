from sdpa.predicate import check_flash_eligible


def predict_backend(query_shape, key_shape, value_shape, dtype, is_causal, dropout_p, scale, has_mask):
    if check_flash_eligible(query_shape, key_shape, value_shape, dtype, is_causal, dropout_p, scale):
        if not has_mask:
            return "flash_attention"
    if dtype in ("torch.float16", "torch.bfloat16", "torch.float32"):
        d = query_shape[-1]
        if d <= 128:
            return "efficient_attention"
    return "math"
