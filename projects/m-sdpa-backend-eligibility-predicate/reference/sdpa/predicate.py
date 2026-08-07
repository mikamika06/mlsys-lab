def check_flash_eligible(query_shape, key_shape, value_shape, dtype, is_causal, dropout_p, scale):
    if dtype not in ("torch.float16", "torch.bfloat16"):
        return False
    if dropout_p > 0.0:
        return False
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False
    b_q, h_q, s_q, d_q = query_shape
    b_k, h_k, s_k, d_k = key_shape
    b_v, h_v, s_v, d_v = value_shape
    if b_q != b_k or b_q != b_v:
        return False
    if d_q != d_k or d_q != d_v:
        return False
    if d_q not in (16, 32, 64, 128, 256):
        return False
    if h_q % h_k != 0:
        return False
    if s_q < 1 or s_k < 1:
        return False
    return True
