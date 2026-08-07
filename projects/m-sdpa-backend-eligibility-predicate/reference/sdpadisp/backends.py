def check_flash_attention(query_shape, key_shape, value_shape, dtype, is_causal, scale):
    if dtype not in ("float16", "bfloat16"):
        return False
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False
    B, H, N_Q, D = query_shape
    _, H_K, N_K, D_K = key_shape
    _, H_V, N_V, D_V = value_shape
    if D != D_K or D != D_V or D > 256:
        return False
    if D % 8 != 0:
        return False
    if H % H_K != 0 or H_V != H_K:
        return False
    if N_Q < 1 or N_K < 1:
        return False
    return True


def check_mem_efficient_attention(query_shape, key_shape, value_shape, dtype, is_causal, scale):
    if dtype not in ("float16", "bfloat16", "float32"):
        return False
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False
    B, H, N_Q, D = query_shape
    _, H_K, N_K, D_K = key_shape
    _, H_V, N_V, D_V = value_shape
    if D != D_K or D != D_V:
        return False
    if H % H_K != 0 or H_V != H_K:
        return False
    if is_causal and D > 128:
        return False
    return True
