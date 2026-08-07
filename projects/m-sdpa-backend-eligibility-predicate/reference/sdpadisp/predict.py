from sdpadisp.backends import check_flash_attention, check_mem_efficient_attention


def predict_backend(query_shape, key_shape, value_shape, dtype, is_causal, scale=None):
    if check_flash_attention(query_shape, key_shape, value_shape, dtype, is_causal, scale):
        return "flash_attention"
    if check_mem_efficient_attention(query_shape, key_shape, value_shape, dtype, is_causal, scale):
        return "mem_efficient_attention"
    return "math"
