import math

CONFIGS = [
    {"batch_size": 2, "num_heads": 8, "seq_len": 1024, "head_dim": 64},
    {"batch_size": 4, "num_heads": 16, "seq_len": 2048, "head_dim": 128},
    {"batch_size": 1, "num_heads": 32, "seq_len": 4096, "head_dim": 64},
]

def estimate_math_peak_memory(b, h, s, d, dtype_bytes=2):
    qkv = 3 * b * h * s * d * dtype_bytes
    attn_weights = b * h * s * s * dtype_bytes
    output = b * h * s * d * dtype_bytes
    return qkv + attn_weights + output

def estimate_efficient_peak_memory(b, h, s, d, dtype_bytes=2):
    qkv = 3 * b * h * s * d * dtype_bytes
    output = b * h * s * d * dtype_bytes
    block_scratch = b * h * s * 64 * dtype_bytes
    return qkv + output + block_scratch

def estimate_hbm_traffic(b, h, s, d, mode="math", dtype_bytes=2):
    qkv = 3 * b * h * s * d * dtype_bytes
    output = b * h * s * d * dtype_bytes
    if mode == "math":
        attn_weights_rw = 2 * b * h * s * s * dtype_bytes
        return qkv + output + attn_weights_rw
    else:
        return qkv + output

def validate_mask_dispatch(is_causal, mask_tensor_present):
    if is_causal and mask_tensor_present:
        return "conflict"
    if is_causal:
        return "flash_causal"
    if mask_tensor_present:
        return "math_masked"
    return "flash_standard"
