def layer_activation_memory(b, s, h, heads, dtype_bytes):
    attn_mem = b * s * h * 2 * dtype_bytes + b * heads * s * s * dtype_bytes
    mlp_mem = b * s * (4 * h) * dtype_bytes
    return attn_mem + mlp_mem
