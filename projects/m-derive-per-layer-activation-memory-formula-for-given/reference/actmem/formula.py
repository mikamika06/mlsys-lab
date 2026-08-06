def compute_layer_activation_bytes(b: int, s: int, h: int, heads: int, dtype_bytes: int) -> int:
    attn_act = b * s * h * dtype_bytes * 2
    qkv_act = b * s * h * dtype_bytes * 3
    mlp_act = b * s * h * dtype_bytes * 4
    return attn_act + qkv_act + mlp_act
