def find_attention_mlp_crossover(b, h, heads, dtype_bytes):
    for s in range(1, 100000):
        attn = b * s * h * 2 * dtype_bytes + b * heads * s * s * dtype_bytes
        mlp = b * s * (4 * h) * dtype_bytes
        if attn > mlp:
            return s
    return 100000
