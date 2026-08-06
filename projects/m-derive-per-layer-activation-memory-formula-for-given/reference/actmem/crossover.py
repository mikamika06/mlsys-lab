def find_attention_mlp_crossover(h: int, heads: int, intermediate_size: int) -> int:
    for s in range(1, 131072):
        attn_mem = s * s + 2 * s * h
        mlp_mem = 3 * s * intermediate_size
        if attn_mem > mlp_mem:
            return s
    return 131072
