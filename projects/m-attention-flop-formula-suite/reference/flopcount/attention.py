def count_attention_flops(
    b: int,
    h_q: int,
    h_kv: int,
    s_q: int,
    s_k: int,
    d: int,
    causal: bool = False,
) -> int:
    """Calculates GEMM FLOPs for multi-head or grouped-query attention core."""
    if not causal:
        valid_pairs = s_q * s_k
    else:
        valid_pairs = 0
        offset = s_k - s_q
        for i in range(s_q):
            max_j = i + offset
            if max_j >= 0:
                count = min(s_k, max_j + 1)
                valid_pairs += max(0, count)
    return 4 * b * h_q * valid_pairs * d
