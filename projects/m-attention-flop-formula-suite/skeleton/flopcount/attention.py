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
    raise NotImplementedError
