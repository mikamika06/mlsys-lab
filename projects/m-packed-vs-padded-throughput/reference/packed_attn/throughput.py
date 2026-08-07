def padded_flops(
    batch_seqlens: list[int],
    num_heads: int,
    head_dim: int,
    max_seqlen: int | None = None,
) -> int:
    if not batch_seqlens:
        return 0
    actual_max = max(batch_seqlens)
    s_used = actual_max if max_seqlen is None else max_seqlen
    if s_used < actual_max:
        raise ValueError("max_seqlen cannot be less than max actual sequence length")
    return 4 * len(batch_seqlens) * (s_used**2) * num_heads * head_dim


def packed_flops(
    batch_seqlens: list[int], num_heads: int, head_dim: int
) -> int:
    if not batch_seqlens:
        return 0
    sum_sq = sum(length**2 for length in batch_seqlens)
    return 4 * sum_sq * num_heads * head_dim


def flops_ratio(
    batch_seqlens: list[int],
    num_heads: int,
    head_dim: int,
    max_seqlen: int | None = None,
) -> float:
    p_flops = padded_flops(batch_seqlens, num_heads, head_dim, max_seqlen)
    k_flops = packed_flops(batch_seqlens, num_heads, head_dim)
    if k_flops == 0:
        return 0.0
    return float(p_flops) / float(k_flops)


def memory_bytes(
    batch_seqlens: list[int],
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    max_seqlen: int | None = None,
) -> dict[str, int]:
    if not batch_seqlens:
        return {"padded_bytes": 0, "packed_bytes": 0}
    actual_max = max(batch_seqlens)
    s_used = actual_max if max_seqlen is None else max_seqlen
    if s_used < actual_max:
        raise ValueError("max_seqlen cannot be less than max actual sequence length")
    b = len(batch_seqlens)
    padded = b * s_used * num_heads * head_dim * dtype_bytes
    packed = sum(batch_seqlens) * num_heads * head_dim * dtype_bytes
    return {"padded_bytes": padded, "packed_bytes": packed}
