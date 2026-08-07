import math

def varlen_block_diagonal_attention(q: list[list[float]], k: list[list[float]], v: list[list[float]], cu_seqlens: list[int]) -> list[list[float]]:
    """Packed varlen self-attention: several sequences packed row-wise
    into (N, d) tensors, boundaries given by `cu_seqlens` (the standard
    FlashAttention-varlen / xformers convention). Each token attends only
    within its own segment (full, non-causal attention inside a segment,
    none across segments).

    q, k, v    : (N, d).
    cu_seqlens : (n_seqs + 1,) int, cu_seqlens[0] == 0,
                 cu_seqlens[-1] == N. Sequence i occupies rows
                 cu_seqlens[i] : cu_seqlens[i+1].

    Returns (N, d).
    """
    raise NotImplementedError('your code here')
