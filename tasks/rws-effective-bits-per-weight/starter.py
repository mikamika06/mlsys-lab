import numpy as np


def effective_bits_per_weight(N: int, M: int, bits: int, group_size: int, scale_bits: float = 16.0) -> float:
    """Effective bits/weight of N:M structured sparsity + quantization + metadata.

    N, M: N:M sparsity pattern (keep N of every M, 1 <= N <= M).
    bits: quantizer bit width applied to each kept weight.
    group_size: number of original (dense) weight positions sharing one
        stored scale.
    scale_bits: bits per stored scale value (default 16.0, fp16).

    density = N / M
    index_bits = density * ceil(log2(M))          -- position index overhead
    scale_overhead = scale_bits / group_size       -- scale storage overhead
    bpw = density * bits + index_bits + scale_overhead
    """
    raise NotImplementedError('your code here')
