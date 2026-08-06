import math
import numpy as np


def effective_bits_per_weight(N: int, M: int, bits: int, group_size: int, scale_bits: float = 16.0) -> float:
    """Effective bits/weight of N:M structured sparsity + quantization + metadata.

    bpw = density*bits + density*ceil(log2(M)) + scale_bits/group_size,
    density = N / M. See task.md for the derivation.
    """
    density = N / M
    index_bits = density * float(math.ceil(math.log2(M)))
    scale_overhead = scale_bits / group_size
    return density * bits + index_bits + scale_overhead
