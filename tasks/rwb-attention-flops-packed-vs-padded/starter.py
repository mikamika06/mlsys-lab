import numpy as np


def attention_flops(lens: np.ndarray, head_dim: int, num_heads: int) -> tuple[int, int]:
    """
    lens      : 1-D int array, per-sequence token counts in a prefill batch.
    head_dim  : dimension per attention head.
    num_heads : number of attention heads.

    Using the per-(query,key)-pair FLOP constant C = 4 * head_dim * num_heads,
    return (packed_flops, padded_flops):
      packed_flops = C * sum(len_i^2)
      padded_flops = C * batch_size * max(len)^2
    as plain Python ints.
    """
    raise NotImplementedError('your code here')
