import numpy as np


def attention_flops(lens: np.ndarray, head_dim: int, num_heads: int) -> tuple[int, int]:
    """
    lens      : 1-D int array, per-sequence token counts in a prefill batch.
    head_dim  : dimension per attention head.
    num_heads : number of attention heads.

    Using a fixed per-(query,key)-pair FLOP constant

        C = 4 * head_dim * num_heads

    (2 FLOPs/dim for the QK^T score, 2 FLOPs/dim for the softmax-weighted V
    accumulation, per head), return:

      packed_flops -- RAGGED / varlen attention: each sequence only ever
        computes its own (len_i x len_i) attention matrix, so total work is
        proportional to sum(len_i^2).

      padded_flops -- naive DENSE batching: every sequence is padded out to
        the batch's longest sequence, so the kernel computes a full
        (batch x max_len x max_len) grid of pairs for every sequence,
        including pairs that touch padding tokens.

    Returns (packed_flops, padded_flops) as plain Python ints.
    """
    lens_list = [int(v) for v in np.asarray(lens).tolist()]
    c = 4 * int(head_dim) * int(num_heads)

    packed_flops = c * sum(n * n for n in lens_list)

    batch = len(lens_list)
    max_len = max(lens_list)
    padded_flops = c * batch * max_len * max_len

    return packed_flops, padded_flops
