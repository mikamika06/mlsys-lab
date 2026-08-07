import math

def block_diagonal_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], seq_lens: list[int]) -> list[list[float]]:
    """Attention over several variable-length sequences PACKED into one
    (N, d) tensor along the row axis (xformers' BlockDiagonalMask). Each
    sequence attends only to its own rows -- full (non-causal) attention
    within a sequence, zero cross-sequence attention.

    Q, K, V   : (N, d), N == sum(seq_lens).
    seq_lens  : list of positive ints, the length of each packed sequence,
                in order.

    Returns (N, d).
    """
    raise NotImplementedError('your code here')
