import numpy as np


def block_diagonal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, seq_lens: list[int]) -> np.ndarray:
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
