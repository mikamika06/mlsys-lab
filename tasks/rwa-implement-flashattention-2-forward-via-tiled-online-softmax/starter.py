import numpy as np


def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int = 32) -> np.ndarray:
    """FlashAttention-2-style forward pass: tiled online softmax.

    Must sweep Q, K, V in blocks of at most `block_size` rows using a
    running max / running normalizer, and must never materialize an
    (N, N) score matrix.
    """
    raise NotImplementedError('your code here')
