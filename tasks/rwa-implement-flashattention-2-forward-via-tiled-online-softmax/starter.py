import math

def flash_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int=32) -> list[list[float]]:
    """FlashAttention-2-style forward pass: tiled online softmax.

    Must sweep Q, K, V in blocks of at most `block_size` rows using a
    running max / running normalizer, and must never materialize an
    (N, N) score matrix.
    """
    raise NotImplementedError('your code here')
