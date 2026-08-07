import math

def tiled_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int) -> list[list[float]]:
    """Tiled (flash-style) attention forward.

    Sweep K/V in blocks of at most `block_size` rows (the last block may be
    smaller if `block_size` doesn't evenly divide the sequence length).
    Maintain, per query row, a running max `m` and running normalizer `l`
    (online softmax): whenever a new block raises the running max, rescale
    the accumulated output and normalizer by exp(m_old - m_new) before
    folding in the new block's contribution. The final output must be
    identical (up to floating-point rounding) to dense
    softmax(Q @ K.T / sqrt(d)) @ V, for ANY block_size -- including
    block_size == 1 and block_size >= N.

    Q, K, V: (N, d) arrays. Returns (N, d) float64 array.
    """
    raise NotImplementedError('your code here')
