import math

def tiled_online_softmax_attention(q: list[float], K: list[list[float]], V: list[list[float]], block_size: int) -> list[float]:
    """FlashAttention-style single-query forward pass: stream over K/V in
    blocks of `block_size`, maintaining a running max `m`, running
    normalizer `l`, and an UNNORMALIZED output accumulator `O`. Returns
    O / l, shape (d,).

    BUG: this rescales `l` whenever the running max grows, but forgets to
    rescale `O` the same way, so earlier blocks stay weighted against
    their own stale local max instead of the final global max.
    """
    raise NotImplementedError('your code here')
