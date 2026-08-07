import math

def tiled_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int) -> list[list[float]]:
    """
    Non-causal full attention O = softmax(Q K^T) V, computed by streaming
    K/V in blocks of `block_size` rows while maintaining a per-query running
    max `m`, running denominator `l`, and running numerator accumulator `acc`.

    BUG: when a new block raises the running max, `m` is updated but the
    existing `l` and `acc` accumulators (which were scaled relative to the
    OLD max) are never rescaled onto the new max before more terms are
    added. This silently corrupts the result whenever a later block
    contains a larger score than all previous blocks.
    """
    raise NotImplementedError('your code here')
