import math

def tiled_causal_attention(Q, K, V, block_size):
    """FlashAttention-style tiled causal self-attention with online softmax.

    Q, K, V: (n, d). block_size: tile edge length along the sequence axis.

    Never materialize the full (n, n) score matrix. Sweep query tiles
    against key/value tiles, maintaining running (max, sum, weighted-output)
    online-softmax statistics. Apply causal masking at tile granularity:
    skip key tiles entirely to the right of the query tile, apply an
    elementwise lower-triangular mask on the diagonal tile, and leave key
    tiles to the left of the query tile unmasked. Returns (n, d).
    """
    raise NotImplementedError('your code here')
