import math

def ragged_causal_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], cu_seqlens: list[int]) -> list[list[float]]:
    """Causal self-attention over a PACKED (ragged) batch.

    Q, K, V: (n, d) -- multiple variable-length sequences concatenated along
    the token axis. cu_seqlens: 1-D int array of length (num_segments + 1)
    giving cumulative sequence boundaries.

    Row i may only attend to keys/values at position j such that j <= i AND
    j is in the same segment as i.

    BUG: this implementation applies a single GLOBAL causal mask (col <= row)
    over the whole packed batch and never looks at `cu_seqlens`. The first
    few tokens of segment i therefore still "see" the tail of segment i-1
    (and any earlier segments) -- attention leaks across sequence boundaries.
    """
    raise NotImplementedError('your code here')
