import math

def packed_attention_with_reset_mask(Q: list[list[float]], K: list[list[float]], V: list[list[float]], segment_ids: list[int]) -> list[list[float]]:
    """Causal self-attention over multiple documents PACKED into one
    training sequence, with the mask meant to RESET at every segment
    boundary. Q, K, V: (n, d). segment_ids: (n,) int array giving each
    token's segment/document index.

    BUG: this implementation applies a single GLOBAL causal mask
    (col <= row) over the whole packed sequence and never looks at
    `segment_ids`. Every document therefore still attends into the TAIL of
    the previous document (and any earlier ones) as if it were its own
    left context.
    """
    raise NotImplementedError('your code here')
