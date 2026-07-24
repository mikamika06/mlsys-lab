def lm_head_flops(S: int, d: int, V: int) -> int:
    """
    Return the number of floating point operations for a forward pass through
    an LM head with weight tying.

    Parameters
    ----------
    S : int
        Sequence length (number of tokens).
    d : int
        Hidden dimension.
    V : int
        Vocabulary size.

    Returns
    -------
    int
        Total FLOPs = 2 * S * d * V.
    """
    return 2 * S * d * V
