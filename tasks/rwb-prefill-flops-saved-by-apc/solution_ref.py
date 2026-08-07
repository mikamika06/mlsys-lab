def flops_saved_by_apc(reused_counts: list[int], per_token_flop: float) -> float:
    """
    Compute the total FLOPs saved by APC.

    Parameters
    ----------
    reused_counts : list[int]
        List of non‑negative integers representing how many times each token was reused.
    per_token_flop : float
        FLOPs saved per reuse of a single token.

    Returns
    -------
    float
        Total FLOPs saved.
    """
    total_reuses = 0
    for count in reused_counts:
        total_reuses += count
    return float(total_reuses * per_token_flop)
