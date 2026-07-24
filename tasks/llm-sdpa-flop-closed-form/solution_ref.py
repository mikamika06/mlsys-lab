def sdpa_flop_closed_form(S: int, d: int) -> int:
    """
    Return the exact number of floating‑point operations required to compute
    scaled dot‑product attention for a sequence length S and feature dimension d.
    
    The formula is derived from two matrix multiplications each costing
    2 * S^2 * d FLOPs, giving a total of 4 * S^2 * d.
    """
    return 4 * S * S * d
