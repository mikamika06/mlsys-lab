def crossover_seq_len(d_model: int, d_ff: int) -> int:
    """
    Return the smallest integer sequence length L such that the FLOPs of
    multi‑head self‑attention exceed those of the feed‑forward network.
    The formula reduces to ceil(d_ff / 2).
    """
    return (d_ff + 1) // 2
