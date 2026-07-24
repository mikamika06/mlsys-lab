def ring_allreduce_bytes_moved(n_ranks: int, data_bytes: int) -> float:
    """Return per-rank send volume in bytes for a ring all-reduce.

    Formula: 2 * (N-1) / N * B
    """
    return 2.0 * (n_ranks - 1) / n_ranks * data_bytes
