def measure_cache_and_attended(k: int, w: int, seq_len: int):
    """
    Return two lists of length `seq_len`:

    * cache_sizes[i] == min(k + w, i+1)
    * attended_pairs[i] == min(w, i)

    Parameters
    ----------
    k : int
        Base cache size.
    w : int
        Sliding‑window size for attention.
    seq_len : int
        Length of the sequence to process.

    Returns
    -------
    Tuple[List[int], List[int]]
        (cache_sizes, attended_pairs)
    """
    cache_sizes = [min(k + w, i) for i in range(1, seq_len + 1)]
    attended_pairs = [min(w, i - 1) for i in range(1, seq_len + 1)]
    return cache_sizes, attended_pairs
