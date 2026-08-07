def unpack_sequences(packed, cu_seqlens):
    """
    Reconstruct per‑sequence lists from a packed ragged batch.

    Parameters
    ----------
    packed : list[list[float]]
        Concatenated structure of shape (N, D).
    cu_seqlens : list[int]
        Cumulative sequence lengths of shape (S+1,).

    Returns
    -------
    list[list[list[float]]]
        A list of S lists each of shape (L_i, D) corresponding to the original sequences.
    """
    return [packed[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
