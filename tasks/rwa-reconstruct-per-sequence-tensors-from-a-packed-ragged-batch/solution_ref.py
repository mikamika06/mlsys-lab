def unpack_sequences(packed, cu_seqlens):
    """
    Reconstruct per‑sequence tensors from a packed ragged batch.

    Parameters
    ----------
    packed : np.ndarray
        Concatenated tensor of shape (N, D).
    cu_seqlens : np.ndarray
        Cumulative sequence lengths of shape (S+1,).

    Returns
    -------
    List[np.ndarray]
        A list of S arrays each of shape (L_i, D) corresponding to the original sequences.
    """
    return [packed[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
