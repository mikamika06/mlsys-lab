def compute_folded_weight_blob_size(params):
    """
    Compute the total number of bytes that would be stored in a compiled .pt2 weight‑blob.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Mapping from tensor names to NumPy arrays representing parameters or constants.

    Returns
    -------
    int
        Sum over all tensors of (number of elements × element byte width).
    """
    total = 0
    for arr in params.values():
        total += arr.size * arr.dtype.itemsize
    return int(total)
