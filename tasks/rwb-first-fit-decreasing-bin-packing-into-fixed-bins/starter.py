def pack_into_fixed_bins(lengths, bin_size):
    """
    Pack sequences into fixed-capacity bins using first-fit decreasing.

    Parameters
    ----------
    lengths : list or numpy.ndarray of int
        Sequence lengths to pack.
    bin_size : int
        Maximum total length per bin.

    Returns
    -------
    tuple (int, list[int])
        (num_bins, assignments) where assignments[i] is the bin index (0-based)
        assigned to the i-th input sequence in original order.
    """
    raise NotImplementedError('your code here')
