from __future__ import annotations

def pack_into_fixed_bins(lengths, bin_size):
    """First-fit decreasing bin packing.

    Parameters
    ----------
    lengths : list or numpy.ndarray of positive ints
        Sequence lengths to pack.
    bin_size : int
        Capacity of each bin.

    Returns
    -------
    tuple (int, list[int])
        (num_bins, assignments) where assignments[i] is the 0-based bin
        assigned to the i-th input sequence (original order).
    """
    seq = list(lengths)
    n = len(seq)
    # Stable descending sort: ties keep original order
    indexed = sorted(enumerate(seq), key=lambda x: x[1], reverse=True)
    bins = []               # remaining capacity per bin
    assignment = [0] * n

    for orig_idx, l in indexed:
        placed = False
        for b_idx, cap in enumerate(bins):
            if cap >= l:
                bins[b_idx] = cap - l
                assignment[orig_idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append(bin_size - l)
            assignment[orig_idx] = len(bins) - 1

    return len(bins), assignment
