"""Length-aware sequence bin packing algorithms."""


def pack_sequences_ffd(seq_lens, max_bin_capacity):
    """Pack sequence lengths into bins using First-Fit Decreasing strategy."""
    indexed = sorted(enumerate(seq_lens), key=lambda x: x[1], reverse=True)
    bins = []
    for idx, length in indexed:
        if length > max_bin_capacity:
            raise ValueError(f"Sequence length {length} exceeds max capacity {max_bin_capacity}")
        placed = False
        for b in bins:
            if sum(b["lengths"]) + length <= max_bin_capacity:
                b["indices"].append(idx)
                b["lengths"].append(length)
                placed = True
                break
        if not placed:
            bins.append({"indices": [idx], "lengths": [length]})
    return bins


def compute_packing_efficiency(bins, max_bin_capacity):
    """Compute ratio of useful tokens to total bin capacity across all bins."""
    if not bins:
        return 0.0
    total_used = sum(sum(b["lengths"]) for b in bins)
    total_capacity = len(bins) * max_bin_capacity
    return float(total_used / total_capacity)
