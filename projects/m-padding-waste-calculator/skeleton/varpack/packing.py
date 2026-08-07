"""Length-aware sequence bin packing algorithms."""


def pack_sequences_ffd(seq_lens, max_bin_capacity):
    """Pack sequence lengths into bins using First-Fit Decreasing strategy."""
    raise NotImplementedError


def compute_packing_efficiency(bins, max_bin_capacity):
    """Compute ratio of useful tokens to total bin capacity across all bins."""
    raise NotImplementedError
