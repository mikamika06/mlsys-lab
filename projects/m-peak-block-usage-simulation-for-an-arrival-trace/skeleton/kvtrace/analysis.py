"""Analysis of KV cache fragmentation and waste metrics."""


def compute_paged_waste(length_histogram, block_size):
    """Compute total wasted token slots in paged allocation across a length histogram."""
    raise NotImplementedError


def compute_contiguous_waste(length_histogram, max_possible_len):
    """Compute wasted token slots assuming worst-case static contiguous reservation."""
    raise NotImplementedError


def compute_waste_ratio(length_histogram, block_size, max_possible_len):
    """Compute ratio of contiguous waste to paged waste."""
    raise NotImplementedError
