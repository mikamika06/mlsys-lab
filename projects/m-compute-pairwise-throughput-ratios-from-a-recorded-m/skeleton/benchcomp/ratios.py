def compute_pairwise_ratios(records):
    """Compute mean throughput ratios across matched configs for framework pairs."""
    raise NotImplementedError


def rank_frameworks(records):
    """Rank frameworks by descending throughput and ascending peak VRAM."""
    raise NotImplementedError
