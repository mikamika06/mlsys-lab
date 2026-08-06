def by_depth(rows, model=None, want="decode"):
    raise NotImplementedError


def decay_table(rows, model=None, want="decode"):
    """Per depth: tokens_per_second (median of the samples, not the reported
    average), relative_to_empty, loss_fraction, separable_from_empty."""
    raise NotImplementedError


def slope_per_1k(table):
    """Least-squares tokens/s lost per 1024 tokens of context."""
    raise NotImplementedError


def extrapolate(table, depth):
    raise NotImplementedError
