def compute_mae(a, b):
    """Compute Mean Absolute Error."""
    raise NotImplementedError


def compute_max_abs_diff(a, b):
    """Compute Maximum Absolute Difference."""
    raise NotImplementedError


def compute_rel_error(a, b):
    """Compute mean relative error: mean(|a - b| / (|b| + eps))."""
    raise NotImplementedError


def compute_snr(a, b):
    """Compute Signal-to-Noise Ratio in dB: 10 * log10(sum(b^2) / sum((a - b)^2))."""
    raise NotImplementedError


def compute_polygraphy_stats(a, b):
    """Compute full dict of Polygraphy error stats."""
    raise NotImplementedError
