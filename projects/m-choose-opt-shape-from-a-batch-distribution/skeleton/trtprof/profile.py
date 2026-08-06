def select_opt_shape(batch_samples, strategy="p50"):
    """Select min, opt, max dynamic shape bounds from batch observations."""
    raise NotImplementedError


def calculate_profile_bounds(batch_samples, strategy="p50", padding_ratio=0.1):
    """Calculate (min_shape, opt_shape, max_shape) for a tensor dimension."""
    raise NotImplementedError
