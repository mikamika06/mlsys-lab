"""Optimization profile splitting and evaluation."""

def split_wide_profile(wide_profile):
    """Split wide profile into low and high profiles."""
    raise NotImplementedError

def evaluate_profile_latency(profiles, query_shapes, cost_fn):
    """Evaluate latency over query shapes using optimal profile selection."""
    raise NotImplementedError
