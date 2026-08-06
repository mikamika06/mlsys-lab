def compute_units_latency(graph_profile, compute_units):
    """Calculate aggregate execution latency for a given compute unit configuration."""
    raise NotImplementedError


def evaluate_all_units(graph_profile):
    """Evaluate pipeline latency across all four MLComputeUnits settings."""
    raise NotImplementedError
