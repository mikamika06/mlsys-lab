def compare_footprint(snapshot):
    """
    Calculates the theoretical model memory footprint based on parameters and optimizer state,
    and compares it against the peak allocation observed in the snapshot timeline.
    Returns (theoretical_bytes, overhead_bytes).
    """
    raise NotImplementedError
