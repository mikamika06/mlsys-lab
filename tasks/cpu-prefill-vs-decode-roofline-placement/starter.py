def roofline_phase_classify(batch_sizes, seq_lengths):
    """Return 'memory-bound' or 'compute-bound' for each (batch, seq) pair."""
    raise NotImplementedError("implement the analytical roofline classification")
