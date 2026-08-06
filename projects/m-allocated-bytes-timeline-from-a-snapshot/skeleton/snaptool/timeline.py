def build_timeline(snapshot):
    """
    Reconstructs the timeline of active allocated bytes from a PyTorch memory snapshot.
    Returns (timeline_events, peak_allocated_bytes) where timeline_events is a list of
    dict(time=int, allocated_bytes=int).
    """
    raise NotImplementedError
