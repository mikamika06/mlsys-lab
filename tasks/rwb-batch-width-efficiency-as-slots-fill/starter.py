def batch_width_utilization(occupancy: list[int], N: int) -> dict:
    """Per-step and mean batch-width utilization from an occupancy trace.

    Returns {"per_step": list[float] of shape (T,), "mean": float}.
    """
    raise NotImplementedError('your code here')
