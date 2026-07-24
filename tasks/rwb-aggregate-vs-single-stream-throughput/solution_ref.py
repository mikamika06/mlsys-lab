import numpy as np

def throughput(trace: np.ndarray) -> np.ndarray:
    """
    Compute aggregate and single‑stream throughput from a binary occupancy trace.

    Parameters
    ----------
    trace : np.ndarray
        2‑D array of shape (steps, slots) with entries 0 or 1.

    Returns
    -------
    np.ndarray
        Array([aggregate_throughput, single_stream_rate]) as float64.
    """
    tokens_per_step = trace.sum(axis=1)
    agg = tokens_per_step.mean()
    single = (tokens_per_step > 0).mean()
    return np.array([agg, single], dtype=np.float64)
