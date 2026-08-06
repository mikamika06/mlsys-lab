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
    steps = trace.shape[0]
    slots = trace.shape[1]

    tokens_per_step = np.zeros(steps, dtype=np.float64)
    for i in range(steps):
        row_sum = 0.0
        for j in range(slots):
            row_sum += trace[i, j]
        tokens_per_step[i] = row_sum

    agg_sum = 0.0
    single_count = 0.0
    for i in range(steps):
        val = tokens_per_step[i]
        agg_sum += val
        if val > 0:
            single_count += 1.0

    agg = agg_sum / steps if steps > 0 else 0.0
    single = single_count / steps if steps > 0 else 0.0

    return np.array([agg, single], dtype=np.float64)
