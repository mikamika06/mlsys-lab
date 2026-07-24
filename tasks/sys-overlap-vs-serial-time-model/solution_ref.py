import numpy as np

def overlap_time(compute_times: np.ndarray,
                 comm_times: np.ndarray) -> float:
    """
    Compute the overlapped time of compute and communication phases.

    Parameters
    ----------
    compute_times : np.ndarray
        1‑D array of per‑step compute times.
    comm_times : np.ndarray
        1‑D array of per‑step communication times.

    Returns
    -------
    float
        The total overlapped time, equal to max(sum(compute), sum(comm)).
    """
    comp_sum = np.sum(np.asarray(compute_times, dtype=np.float64))
    comm_sum = np.sum(np.asarray(comm_times, dtype=np.float64))
    return float(np.maximum(comp_sum, comm_sum))
