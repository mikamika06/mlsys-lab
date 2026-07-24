import numpy as np

def compute_throughput(concurrency: np.ndarray,
                       latency: np.ndarray) -> np.ndarray:
    """
    Compute throughput from concurrency and mean latency using Little's Law.
    Parameters
    ----------
    concurrency : np.ndarray
        Average number of concurrent requests (shape (n,)).
    latency : np.ndarray
        Mean service time in seconds (shape (n,)).
    Returns
    -------
    np.ndarray
        Throughput values (float64) with shape (n,).
    """
    return np.asarray(concurrency, dtype=np.float64) / np.asarray(latency, dtype=np.float64)
