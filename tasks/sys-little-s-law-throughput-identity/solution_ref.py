def compute_throughput(concurrency: list[float],
                       latency: list[float]) -> list[float]:
    """
    Compute throughput from concurrency and mean latency using Little's Law.
    Parameters
    ----------
    concurrency : list[float]
        Average number of concurrent requests.
    latency : list[float]
        Mean service time in seconds.
    Returns
    -------
    list[float]
        Throughput values.
    """
    result = []
    for c, l in zip(concurrency, latency):
        result.append(c / l)
    return result
