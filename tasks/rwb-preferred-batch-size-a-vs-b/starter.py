def compare_preferred_batch_sizes(arrivals: list[float], cap: int, max_queue_delay: float, batch_time: float, preferred_a: int, preferred_b: int) -> list[float]:
    """
    arrivals: 1-D sorted array of request arrival timestamps.
    cap: max_batch_size, shared by both configurations.
    max_queue_delay: max time (seconds) a queue waits before a forced
        dispatch, shared by both configurations.
    batch_time: fixed processing duration (seconds) of one dispatched
        batch, regardless of its size, shared by both configurations.
    preferred_a, preferred_b: the two preferred_batch_size values to
        compare (same cap/delay/batch_time otherwise).

    Simulate a single-server dynamic batcher once per configuration and
    return (mean_latency_a, throughput_a, mean_latency_b, throughput_b).
    """
    raise NotImplementedError('your code here')
