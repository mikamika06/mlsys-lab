import numpy as np


def _simulate(arrivals: np.ndarray, P: int, cap: int, D: float, T: float):
    """Single-server dynamic-batcher event simulation.

    Each cycle, once the server is free and at least one request is
    queued, it dispatches as soon as EITHER trigger fires (whichever
    comes first):
      - size trigger: the queue already holds >= P requests,
      - delay trigger: the oldest queued request has waited D seconds.
    The dispatched batch takes min(queue length at trigger time, cap)
    requests (FIFO) and takes a fixed T seconds to finish, during which
    the server is unavailable for the next batch.

    Returns (mean_latency, throughput).
    """
    arrivals = np.asarray(arrivals, dtype=np.float64)
    n = len(arrivals)
    pos = 0
    server_free = 0.0
    lat_sum = 0.0
    count = 0
    last_finish = 0.0

    while pos < n:
        start_wait = max(server_free, arrivals[pos])
        j = pos
        while j < n and arrivals[j] <= start_wait:
            j += 1
        qcount = j - pos

        if qcount >= P:
            dispatch_time = start_wait
            take = min(qcount, cap)
        else:
            deadline = arrivals[pos] + D
            if pos + P - 1 < n and arrivals[pos + P - 1] <= deadline:
                dispatch_time = max(arrivals[pos + P - 1], start_wait)
                take = P
            else:
                dispatch_time = deadline
                k = pos
                while k < n and arrivals[k] <= deadline:
                    k += 1
                take = min(k - pos, cap)

        batch_end = pos + take
        finish_time = dispatch_time + T
        lat_sum += float(np.sum(finish_time - arrivals[pos:batch_end]))
        count += take
        last_finish = finish_time
        server_free = finish_time
        pos = batch_end

    mean_latency = lat_sum / count
    makespan = last_finish - arrivals[0]
    throughput = count / makespan
    return mean_latency, throughput


def compare_preferred_batch_sizes(
    arrivals: np.ndarray,
    cap: int,
    max_queue_delay: float,
    batch_time: float,
    preferred_a: int,
    preferred_b: int,
):
    """
    arrivals: 1-D sorted array of request arrival timestamps.
    cap: max_batch_size, shared by both configurations.
    max_queue_delay: max time (seconds) a queue waits before a forced
        dispatch, shared by both configurations.
    batch_time: fixed processing duration (seconds) of one dispatched
        batch, regardless of its size, shared by both configurations.
    preferred_a, preferred_b: the two preferred_batch_size values to
        compare (same cap/delay/batch_time otherwise).

    Simulate the single-server dynamic batcher once per configuration
    and return (mean_latency_a, throughput_a, mean_latency_b, throughput_b).
    """
    ml_a, tp_a = _simulate(arrivals, preferred_a, cap, max_queue_delay, batch_time)
    ml_b, tp_b = _simulate(arrivals, preferred_b, cap, max_queue_delay, batch_time)
    return np.array([ml_a, tp_a, ml_b, tp_b])
