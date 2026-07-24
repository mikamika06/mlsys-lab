import numpy as np


def mean_added_queue_latency(arrivals, preferred_batch_size: int, max_queue_delay: int) -> float:
    """
    Simulate the dynamic batcher (size trigger: queue reaches
    preferred_batch_size; delay trigger: oldest queued item has waited
    max_queue_delay -- on an exact tie the size trigger wins) and return
    the mean, over all requests, of (dispatch_time - arrival_time).
    """
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue = []  # list of (arrival_time, original_index)
    dispatch = [None] * n

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0][0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append((arrivals[i], i))
                i += 1
                if len(queue) == preferred_batch_size:
                    for _, idx in queue:
                        dispatch[idx] = t
                    queue = []
        else:
            t = next_timeout
            for _, idx in queue:
                dispatch[idx] = t
            queue = []

    latencies = [d - a for a, d in zip(arrivals, dispatch)]
    return float(sum(latencies) / len(latencies))
