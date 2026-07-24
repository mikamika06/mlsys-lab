def dynamic_batcher_simulate(arrivals, preferred_batch_size, max_queue_delay):
    """
    arrivals: sorted (non-decreasing) sequence of integer arrival
        timestamps; ties (same timestamp) allowed, requests are FIFO.
    preferred_batch_size: dispatch a full batch as soon as the queue
        reaches this many pending requests.
    max_queue_delay: dispatch the entire current queue (a partial batch,
        possibly smaller than preferred_batch_size) as soon as the oldest
        queued item has waited this long. On a tie between the two
        triggers at the same timestamp, the size trigger wins.

    Returns (batch_formation_times, batch_sizes): two equal-length
    sequences, in dispatch order. sum(batch_sizes) == len(arrivals).
    """
    raise NotImplementedError('your code here')
