def mean_added_queue_latency(arrivals: list[int], preferred_batch_size: int, max_queue_delay: int) -> float:
    """Mean added queue latency under a dynamic batcher.

    arrivals: sequence of integer arrival timestamps (not necessarily
        pre-sorted); ties (simultaneous arrivals) are allowed.
    preferred_batch_size: dispatch the whole current queue as soon as it
        reaches this many requests (the "size trigger").
    max_queue_delay: dispatch the whole current queue as soon as the
        oldest item in it has been waiting this long (the "delay
        trigger"), even if the queue isn't full. On an exact tie between
        the two triggers, the size trigger wins.

    Simulate the batcher event by event and return the mean, over every
    request, of (dispatch_time - arrival_time) -- i.e. how much extra
    latency batching added on top of instant dispatch.
    """
    raise NotImplementedError('your code here')
