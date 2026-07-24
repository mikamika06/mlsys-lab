def batch_formation_times(arrivals: list[int], preferred_size: int, max_queue_delay: int) -> list[tuple[int, int]]:
    """Simulate a dynamic batcher (e.g. vLLM/TGI-style continuous batching
    intake queue) that dispatches a batch whenever EITHER trigger fires
    first:

      (a) size trigger  -- the pending queue reaches `preferred_size`
          (dispatch immediately, exactly `preferred_size` items), or
      (b) delay trigger -- the OLDEST pending item has been waiting
          `max_queue_delay` time units (dispatch the ENTIRE current
          queue, however small -- a partial batch).

    arrivals        : sorted (non-decreasing) list of integer arrival
                       times; more than one request may arrive at the
                       same time. Requests are queued FIFO.
    preferred_size   : positive int, the size trigger.
    max_queue_delay  : positive int, the delay trigger (in the same time
                       units as `arrivals`).

    Returns a list of (formation_time, batch_size) tuples, in the order
    batches are formed. Every arrival ends up in exactly one batch, and
    every batch size is <= preferred_size. At a tie between the two
    triggers at the same instant, the size trigger takes priority (an
    arrival that completes a full batch fires immediately, even if the
    delay cap for the oldest item would also fire at that exact time).
    """
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue: list[int] = []
    batches: list[tuple[int, int]] = []

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append(arrivals[i])
                i += 1
                if len(queue) == preferred_size:
                    batches.append((t, preferred_size))
                    queue = []
        else:
            t = next_timeout
            batches.append((t, len(queue)))
            queue = []

    return batches
