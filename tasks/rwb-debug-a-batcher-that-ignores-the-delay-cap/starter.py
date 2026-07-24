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
    every batch size is <= preferred_size.

    BUG: this only ever checks the size trigger. Whenever the arrival
    stream ends (or thins out) with fewer than `preferred_size` items
    still pending, that trailing partial batch is silently never
    dispatched.
    """
    arrivals = sorted(int(a) for a in arrivals)
    queue: list[int] = []
    batches: list[tuple[int, int]] = []

    for t in arrivals:
        queue.append(t)
        if len(queue) == preferred_size:
            batches.append((t, preferred_size))
            queue = []
    # BUG: no delay-cap check here -- any leftover `queue` is dropped.

    return batches
