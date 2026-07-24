def _batch_sizes(arrivals: list[int], preferred_size: int, max_queue_delay: int) -> list[int]:
    """Same two-trigger dynamic batcher (size trigger == preferred_size,
    delay trigger == max_queue_delay, size wins on an exact tie); returns
    just the list of formed batch sizes, in formation order."""
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue: list[int] = []
    sizes: list[int] = []

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append(arrivals[i])
                i += 1
                if len(queue) == preferred_size:
                    sizes.append(preferred_size)
                    queue = []
        else:
            sizes.append(len(queue))
            queue = []

    return sizes


def throughput_vs_cap_curve(arrivals: list[int], caps: list[int],
                             preferred_size: int, service_time: float) -> dict:
    """No wall-clock: `service_time` is a FIXED model constant charged
    once per formed batch (independent of that batch's size), the way a
    GPU forward pass costs roughly the same whether the batch is 90% or
    100% full. For each cap, simulate the two-trigger batcher over
    `arrivals` (size trigger == preferred_size, delay trigger == cap) and
    derive:

      mean_batch_size = N / num_batches
                       (N = len(arrivals), num_batches = batches formed)
      throughput      = mean_batch_size / service_time
                       (requests served per unit of service time)

    arrivals       : list of int arrival times.
    caps           : list of positive int candidate max_queue_delay
                      values, order preserved in the output.
    preferred_size : positive int, the batcher's size trigger.
    service_time   : positive float, fixed cost per formed batch.

    Returns {"mean_batch_size": list[float], "throughput": list[float]},
    each the same length and order as `caps`.
    """
    n = len(arrivals)
    mean_batch_size = []
    throughput = []
    for cap in caps:
        sizes = _batch_sizes(arrivals, preferred_size, int(cap))
        num_batches = len(sizes)
        mbs = (n / num_batches) if num_batches else 0.0
        mean_batch_size.append(mbs)
        throughput.append(mbs / service_time)
    return {"mean_batch_size": mean_batch_size, "throughput": throughput}
