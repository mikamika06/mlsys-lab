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
    raise NotImplementedError('your code here')
