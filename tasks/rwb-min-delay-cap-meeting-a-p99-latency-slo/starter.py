def min_cap_meeting_slo(arrivals: list[int], candidate_caps: list[int],
                         preferred_size: int, service_time: int, slo: float) -> int:
    """Sweep candidate delay caps and return the SMALLEST one (by value,
    not list position) whose resulting p99 end-to-end latency meets the
    SLO, or -1 if none of them do.

    For a given cap C, every request's latency is
    (batch_formation_time - arrival_time) + service_time, using the
    two-trigger batcher (size == preferred_size, delay == C) over
    `arrivals`. `service_time` is a fixed per-batch compute cost applied
    identically to every request. p99 is the nearest-rank order
    statistic: sort latencies ascending, take index
    ceil(0.99 * n) - 1 (0-indexed), n = len(arrivals).

    arrivals        : list of int arrival times (need not be pre-sorted).
    candidate_caps  : list of positive int candidate `max_queue_delay`
                       values (need not be pre-sorted; duplicates
                       possible).
    preferred_size  : positive int, the batcher's size trigger.
    service_time    : non-negative int, fixed compute time per batch.
    slo             : float, the p99 latency budget.

    Returns the smallest passing cap value, or -1 if no candidate meets
    the SLO.
    """
    raise NotImplementedError('your code here')
